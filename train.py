"""
PPO training pipeline for the RL State Transition Risk Model.

Supports single-GPU, multi-GPU (DataParallel), and CPU.

Usage:
    from rl_state_modeling.train import PPOTrainer

    # single GPU
    trainer = PPOTrainer(config, device='cuda')

    # multi-GPU: uses DataParallel
    trainer = PPOTrainer(config, device='cuda', gpu_ids=[0, 1, 2, 3])

    history = trainer.train(behavior_seq, labels, tp_labels, n_episodes=100)
    trainer.save('checkpoint.pt')
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

from .config import Config
from .model import (
    RLStateTransitionRiskModel,
    RiskEnvironment,
    RolloutBuffer,
)


def _gpu_info(device: torch.device) -> str:
    """Human-readable multi-GPU info string."""
    if device.type != 'cuda':
        return ''
    n_gpus = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    gpu_names = ', '.join(names[:4])
    if n_gpus > 4:
        gpu_names += f' ... (+{n_gpus - 4})'
    return f"GPU(s): {n_gpus} × [{gpu_names}]"


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) trainer with optional multi-GPU support.

    Multi-GPU strategy:
      - Model is wrapped with nn.DataParallel when gpu_ids has >1 devices
      - Trajectories are batched in update(); DataParallel splits the batch across GPUs
    """

    def __init__(self, config: Config, device: str = 'cpu',
                 gpu_ids: Optional[List[int]] = None):
        self.config = config
        self.device = torch.device(device)

        model = RLStateTransitionRiskModel(config).to(self.device)

        # --- DataParallel wrapping ---
        self.is_parallel = False
        if self.device.type == 'cuda':
            if gpu_ids is not None and len(gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=gpu_ids)
                model = model.to(self.device)
                self.is_parallel = True
                self._gpu_ids = gpu_ids
            elif torch.cuda.device_count() > 1:
                # Auto-detect: wrap with all available GPUs
                gpu_ids = list(range(torch.cuda.device_count()))
                model = nn.DataParallel(model, device_ids=gpu_ids)
                model = model.to(self.device)
                self.is_parallel = True
                self._gpu_ids = gpu_ids

        self.model = model
        self.env = RiskEnvironment(config, device)
        self.buffer = RolloutBuffer()

        # Optimizers (actor shares feature extractors)
        self.actor_opt = torch.optim.Adam(
            list(self._unwrap().actor.parameters())
            + list(self._unwrap().state_compression.parameters())
            + list(self._unwrap().multi_sub_models.parameters()),
            lr=config.learning_rate,
            weight_decay=config.l2_reg,
        )
        self.critic_opt = torch.optim.Adam(
            self._unwrap().critic.parameters(),
            lr=config.learning_rate * 2,
            weight_decay=config.l2_reg,
        )

    def _unwrap(self) -> RLStateTransitionRiskModel:
        """Unwrap DataParallel to get the raw model."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # ---- Rollout ----

    def collect_rollout(
        self,
        behavior: np.ndarray,
        true_label: int,
        tp_labels: np.ndarray,
    ) -> float:
        """Execute one episode, store experience in buffer."""
        state = self.env.reset(behavior, true_label, tp_labels)
        ep_reward = 0.0

        for _ in range(self.config.n_steps):
            cur_seq = state['current_seq']
            sm_out = self._unwrap().get_sub_model_outputs(cur_seq)
            state['sub_model_outputs'] = sm_out[0, -1, :].unsqueeze(0)

            with torch.no_grad():
                action = self._unwrap().act(state)

            alert_prob = action['alert_prob'].item()
            alerted = int(torch.rand(1).item() < alert_prob)

            # Compute weighted risk score for intermediate reward
            sw = action['sub_weights'].squeeze(0).detach().cpu().numpy()
            sm_last = sm_out[0, -1, :].detach().cpu().numpy()
            risk_score = float(np.dot(sm_last, sw))

            sampled = {
                'sub_weights': action['sub_weights'],
                'alert_prob': action['alert_prob'],
                'alerted': alerted,
                'risk_score': risk_score,
            }

            log_p = F.binary_cross_entropy_with_logits(
                action['alert_logit'].squeeze(),
                torch.tensor(alerted, dtype=torch.float, device=self.device),
                reduction='sum',
            )

            val = self._unwrap().critic(state).item()
            next_state, reward, done, _ = self.env.step(sampled)

            self.buffer.states.append(state.copy())
            self.buffer.actions.append(sampled)
            self.buffer.rewards.append(reward)
            self.buffer.values.append(val)
            self.buffer.log_probs.append(log_p.item())
            self.buffer.dones.append(done)

            ep_reward += reward
            state = next_state
            if done:
                break

        return ep_reward

    def collect_rollouts(
        self,
        behaviors: np.ndarray,
        true_labels: np.ndarray,
        tp_labels: np.ndarray,
    ) -> List[float]:
        """Collect multiple trajectories into buffer."""
        rewards = []
        for i in range(len(behaviors)):
            r = self.collect_rollout(behaviors[i], int(true_labels[i]), tp_labels[i])
            rewards.append(r)
        return rewards

    # ---- GAE ----

    @staticmethod
    def _compute_gae_slice(
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        gamma: float,
        lam: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns for one trajectory slice."""
        advantages: List[float] = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = (rewards[t] + gamma * nv * (1 - dones[t])) - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    # ---- PPO Update (batch-aware) ----

    def update(self) -> Dict[str, float]:
        """
        Batched PPO update over all buffered trajectories.

        Trajectories are grouped by step-index so that all entries
        at the same timestep share the same sequence length → can be stacked.
        """
        n_steps = self.config.n_steps
        total = self.buffer.size()
        n_trajs = total // n_steps

        # Compute GAE per trajectory
        all_advantages = []
        all_returns = []
        for t_idx in range(n_trajs):
            start = t_idx * n_steps
            end = start + n_steps
            adv, ret = self._compute_gae_slice(
                self.buffer.rewards[start:end],
                self.buffer.values[start:end],
                self.buffer.dones[start:end],
                self.config.gamma,
                self.config.gae_lambda,
            )
            all_advantages.extend(adv)
            all_returns.extend(ret)

        adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)  # (total,)
        ret_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_a, total_c, total_e = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.config.update_epochs):
            # Shuffle step indices
            step_order = torch.randperm(n_steps).tolist()
            for step in step_order:
                # Gather all trajectories at this step
                indices = [t_idx * n_steps + step for t_idx in range(n_trajs)]

                # ---- Build batch state ----
                batch_seqs = torch.stack([
                    self.buffer.states[idx]['current_seq'].squeeze(0)  # (step+1, feat)
                    for idx in indices
                ]).to(self.device)                                     # (B, step+1, feat)

                sm_out = self.model.get_sub_model_outputs(batch_seqs)   # (B, step+1, num_sub)

                # ---- Old log-probs ----
                old_lps = torch.tensor(
                    [self.buffer.log_probs[idx] for idx in indices],
                    dtype=torch.float32, device=self.device,
                )

                # ---- Alert targets ----
                alerted_batch = torch.tensor(
                    [self.buffer.actions[idx]['alerted'] for idx in indices],
                    dtype=torch.float, device=self.device,
                )

                # ---- Forward through model ----
                # Uses forward() → DataParallel auto-splits batch across GPUs
                _, alert_logit, values = self.model(
                    batch_seqs, sm_out[:, -1, :],
                )
                alert_dist = torch.distributions.Bernoulli(logits=alert_logit)
                new_lps = alert_dist.log_prob(alerted_batch).sum(dim=-1)  # (B,)

                # ---- PPO clipped surrogate ----
                ratio = torch.exp(new_lps - old_lps)
                adv_batch = adv_t[indices]  # (B,)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range,
                                    1 + self.config.clip_range) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values.squeeze(-1), ret_t[indices])

                sw, _ = self._unwrap().actor({
                    'current_seq': batch_seqs,
                    'sub_model_outputs': sm_out[:, -1, :],
                })
                entropy = -(sw * torch.log(sw + 1e-8)).sum(dim=-1).mean()

                total_loss = (actor_loss
                              + self.config.value_coef * critic_loss
                              - self.config.entropy_coef * entropy)

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()

                total_a += actor_loss.item()
                total_c += critic_loss.item()
                total_e += entropy.item()
                n_updates += 1

        return {
            'actor_loss': total_a / max(n_updates, 1),
            'critic_loss': total_c / max(n_updates, 1),
            'entropy': total_e / max(n_updates, 1),
        }

    # ---- Training loop ----

    def train(
        self,
        behavior_sequences: np.ndarray,
        true_labels: np.ndarray,
        timepoint_labels: np.ndarray,
        n_episodes: int = 100,
    ) -> List[Dict]:
        """Run the full training loop."""
        n_samples = behavior_sequences.shape[0]
        T = self.config.trajs_per_update
        n_steps = self.config.n_steps

        # ---- Device / GPU info ----
        param_device = next(self.model.parameters()).device
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model device : {param_device}")
        print(f"  Total params : {total_params:,}")
        print(f"  {_gpu_info(self.device)}")
        print(f"  n_samples    : {n_samples}")
        print(f"  n_episodes   : {n_episodes}")
        print(f"  trajs/update : {T}  (buffer = {T} × {n_steps} = {T * n_steps} steps)")
        if self.is_parallel:
            print(f"  DataParallel : enabled on GPUs {self._gpu_ids}")
        print(f"  update_epochs: {self.config.update_epochs}")
        if self.device.type == 'cuda':
            alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU mem init : {alloc:.0f} MiB")
        print()

        history: List[Dict] = []
        t_start = time.time()

        # Use a running list of episodes per update round
        ep_rewards_window: List[float] = []

        for ep in range(n_episodes):
            # Sample T trajectories
            indices = np.random.choice(n_samples, size=T, replace=True)
            traj_rewards = self.collect_rollouts(
                behavior_sequences[indices],
                true_labels[indices],
                timepoint_labels[indices],
            )

            # PPO update on all T trajectories
            t0 = time.time()
            stats = self.update()
            t_update = time.time() - t0

            buf_size = self.buffer.size()
            self.buffer.clear()

            avg_r = sum(traj_rewards) / len(traj_rewards)
            ep_rewards_window.append(avg_r)
            stats['episode'] = ep
            stats['reward'] = avg_r
            history.append(stats)

            # Print every ~10% of training
            if (ep + 1) % max(1, n_episodes // 10) == 0:
                recent_r = np.mean(ep_rewards_window[-10:])
                elapsed = time.time() - t_start
                mem_str = ''
                if self.device.type == 'cuda':
                    mem_mb = torch.cuda.memory_allocated() / 1024**2
                    mem_str = f'  GPU mem: {mem_mb:.0f} MiB'
                print(
                    f"Ep {ep+1:>4}/{n_episodes}"
                    f"  |  buf: {T:>2}×{buf_size // T}={buf_size} steps"
                    f"  |  avg_rew: {recent_r:>8.3f}"
                    f"  |  actor: {stats['actor_loss']:>8.4f}"
                    f"  critic: {stats['critic_loss']:>8.4f}"
                    f"  entr: {stats['entropy']:>6.4f}"
                    f"  |  t_update: {t_update:.2f}s"
                    f"  elapsed: {elapsed:.0f}s"
                    f"{mem_str}"
                )

        print(f"\n  Training complete.  Total time: {time.time() - t_start:.0f}s")
        return history

    # ---- Persistence ----

    def save(self, path: str):
        m = self._unwrap()
        torch.save({
            'actor': m.actor.state_dict(),
            'critic': m.critic.state_dict(),
            'state_compression': m.state_compression.state_dict(),
            'multi_sub_models': m.multi_sub_models.state_dict(),
            'config': self.config,
        }, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        m = self._unwrap()
        m.actor.load_state_dict(ckpt['actor'])
        m.critic.load_state_dict(ckpt['critic'])
        m.state_compression.load_state_dict(ckpt['state_compression'])
        m.multi_sub_models.load_state_dict(ckpt['multi_sub_models'])
        print(f"Loaded from {path}")
