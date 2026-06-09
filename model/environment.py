"""
RiskEnvironment: simulates the risk scenario for RL training.

Each episode = one user's behavioral sequence (126 days).
The agent decides at each timestep t (1~120):
  1. Allocate trust weights across sub-models
  2. Whether to trigger a risk alert

Reward is computed at episode end based on alert accuracy and confidence.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

from ..config import Config


class RiskEnvironment:
    """Simulated risk assessment environment for RL."""

    def __init__(self, config: Config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)

    def reset(
        self,
        behavior_sequence: np.ndarray,
        true_label: int,
        timepoint_labels: np.ndarray,
    ) -> Dict:
        """Initialize one episode."""
        self.behavior = torch.FloatTensor(behavior_sequence).to(self.device)
        self.true_label = true_label
        self.timepoint_labels = torch.FloatTensor(timepoint_labels).to(self.device)
        self.current_step = 0
        self.decisions: List[bool] = []
        self.weights_history: List[np.ndarray] = []
        return self._get_state()

    def _get_state(self) -> Dict:
        current_seq = self.behavior[:self.current_step + 1].unsqueeze(0)
        return {
            'current_seq': current_seq,
            'step': self.current_step,
            'seq_len': self.config.seq_len,
            'score_window': self.config.score_window,
        }

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        sub_weights = action['sub_weights']
        alert_prob = action['alert_prob'].item()
        risk_score = action.get('risk_score', None)

        alerted = torch.rand(1).item() < alert_prob
        self.decisions.append(alerted)
        self.weights_history.append(sub_weights.detach().cpu().numpy())

        # ---- intermediate per-step reward ----
        step_reward = 0.0
        if risk_score is not None and self.current_step < len(self.timepoint_labels):
            # risk_score: weighted sum of sub-model outputs by actor's weights
            # timepoint_labels[t]: true risk intensity at this timestep
            true_risk = self.timepoint_labels[self.current_step].item()
            # negative L1 loss → higher reward when prediction matches truth
            step_reward = -abs(risk_score - true_risk) * self.config.step_reward_coef

        self.current_step += 1
        done = self.current_step >= self.config.score_window
        episode_reward = self._compute_episode_reward() if done else 0.0
        reward = step_reward + episode_reward

        return (
            self._get_state(), reward, done,
            {'alerted': alerted, 'step': self.current_step,
             'sub_weights': sub_weights.detach().cpu().numpy(),
             'step_reward': step_reward,
             'episode_reward': episode_reward},
        )

    def _compute_episode_reward(self) -> float:
        total = 0.0
        n_alerts = sum(self.decisions)

        # Alert accuracy
        if self.true_label == 1:
            total += 2.0 if n_alerts > 0 else -3.0
        else:
            total -= 1.0 * n_alerts

        # Confidence bonus
        w = np.array(self.weights_history)
        mean_w = w.mean(axis=0)
        entropy = -np.sum(mean_w * np.log(mean_w + 1e-8))
        total += (1.0 - entropy / np.log(self.config.num_sub_models)) * 0.5

        # Timing bonus
        if n_alerts > 0 and self.true_label == 1:
            first = self.decisions.index(True)
            total += max(0, 1.0 - first / self.config.score_window) * 0.5

        return total
