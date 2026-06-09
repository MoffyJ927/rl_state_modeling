"""
Inference / prediction for the RL State Transition Risk Model.

Usage:
    from rl_state_modeling.predict import Predictor
    pred = Predictor.load('checkpoint.pt', device='cpu')
    result = pred.predict(behavior_sequence)          # single sample
    results = pred.predict_batch(behavior_sequences)  # batch
"""

import torch
import os
import numpy as np
from typing import Dict, List, Optional

from .config import Config
from .model import RLStateTransitionRiskModel, RiskEnvironment


class Predictor:
    """Wraps a trained model for inference-only use."""

    def __init__(self, model: RLStateTransitionRiskModel,
                 config: Config, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.config = config
        self.device = torch.device(device)
        self.env = RiskEnvironment(config, device)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'Predictor':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config: Config = ckpt['config']
        model = RLStateTransitionRiskModel(config)
        model.actor.load_state_dict(ckpt['actor'])
        model.critic.load_state_dict(ckpt['critic'])
        model.state_compression.load_state_dict(ckpt['state_compression'])
        model.multi_sub_models.load_state_dict(ckpt['multi_sub_models'])
        return cls(model, config, device)

    def predict(self, behavior_sequence: np.ndarray) -> Dict:
        """
        Run inference on a single behavioral sequence.

        Returns a dict with:
          decisions     – list of per-timestep decisions
          risk_scores   – (score_window,) weighted risk per timestep
        """
        tp_labels = np.zeros(self.config.score_window)
        state = self.env.reset(behavior_sequence, 0, tp_labels)

        decisions: List[Dict] = []
        risk_scores: List[float] = []

        for step in range(self.config.score_window):
            cur_seq = state['current_seq']
            sm_out = self.model.get_sub_model_outputs(cur_seq)
            state['sub_model_outputs'] = sm_out[0, -1, :].unsqueeze(0)

            with torch.no_grad():
                action = self.model.act(state)

            sw = action['sub_weights'].squeeze().cpu().numpy()
            ap = action['alert_prob'].item()
            alerted = int(torch.rand(1).item() < ap)

            sm_last = sm_out[0, -1, :].detach().cpu().numpy()
            rs = float(np.dot(sm_last, sw))

            decisions.append({
                'alerted': alerted,
                'alert_prob': ap,
                'sub_weights': sw,
                'risk_score': rs,
            })
            risk_scores.append(rs)

            self.env.step({
                'sub_weights': torch.FloatTensor(sw).to(self.device),
                'alert_prob': torch.FloatTensor([ap]).to(self.device),
                'alerted': alerted,
            })
            if state['step'] >= self.config.score_window:
                break

        return {
            'decisions': decisions,
            'risk_scores': np.array(risk_scores),
        }

    def predict_batch(self, behavior_sequences: np.ndarray) -> List[Dict]:
        """Run inference on a batch of sequences."""
        return [self.predict(seq) for seq in behavior_sequences]


# ==========================================
# Quick demo
# ==========================================

def main():
    # Import package first → triggers OMP fix in __init__.py
    from . import config as _pkg_init
    from .config import Config, generate_synthetic_data
    from .train import PPOTrainer

    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Generate demo data
    seq, labels, tp = generate_synthetic_data(50, config)

    # Quick training
    trainer = PPOTrainer(config, device)
    trainer.train(seq, labels, tp, n_episodes=10)
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint.pt')
    trainer.save(ckpt_path)

    # Inference demo
    predictor = Predictor.load(ckpt_path, device)
    result = predictor.predict(seq[0])

    n_alerts = sum(1 for d in result['decisions'] if d['alerted'])
    print(f"\nInference on sample 0:")
    print(f"  Timesteps : {len(result['decisions'])}")
    print(f"  Alerts    : {n_alerts}")
    print(f"  Avg risk  : {result['risk_scores'].mean():.4f}")

    # Cleanup demo checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == '__main__':
    main()
