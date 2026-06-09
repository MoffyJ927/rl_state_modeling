"""
RLStateTransitionRiskModel — end-to-end Actor-Critic model.

Preserves shared feature extraction (compression + sub-models) and adds
Actor/Critic heads that decide sub-model weights + alert timing via RL.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from ..config import Config
from .shared import StateCompressionModel, MultiSubModels
from .actor_critic import Actor, Critic


class RLStateTransitionRiskModel(nn.Module):
    """Full Actor-Critic model for RL-based risk assessment."""

    def __init__(self, config: Config):
        super().__init__()
        self.state_compression = StateCompressionModel(config)
        self.multi_sub_models = MultiSubModels(config)
        self.actor = Actor(config)
        self.critic = Critic(config)

    def get_sub_model_outputs(self, seq: torch.Tensor) -> torch.Tensor:
        """Forward pass through compression + sub-models."""
        return self.multi_sub_models(self.state_compression(seq))

    def act(self, state: Dict) -> Dict[str, torch.Tensor]:
        """Select actions (for rollout / inference)."""
        sub_weights, alert_logit = self.actor(state)
        return {
            'sub_weights': sub_weights,
            'alert_prob': torch.sigmoid(alert_logit),
            'alert_logit': alert_logit,
        }

    def forward(
        self,
        current_seq: torch.Tensor,
        sub_model_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DataParallel-compatible forward for PPO update batches.

        Takes flat tensors (not dict) so DataParallel can auto-scatter across GPUs.
        Returns only tensors (no distributions) so torch.jit.trace / tensorboard works.

        Returns:
            sub_weights : (B, num_sub_models)
            alert_logit : (B, 1)  logits for Bernoulli alert
            value       : (B,)    critic state-value estimate
        """
        state = {
            'current_seq': current_seq,
            'sub_model_outputs': sub_model_outputs,
        }
        sub_weights, alert_logit = self.actor(state)
        value = self.critic(state)
        return sub_weights, alert_logit, value.squeeze(-1)

    def evaluate(
        self, state: Dict, actions: Dict = None,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution, torch.Tensor]:
        """Single-sample evaluate (for non-batched use / backward compat)."""
        sub_weights, alert_logit, value = self.forward(
            state['current_seq'],
            state.get('sub_model_outputs'),
        )
        alert_dist = torch.distributions.Bernoulli(logits=alert_logit)
        return sub_weights, alert_dist, value
