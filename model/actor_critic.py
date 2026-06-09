"""
Actor-Critic networks for the RL State Transition Risk Model.

  Actor  — policy network: outputs sub_weights (softmax) + alert_logit
  Critic — value network: estimates V(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..config import Config


FEAT = 256  # hidden_dim * 2 (bidirectional)


class Actor(nn.Module):
    """Policy network: sub-model trust weights + alert trigger logit."""

    def __init__(self, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(config.feature_dim, config.hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

        self.weight_head = nn.Sequential(
            nn.Linear(FEAT, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_sub_models),
        )
        self.alert_head = nn.Sequential(
            nn.Linear(FEAT, config.hidden_dim // 2), nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.sm_proj = nn.Linear(config.num_sub_models, FEAT)

    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = state['current_seq']          # (B, t+1, feature_dim)
        h, _ = self.lstm(seq)
        h = self.dropout(h)
        last = h[:, -1, :]                  # (B, FEAT)

        if state.get('sub_model_outputs') is not None:
            last = last + self.sm_proj(state['sub_model_outputs'])

        sub_weights = F.softmax(self.weight_head(last), dim=-1)
        alert_logit = self.alert_head(last)
        return sub_weights, alert_logit


class Critic(nn.Module):
    """Value network: estimates V(s)."""

    def __init__(self, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(config.feature_dim, config.hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

        self.value_head = nn.Sequential(
            nn.Linear(FEAT, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.sm_proj = nn.Linear(config.num_sub_models, FEAT)

    def forward(self, state: Dict) -> torch.Tensor:
        seq = state['current_seq']
        h, _ = self.lstm(seq)
        h = self.dropout(h)
        last = h[:, -1, :]

        if state.get('sub_model_outputs') is not None:
            last = last + self.sm_proj(state['sub_model_outputs'])

        return self.value_head(last)
