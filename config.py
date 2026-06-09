"""
Configuration and data utilities for the RL State Transition Risk Model.

Usage:
    config = Config()                           # defaults
    config = Config(hidden_dim=256)             # single override via dataclass
    config = Config.update_from_args(args)      # from argparse namespace
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any
import argparse


@dataclass
class Config:
    """All hyperparameters and dimensional settings."""

    # ---- Sequence ----
    seq_len: int = 126           # Day 1 ~ Day 126
    feature_dim: int = 32        # daily behavior feature dimension
    hidden_dim: int = 128        # LSTM hidden dimension

    # ---- Sub-models ----
    num_sub_models: int = 8      # number of parallel sub-models
    state_dim: int = 64          # per sub-model output dimension

    # ---- Scoring ----
    score_window: int = 120      # evaluation window (Day 1 ~ Day 120)

    # ---- RL ----
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE λ
    entropy_coef: float = 0.01   # exploration bonus
    value_coef: float = 0.5      # critic loss weight
    clip_range: float = 0.2      # PPO clip ε
    step_reward_coef: float = 0.1  # intermediate per-step reward weight

    # ---- Training ----
    learning_rate: float = 3e-4
    l2_reg: float = 1e-4
    update_epochs: int = 2       # PPO update epochs per buffer
    trajs_per_update: int = 4    # trajectories to collect before each PPO update
    n_steps: int = 120           # decision steps per episode

    # ---- CLI alias map: argparse flag → Config field name ----
    # Only needed when the flag differs from the field name.
    _CLI_ALIASES: Dict[str, str] = None  # set below

    @classmethod
    def cli_alias_map(cls) -> Dict[str, str]:
        """Map from argparse dest name to Config field name."""
        return {
            'lr': 'learning_rate',
        }

    @classmethod
    def from_args(cls, args: argparse.Namespace, base: 'Config' = None) -> 'Config':
        """Build Config from argparse namespace.

        Any arg that matches a Config field name gets applied.
        Aliases (e.g. 'lr' → 'learning_rate') are resolved via cli_alias_map().

        Parameters
        ----------
        args : argparse.Namespace
            Parsed CLI arguments.
        base : Config, optional
            If provided, start from this Config and overlay the CLI args on top.
            Used for JSON + CLI override: Config.from_args(args, base=Config.from_json(...))
        """
        config = base if base is not None else cls()
        alias = cls.cli_alias_map()
        args_dict = vars(args)

        for field_name in cls.__dataclass_fields__:
            # Direct match
            if field_name in args_dict:
                setattr(config, field_name, args_dict[field_name])
            else:
                # Check aliases
                for arg_key, mapped_field in alias.items():
                    if mapped_field == field_name and arg_key in args_dict:
                        setattr(config, field_name, args_dict[arg_key])
                        break

        return config

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Build Config from a JSON file.

        The JSON file should be a flat dict of Config field names to values.
        Example config.json:
        {
            "hidden_dim": 256,
            "learning_rate": 1e-4,
            "n_steps": 120
        }
        """
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        valid_fields = set(cls.__dataclass_fields__)
        alias = cls.cli_alias_map()
        # reverse alias map: field_name → arg_key
        rev_alias = {v: k for k, v in alias.items()}
        for key, value in data.items():
            # direct field name
            if key in valid_fields:
                setattr(config, key, value)
            # CLI alias (e.g. "lr" → "learning_rate")
            elif key in alias:
                setattr(config, alias[key], value)
            # field name used as CLI arg (e.g. "learning_rate" could be "lr" in JSON)
            elif key in rev_alias:
                setattr(config, key, value)
            else:
                print(f"[Config.from_json] Warning: unknown key '{key}', ignored.")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export all fields as plain dict."""
        return {k: v for k, v in asdict(self).items()
                if not k.startswith('_')}

    def summary(self) -> str:
        """Human-readable one-line summary."""
        d = self.to_dict()
        return (
            f"feature={d['feature_dim']}, hidden={d['hidden_dim']}, seq={d['seq_len']}"
        )

    # ---- argparse builder (kept here so schema lives with Config) ----

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Register all Config fields as CLI flags on an existing parser.
        Call from app.py after creating the parser.
        """
        d = Config()  # defaults

        g = parser.add_argument_group('Model Architecture')
        g.add_argument('--feature_dim', type=int, default=d.feature_dim)
        g.add_argument('--hidden_dim', type=int, default=d.hidden_dim)
        g.add_argument('--seq_len', type=int, default=d.seq_len)
        g.add_argument('--num_sub_models', type=int, default=d.num_sub_models)
        g.add_argument('--state_dim', type=int, default=d.state_dim)
        g.add_argument('--score_window', type=int, default=d.score_window)

        g = parser.add_argument_group('RL Hyperparameters')
        g.add_argument('--gamma', type=float, default=d.gamma)
        g.add_argument('--gae_lambda', type=float, default=d.gae_lambda)
        g.add_argument('--entropy_coef', type=float, default=d.entropy_coef)
        g.add_argument('--value_coef', type=float, default=d.value_coef)
        g.add_argument('--clip_range', type=float, default=d.clip_range)
        g.add_argument('--step_reward_coef', type=float, default=d.step_reward_coef)

        g = parser.add_argument_group('Training')
        g.add_argument('--lr', type=float, default=d.learning_rate)
        g.add_argument('--l2_reg', type=float, default=d.l2_reg)
        g.add_argument('--n_steps', type=int, default=d.n_steps)
        g.add_argument('--trajs_per_update', type=int, default=d.trajs_per_update)
        g.add_argument('--update_epochs', type=int, default=d.update_epochs)


# ==========================================
# Data utilities
# ==========================================

def generate_synthetic_data(
    n_samples: int,
    config: Config,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic behavioral sequences + labels for demo / testing.

    Returns:
        behavior_sequences: (n_samples, seq_len, feature_dim)
        true_labels:        (n_samples,)  0/1 final risk event
        timepoint_labels:   (n_samples, score_window) per-timestep risk state
    """
    rng = np.random.RandomState(seed)

    behavior_sequences = rng.randn(
        n_samples, config.seq_len, config.feature_dim
    ).astype(np.float32)

    true_labels = rng.randint(0, 2, n_samples).astype(np.float32)
    timepoint_labels = rng.rand(n_samples, config.score_window).astype(np.float32)

    return behavior_sequences, true_labels, timepoint_labels


def generate_self_supervised_labels(
    behavior_sequences: np.ndarray,
    config: Config,
) -> tuple:
    """
    Generate self-supervised pseudo-labels based on behavior change intensity.

    Strategy:
      - Compute L2 norm of adjacent timestep differences
      - High change → higher risk probability

    Returns:
        final_labels:      (n_samples,)  binary final risk label
        timepoint_labels:  (n_samples, score_window)  per-timestep risk state
    """
    batch_size, seq_len, feature_dim = behavior_sequences.shape

    # Adjacent timestep differences
    diffs = np.diff(behavior_sequences, axis=1)  # (batch, seq_len-1, feature_dim)

    # L2 norm as change intensity
    change_intensity = np.linalg.norm(diffs, axis=-1)  # (batch, seq_len-1)

    # Normalize to [0, 1]
    min_v = change_intensity.min(axis=1, keepdims=True)
    max_v = change_intensity.max(axis=1, keepdims=True)
    change_intensity = (change_intensity - min_v) / (max_v + 1e-8)

    # Final label: whether max change exceeds threshold
    final_labels = (change_intensity.max(axis=1) > 0.5).astype(np.float32)

    # Timepoint labels: truncate/pad to score_window
    sw = config.score_window
    if seq_len - 1 < sw:
        padded = np.zeros((batch_size, sw), dtype=np.float32)
        padded[:, :seq_len - 1] = change_intensity
        timepoint_labels = padded
    else:
        timepoint_labels = change_intensity[:, :sw].astype(np.float32)

    return final_labels, timepoint_labels
