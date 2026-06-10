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
from typing import Dict, Any, List, Optional
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

_DIST_GENERATORS = {
    'normal':   lambda rng, shape, p: rng.normal(p.get('loc', 0), p.get('scale', 1), shape),
    'uniform':  lambda rng, shape, p: rng.uniform(p.get('low', 0), p.get('high', 1), shape),
    'bernoulli': lambda rng, shape, p: rng.binomial(1, p.get('p', 0.5), shape),
    'poisson':  lambda rng, shape, p: rng.poisson(p.get('lam', 1), shape),
    'exponential': lambda rng, shape, p: rng.exponential(p.get('scale', 1), shape),
    'categorical': lambda rng, shape, p: rng.choice(p['categories'], shape, p=p.get('probs')),
    'constant': lambda rng, shape, p: np.full(shape, p.get('value', 0)),
}


def generate_synthetic_data(
    n_samples: int,
    config: Config,
    seed: int = 42,
    feature_dim: Optional[int] = None,
    col_spec: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """
    Generate synthetic behavioral sequences + labels for demo / testing.

    Parameters
    ----------
    n_samples : int
    config : Config
    seed : int
    feature_dim : int, optional
        Override config.feature_dim. If None, use config.feature_dim.
    col_spec : list of dict, optional
        Per-column distribution specification. Columns not listed default to normal.
        Example (128 features, all 7 distributions):
            col_spec = [
                {"indices": [0, 1, 2], "dist": "bernoulli", "params": {"p": 0.3}},
                {"indices": [3, 4],    "dist": "uniform", "params": {"low": -1, "high": 1}},
                {"indices": [5, 6],    "dist": "poisson", "params": {"lam": 5}},
                {"indices": [7],       "dist": "exponential", "params": {"scale": 2}},
                {"indices": [8, 9],    "dist": "categorical", "params": {"categories": [0, 1, 2], "probs": [0.1, 0.3, 0.6]}},
                {"indices": [10],      "dist": "constant", "params": {"value": 0}},
                {"indices": [11, 12],  "dist": "normal", "params": {"loc": 0, "scale": 1}},
            ]
        Supported distributions and their params:
          'normal'      — {'loc': 0, 'scale': 1}
          'uniform'     — {'low': 0, 'high': 1}
          'bernoulli'   — {'p': 0.5}
          'poisson'     — {'lam': 1}
          'exponential' — {'scale': 1}
          'categorical' — {'categories': [a, b, c], 'probs': [...]}  # probs 可选，默认均匀
          'constant'    — {'value': 0}

    Returns
    -------
        behavior_sequences: (n_samples, seq_len, feature_dim)
        true_labels:        (n_samples,)  0/1 final risk event
        timepoint_labels:   (n_samples, score_window) per-timestep risk state
    """
    rng = np.random.RandomState(seed)
    fd = feature_dim if feature_dim is not None else config.feature_dim

    # Build per-column generator lookup: col_idx → (generator_fn, params)
    col_gen = {}
    if col_spec:
        for entry in col_spec:
            gen_fn = _DIST_GENERATORS[entry['dist']]
            params = entry.get('params', {})
            for idx in entry['indices']:
                col_gen[idx] = (gen_fn, params)

    # Generate feature by feature
    # shape for one column: (n_samples, seq_len)
    col_shape = (n_samples, config.seq_len)
    cols = []
    for c in range(fd):
        if c in col_gen:
            fn, params = col_gen[c]
            col = fn(rng, col_shape, params)
        else:
            col = rng.randn(*col_shape)
        cols.append(col.astype(np.float32))

    behavior_sequences = np.stack(cols, axis=-1)  # (n_samples, seq_len, feature_dim)

    true_labels = rng.randint(0, 2, n_samples).astype(np.float32)
    timepoint_labels = rng.rand(n_samples, config.score_window).astype(np.float32)

    return behavior_sequences, true_labels, timepoint_labels
