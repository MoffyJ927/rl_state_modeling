# Must be set before ANY torch import (macOS OpenMP libomp conflict)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
RL State Transition Risk Model

A PPO-based actor-critic architecture for long time-window
state transition risk assessment.

Unlike the self-learning variants, here the agent learns to:
  1. Dynamically allocate trust across sub-models at each timestep
  2. Decide when to trigger risk alerts

Reward is defined by alert accuracy, weight confidence, and timing.

Modules:
  config  – hyperparameters + data utilities
  model/  – environment, feature extraction, actor-critic, buffer
  train   – PPO training pipeline
  predict – inference wrapper

Quick start:
    from rl_git.config import Config, generate_synthetic_data
    from rl_git.train import PPOTrainer
    from rl_git.predict import Predictor

    cfg = Config()
    X, y, tp = generate_synthetic_data(100, cfg)

    trainer = PPOTrainer(cfg, device='cpu')
    trainer.train(X, y, tp, n_episodes=50)
    trainer.save('model.pt')

    pred = Predictor.load('model.pt')
    result = pred.predict(X[0])
"""
