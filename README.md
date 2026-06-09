# RL State Transition Risk Model

基于 PPO Actor-Critic 的长时序状态转移风险评估模型。

## 核心设计

Agent 在每个 timestep 学习两个决策：

1. **子模型信任权重分配** — 动态分配 8 个并行子模型的权重
2. **风险预警触发** — 是否在当前时刻发出风险警报

Reward 由预警准确度 + 置信度 + 时机奖励构成。每步由 Actor 输出的加权风险分数与真实标签的 L1 对齐提供即时间奖励，解决稀疏奖励问题。

## 目录结构

```
rl_git/
├── config.py               # 超参数配置 + 数据生成工具 + argparse 注册
├── train.py                # PPO 训练器（CPU / 单GPU / DataParallel 多GPU）
├── predict.py              # 推理接口 Predictor
├── app.py                  # 完整流程 CLI 入口
├── model/
│   ├── shared.py           # StateCompression + MultiSubModels
│   ├── actor_critic.py     # Actor (策略) + Critic (价值)
│   ├── environment.py      # RiskEnvironment (RL 环境 + 奖励函数)
│   ├── buffer.py           # RolloutBuffer (PPO 轨迹存储)
│   └── model.py            # RLStateTransitionRiskModel (端到端)
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install torch numpy matplotlib
```

### 跑完整流程

```bash
cd rl_git
python app.py
```

### 覆盖超参数

```bash
python app.py --n_episodes 200 --lr 1e-4 --hidden_dim 256
python app.py --step_reward_coef 0.2 --entropy_coef 0.02
python app.py --gpu_ids 0 1 2 3         # 多GPU DataParallel
python app.py --help                     # 查看所有参数
```

### 作为 Python 包使用

```python
from rl_git.config import Config, generate_synthetic_data
from rl_git.train import PPOTrainer
from rl_git.predict import Predictor

config = Config(hidden_dim=256, lr=1e-4)
X, y, tp = generate_synthetic_data(200, config)

# 训练
trainer = PPOTrainer(config, device='cuda', gpu_ids=[0, 1])
trainer.train(X, y, tp, n_episodes=100)
trainer.save('checkpoint.pt')

# 推理
pred = Predictor.load('checkpoint.pt')
result = pred.predict(X[0])
print(result['risk_scores'])
```

## 模型架构

```
Input (B, 126, 32)          ← 每日行为特征序列
    ↓
State Compression           ← BiLSTM×2 + Tanh
    ↓
Multi Sub-Models ×8         ← 8× (Conv1D → LayerNorm → Sigmoid)
    ↓               ↓
Actor π(a|s)        Critic V(s)
  - sub_weights        - scalar value
  - alert_logit
    ↓
Risk Score → PPO Reward → Gradient ↩
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dim` | 128 | LSTM 隐藏维度 |
| `num_sub_models` | 8 | 并行子模型数量 |
| `n_steps` | 120 | 每 episode 决策步数 |
| `trajs_per_update` | 4 | 每次攒多少条轨迹后 PPO update |
| `step_reward_coef` | 0.1 | 中间奖励系数（越大引导越强） |
| `entropy_coef` | 0.01 | 策略熵系数（探索强度） |
| `gamma` | 0.99 | 折扣因子 |
| `clip_range` | 0.2 | PPO clip epsilon |

## 输出

运行 `app.py` 后在 `output_demo/` 生成：

- `checkpoint.pt` — 模型权重
- `results.png` — 训练曲线 + 预测分布可视化
- `summary.csv` — 每个样本的预测汇总

## 多 GPU

```python
# 指定 GPU
trainer = PPOTrainer(config, device='cuda', gpu_ids=[0, 1, 2, 3])

# 或自动检测所有 GPU（不指定 gpu_ids）
trainer = PPOTrainer(config, device='cuda')
```

## License

MIT
