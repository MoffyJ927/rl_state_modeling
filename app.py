"""
完整流程应用：RL State Transition Risk Model

运行整个 pipeline：
  1. 生成伪造数据
  2. PPO 训练
  3. 推理预测
  4. 可视化 + CSV 导出

Usage:
    cd sequential/rl_git
    python app.py                                    # 默认参数
    python app.py --n_episodes 200 --lr 1e-4         # 覆盖参数
    python app.py --help                              # 查看所有参数
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SELF_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl_git.config import Config, generate_synthetic_data, generate_self_supervised_labels
from rl_git.train import PPOTrainer
from rl_git.predict import Predictor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='RL State Transition Risk Model — 完整流程',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='\nExamples:\n'
               '  python app.py\n'
               '  python app.py --n_samples 1000 --n_episodes 200 --lr 1e-4\n'
               '  python app.py --gpu_ids 0 1 2 3\n',
    )
    # ---- Data ----
    g = p.add_argument_group('Data')
    g.add_argument('--n_samples', type=int, default=200)
    g.add_argument('--seed', type=int, default=42)
    # ---- Training loop ----
    g = p.add_argument_group('Training Loop')
    g.add_argument('--n_episodes', type=int, default=80)
    # ---- Model config (defined in Config.add_argparse_args) ----
    Config.add_argparse_args(p)
    # ---- Hardware ----
    g = p.add_argument_group('Hardware')
    g.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    g.add_argument('--gpu_ids', type=int, nargs='*', default=None,
                   help='GPU IDs for DataParallel, e.g. 0 1 2 3')
    # ---- Output ----
    g = p.add_argument_group('Output')
    g.add_argument('--output_dir', type=str, default=None)
    g.add_argument('--no_plot', action='store_true')
    g.add_argument('--no_csv', action='store_true')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(_SELF_DIR, 'output_demo')
    os.makedirs(output_dir, exist_ok=True)

    # ---- Config from CLI args ----
    config = Config.from_args(args)

    print("=" * 60)
    print("  RL State Transition Risk Model — 完整流程")
    print("=" * 60)
    print(f"  Config: {config.summary()}")
    print(f"  Data  : n_samples={args.n_samples}, n_episodes={args.n_episodes}")
    print(f"  Device: {args.device}"
          + (f", DataParallel GPUs {args.gpu_ids}" if args.gpu_ids and len(args.gpu_ids) > 1 else ""))
    print("─" * 60)

    # ============================================================
    # 1. 生成伪造数据
    # ============================================================
    print("\n[Step 1] 生成伪造数据 ...")
    behaviors, true_labels, tp_labels = generate_synthetic_data(
        args.n_samples, config, seed=args.seed)
    print(f"  行为序列: {behaviors.shape}")
    print(f"  正样本占比: {true_labels.mean():.2%}")
    ssl_labels, ssl_tp = generate_self_supervised_labels(behaviors, config)
    print(f"  自监督正样本占比: {ssl_labels.mean():.2%}")

    # ============================================================
    # 2. 训练
    # ============================================================
    print(f"\n[Step 2] PPO 训练 ...")
    trainer = PPOTrainer(config, device=args.device, gpu_ids=args.gpu_ids)
    history = trainer.train(behaviors, true_labels, tp_labels,
                            n_episodes=args.n_episodes)

    rewards = [h['reward'] for h in history]
    a_losses = [h['actor_loss'] for h in history]
    c_losses = [h['critic_loss'] for h in history]
    entropies = [h['entropy'] for h in history]

    print(f"\n  训练完成!")
    print(f"  前10轮平均 Reward: {np.mean(rewards[:10]):.3f}")
    print(f"  后10轮平均 Reward: {np.mean(rewards[-10:]):.3f}")
    print(f"  最终 Actor/Critic Loss: {a_losses[-1]:.4f} / {c_losses[-1]:.4f}")

    # ============================================================
    # 3. 保存模型
    # ============================================================
    ckpt_path = os.path.join(output_dir, 'checkpoint.pt')
    trainer.save(ckpt_path)

    # ============================================================
    # 4. 推理预测
    # ============================================================
    print(f"\n[Step 3] 推理预测 ...")
    predictor = Predictor.load(ckpt_path, device=args.device)
    results = predictor.predict_batch(behaviors)

    summary = []
    for i, r in enumerate(results):
        decisions = r['decisions']
        summary.append({
            'sample': i,
            'true_label': int(true_labels[i]),
            'n_alerts': sum(1 for d in decisions if d['alerted']),
            'avg_risk': r['risk_scores'].mean(),
            'max_risk': r['risk_scores'].max(),
            'avg_alert_prob': np.mean([d['alert_prob'] for d in decisions]),
        })

    pos = [s for s in summary if s['true_label'] == 1]
    neg = [s for s in summary if s['true_label'] == 0]
    print(f"\n  预测汇总:")
    print(f"  {'':<25} {'正样本 (label=1)':>20} {'负样本 (label=0)':>20}")
    print(f"  {'样本数':<25} {len(pos):>20} {len(neg):>20}")
    print(f"  {'平均预警次数':<25} {np.mean([s['n_alerts'] for s in pos]):>20.1f} {np.mean([s['n_alerts'] for s in neg]):>20.1f}")
    print(f"  {'平均风险分数':<25} {np.mean([s['avg_risk'] for s in pos]):>20.4f} {np.mean([s['avg_risk'] for s in neg]):>20.4f}")

    # ============================================================
    # 5. 可视化
    # ============================================================
    if not args.no_plot:
        print(f"\n[Step 4] 生成图表 ...")
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('RL State Transition Risk Model — Results', fontsize=14, fontweight='bold')

        # (a) Reward
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, color='steelblue')
        if len(rewards) >= 10:
            ax.plot(range(9, len(rewards)),
                    np.convolve(rewards, np.ones(10) / 10, mode='valid'),
                    color='darkblue', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set(xlabel='Episode', ylabel='Reward', title='Training Reward')

        # (b) Loss
        ax = axes[0, 1]
        ax.plot(a_losses, alpha=0.6, label='Actor', color='#E65100')
        ax.plot(c_losses, alpha=0.6, label='Critic', color='#6A1B9A')
        ax.set(xlabel='Episode', ylabel='Loss', title='Actor & Critic Loss')
        ax.legend(fontsize=8)

        # (c) Entropy
        ax = axes[0, 2]
        ax.plot(entropies, color='#2E7D32')
        ax.set(xlabel='Episode', ylabel='Entropy', title='Policy Entropy')

        # (d) Alert distribution
        ax = axes[1, 0]
        pos_a = [s['n_alerts'] for s in pos]
        neg_a = [s['n_alerts'] for s in neg]
        n_max = max(max(pos_a, default=0), max(neg_a, default=0))
        bins = np.linspace(0, n_max + 1, min(15, n_max + 2))
        ax.hist(pos_a, bins=bins, alpha=0.6, label=f'Positive (n={len(pos)})', color='#C62828')
        ax.hist(neg_a, bins=bins, alpha=0.6, label=f'Negative (n={len(neg)})', color='#1565C0')
        ax.set(xlabel='Alert Count', title='Alert Count Distribution')
        ax.legend(fontsize=8)

        # (e) Risk score distribution
        ax = axes[1, 1]
        ax.hist([s['avg_risk'] for s in pos], bins=20, alpha=0.6, label='Positive', color='#C62828')
        ax.hist([s['avg_risk'] for s in neg], bins=20, alpha=0.6, label='Negative', color='#1565C0')
        ax.set(xlabel='Avg Risk Score', title='Risk Score Distribution')
        ax.legend(fontsize=8)

        # (f) Single sample trajectory
        ax = axes[1, 2]
        demo_idx = pos[0]['sample'] if pos else 0
        demo = results[demo_idx]
        steps = range(len(demo['risk_scores']))
        ax.plot(steps, demo['risk_scores'], color='steelblue', linewidth=1.5, label='Risk')
        ax.plot(steps, [d['alert_prob'] for d in demo['decisions']],
                color='darkorange', linestyle='--', alpha=0.7, label='Alert Prob')
        for t in steps:
            if demo['decisions'][t]['alerted']:
                ax.axvline(x=t, color='red', alpha=0.3, linewidth=0.8)
        ax.set(xlabel='Timestep', title=f'Sample #{demo_idx} (label={true_labels[demo_idx]:.0f})')
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'results.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图表: {fig_path}")

    # ============================================================
    # 6. CSV
    # ============================================================
    if not args.no_csv:
        import csv
        csv_path = os.path.join(output_dir, 'summary.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
        print(f"  CSV: {csv_path}")

    print(f"\n{'=' * 60}")
    print(f"  完成!  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
