"""
Demo Plotter — 绘制 GT vs Predicted 动作对比图。

输入: GT action vs Predicted action
输出: JPEG 对比图 + MSE/MAE 指标
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    action_keys: Optional[list] = None,
    traj_id: int = 0,
    save_path: str = "output/traj_0.jpeg",
) -> dict:
    """
    绘制动作对比图并保存。

    Args:
        gt_actions: (T, action_dim) Ground Truth
        pred_actions: (T, action_dim) 模型预测
        action_keys: 动作维度名称列表
        traj_id: 轨迹 ID
        save_path: 保存路径

    Returns:
        metrics: {"mse": float, "mae": float, "save_path": str}
    """
    action_dim = gt_actions.shape[1]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 生成默认 key
    if action_keys is None:
        action_keys = [f"dim_{i}" for i in range(action_dim)]

    # 绘图
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    for i in range(action_dim):
        ax = axes[i]
        label = action_keys[i] if i < len(action_keys) else f"dim_{i}"
        ax.plot(gt_actions[:, i], "g-", label="GT", linewidth=1.5)
        ax.plot(pred_actions[:, i], "r--", label="Pred", linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))
    fig.suptitle(f"Trajectory {traj_id} | MSE={mse:.6f} | MAE={mae:.6f}", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"mse": float(mse), "mae": float(mae), "save_path": str(save_path)}
