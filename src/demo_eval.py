"""
Demo Eval — 本地开环推理评估。

连接云端 Policy Server，加载 demo 数据集，执行推理，保存对比图。
"""

import os
from typing import Optional

import numpy as np

from src.policy_client import GR00TClient
from src.observation_builder import ObservationBuilder
from src.demo_plotter import plot_trajectory_comparison
from src.lerobot_loader import LeRobotEpisodeLoader


def run_demo_eval(
    dataset_path: str,
    embodiment_tag: str,
    host: str = "127.0.0.1",
    port: int = 5555,
    output_dir: str = "./output",
    traj_ids: Optional[list] = None,
    action_horizon: int = 8,
):
    """
    运行 Demo 推理评估。

    Args:
        dataset_path: LeRobot 格式数据集路径
        embodiment_tag: 具身标签
        host: Policy Server 地址
        port: Policy Server 端口
        output_dir: 输出目录
        traj_ids: 要评估的轨迹 ID 列表
        action_horizon: 动作预测步数
    """
    traj_ids = traj_ids or [1, 2]

    # 连接云端
    with GR00TClient(host=host, port=port) as client:
        modality_config = client.get_modality_config()
        print(f"📋 Modality config keys: {list(modality_config.keys()) if isinstance(modality_config, dict) else type(modality_config)}")

        # 加载数据集
        dataset = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_config,
        )
        print(f"📊 数据集加载完成: {len(dataset)} episodes")

        # 构建观测构造器
        video_keys = list(modality_config.get("video", {}).keys()) if isinstance(modality_config, dict) else ["exterior_image_1_left"]
        obs_builder = ObservationBuilder(camera_keys=video_keys)

        all_mse = []
        all_mae = []

        for traj_id in traj_ids:
            if traj_id >= len(dataset):
                print(f"⚠️  traj_id={traj_id} 超出范围（共 {len(dataset)} episodes），跳过")
                continue

            traj = dataset[traj_id]
            print(f"\n🎯 执行轨迹 {traj_id} ({len(traj)} steps)...")

            gt_actions = []
            pred_actions = []

            # 逐步推理
            step = 0
            while step < len(traj):
                # 提取当前帧
                data = _extract_step_data(traj, step)
                obs = obs_builder.build(
                    images=data["images"],
                    state=data["state"],
                )

                # 云端推理
                action, info = client.get_action(obs)
                gt_action = data["gt_action"]

                gt_actions.append(gt_action)
                pred_actions.append(action)

                step += action_horizon  # 跳步

            # 转换为 numpy
            gt_arr = np.array(gt_actions)
            pred_arr = np.array(pred_actions)

            # 绘图
            save_path = os.path.join(output_dir, f"traj_{traj_id}.jpeg")
            metrics = plot_trajectory_comparison(
                gt_actions=gt_arr,
                pred_actions=pred_arr,
                traj_id=traj_id,
                save_path=save_path,
            )
            all_mse.append(metrics["mse"])
            all_mae.append(metrics["mae"])
            print(f"   MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.6f}")
            print(f"   保存: {save_path}")

        # 汇总
        if all_mse:
            avg_mse = np.mean(all_mse)
            avg_mae = np.mean(all_mae)
            print(f"\n{'=' * 50}")
            print(f"📊 汇总:")
            print(f"   平均 MSE: {avg_mse:.6f}")
            print(f"   平均 MAE: {avg_mae:.6f}")
            print(f"   结果目录: {output_dir}")
            print(f"{'=' * 50}")


def _extract_step_data(traj, step: int) -> dict:
    """从数据集轨迹中提取单步数据"""
    # 图像
    images = {}
    for key in ["exterior_image_1_left", "wrist_image_left"]:
        if key in traj.columns:
            img = traj[key].iloc[step]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            images[key] = img

    # 状态
    state_cols = [c for c in traj.columns if c.startswith("state.")]
    if state_cols:
        state = np.vstack([traj[c].iloc[step] for c in state_cols]).flatten().astype(np.float32)
    else:
        state = np.zeros(17, dtype=np.float32)

    # GT 动作
    action_cols = [c for c in traj.columns if c.startswith("action.")]
    if action_cols:
        gt_action = np.vstack([traj[c].iloc[step] for c in action_cols]).flatten().astype(np.float32)
    else:
        gt_action = np.zeros(17, dtype=np.float32)

    return {"images": images, "state": state, "gt_action": gt_action}
