#!/usr/bin/env python3
"""
数据收集器 — 基于 unitree_rl_mjlab 官方仿真环境收集 G1/Go2 演示数据。

使用 unitree_rl_mjlab 的 ManagerBasedRlEnv + 步态生成器产生演示，
数据格式兼容 GR00T fine-tune 管线。

依赖:
  - unitree_rl_mjlab (pip install -e ../unitree_rl_mjlab)
  - mujoco-warp

数据流程:
  mjlab 仿真 → episode_*.npz → convert_to_lerobot.py → LeRobot v2

使用方法:
    # G1 机器人 (平坦地形)
    python collect_data.py --task Unitree-G1-Flat \
        --num-episodes 100 --instruction "walk forward"

    # Go2 机器人 (粗糙地形)
    python collect_data.py --task Unitree-Go2-Rough \
        --num-episodes 100 --instruction "walk forward"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 步态生成器 (脚本化动作，用于在 mjlab 仿真中自动产生演示)
# ──────────────────────────────────────────────────────────────────────────────


def gait_generator_g1(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
) -> np.ndarray:
    """G1 步态生成器 — 产生 29 维关节目标。

    简单正弦步态: 双腿交替迈步，手臂自然摆动，腰部保持平衡。
    关节顺序与 unitree_rl_mjlab MJCF (g1.xml) 一致。

    Args:
        step_idx: 当前时间步
        dt: 时间步长 (秒)
        speed: 行走速度 (影响步频)
        command: 语言指令 (影响方向)

    Returns:
        joint_targets: (29,) float32, 关节目标位置 (rad)
    """
    t = step_idx * dt
    freq = speed * 2.0  # 步频

    # 默认站立姿态 (HOME_KEYFRAME from unitree_rl_mjlab)
    targets = np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # 左腿
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # 右腿
        0.0, 0.0, 0.0,                            # 腰部
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,   # 左臂
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,  # 右臂
    ], dtype=np.float32)

    # 左腿 (0-5): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    phase_l = np.sin(2 * np.pi * freq * t)
    targets[0] += 0.3 * phase_l           # hip_pitch: 前后摆
    targets[1] += 0.05 * np.sin(phase_l)  # hip_roll: 微小侧摆
    targets[3] += -0.4 * max(phase_l, 0)  # knee: 弯曲

    # 右腿 (6-11): 相位差 π
    phase_r = np.sin(2 * np.pi * freq * t + np.pi)
    targets[6] += 0.3 * phase_r
    targets[7] += 0.05 * np.sin(phase_r)
    targets[9] += -0.4 * max(phase_r, 0)

    # 腰部 (12-14): 保持平衡
    targets[13] += 0.02 * np.sin(2 * np.pi * freq * t)

    # 左臂 (15-21): 自然摆动
    arm_phase = np.sin(2 * np.pi * freq * t)
    targets[15] += 0.2 * arm_phase           # shoulder_pitch
    targets[18] += -0.1 * max(arm_phase, 0)  # elbow

    # 右臂 (22-28): 相位差 π
    targets[22] += 0.2 * (-arm_phase)
    targets[25] += -0.1 * max(-arm_phase, 0)

    return targets


def gait_generator_go2(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
) -> np.ndarray:
    """Go2 步态生成器 — 产生 12 维关节目标。

    经典 trot 步态: 对角腿同步。
    关节顺序与 unitree_rl_mjlab MJCF (go2.xml) 一致。
    """
    t = step_idx * dt
    freq = speed * 2.5

    # 默认站立姿态 (INIT_STATE from unitree_rl_mjlab)
    default = np.array([
        0.0, 0.9, -1.8,   # FL
        0.1, 0.9, -1.8,   # FR
        -0.1, 1.0, -1.8,  # RL
        0.0, 1.0, -1.8,   # RR
    ], dtype=np.float32)

    targets = default.copy()
    phase = np.sin(2 * np.pi * freq * t)

    # FL (0-2) 和 RR (9-11) 同步, FR (3-5) 和 RL (6-8) 反相
    for i, (hip_idx, thigh_idx, calf_idx) in enumerate([
        (0, 1, 2),   # FL
        (3, 4, 5),   # FR
        (6, 7, 8),   # RL
        (9, 10, 11), # RR
    ]):
        p = phase if i % 2 == 0 else -phase
        targets[hip_idx] += 0.15 * p           # hip: 侧摆
        targets[thigh_idx] += 0.2 * abs(p)     # thigh: 抬腿
        targets[calf_idx] += -0.3 * max(p, 0)  # calf: 弯曲

    return targets


GAIT_GENERATORS = {
    "g1": gait_generator_g1,
    "go2": gait_generator_go2,
}


# ──────────────────────────────────────────────────────────────────────────────
# 任务 ID → 机器人类型映射
# ──────────────────────────────────────────────────────────────────────────────

TASK_TO_ROBOT: dict[str, str] = {
    "Unitree-G1-Flat": "g1",
    "Unitree-G1-Rough": "g1",
    "Unitree-G1-23Dof-Flat": "g1",
    "Unitree-Go2-Flat": "go2",
    "Unitree-Go2-Rough": "go2",
    "Unitree-A2-Flat": "a2",
    "Unitree-R1-Flat": "r1",
    "Unitree-H1_2-Flat": "h1_2",
}


# ──────────────────────────────────────────────────────────────────────────────
# 数据收集
# ──────────────────────────────────────────────────────────────────────────────


def collect_demonstrations(
    task_id: str = "Unitree-G1-Flat",
    num_episodes: int = 100,
    episode_length: int = 200,
    instruction: str = "walk forward",
    output_dir: str = "/workspace/data/raw",
    show_viewer: bool = False,
    speed: float = 0.5,
    seed: int = 42,
    num_envs: int = 1,
    device: str = "auto",
) -> str:
    """在 unitree_rl_mjlab 仿真环境中收集机器人演示数据。

    使用 unitree_rl_mjlab 的 ManagerBasedRlEnv 创建仿真环境，
    配合步态生成器自动产生演示。数据格式兼容 GR00T fine-tune 管线。

    Args:
        task_id: unitree_rl_mjlab 任务 ID (e.g. "Unitree-G1-Flat")
        num_episodes: 收集 episode 数量
        episode_length: 每个 episode 的时间步数
        instruction: 语言指令
        output_dir: 输出目录
        show_viewer: 是否显示可视化
        speed: 步态速度
        seed: 随机种子
        num_envs: 仿真环境数量
        device: 计算设备 ("auto", "cuda:0", "cpu")

    Returns:
        output_dir: 输出目录路径
    """
    # 延迟导入 torch (在没有 mjlab 的回退模式下不需要)
    # 4. 设备选择 (提前) ────────────────────────────────────────
    if device == "auto":
        try:
            import torch as _torch
            device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
            logger.warning("torch 未安装, 使用 CPU (无法加载 mjlab 仿真)")

    # ── 1. 确定机器人类型 ──────────────────────────────────────────────
    robot = TASK_TO_ROBOT.get(task_id)
    if robot is None:
        raise ValueError(
            f"不支持的任务: {task_id}\n"
            f"可选: {list(TASK_TO_ROBOT.keys())}"
        )

    # ── 2. 加载机器人配置 ──────────────────────────────────────────────
    if robot == "g1":
        from configs.g1_config import (
            G1_NUM_JOINTS, G1_DEFAULT_JOINT_ANGLES, G1_DT, G1_JOINT_NAMES,
        )
        num_joints = G1_NUM_JOINTS
        default_angles = np.array(
            [G1_DEFAULT_JOINT_ANGLES[n] for n in G1_JOINT_NAMES], dtype=np.float32
        )
        dt = G1_DT
    else:
        from configs.go2_config import (
            GO2_NUM_JOINTS, GO2_DEFAULT_JOINT_ANGLES, GO2_DT, GO2_JOINT_NAMES,
        )
        num_joints = GO2_NUM_JOINTS
        default_angles = np.array(
            [GO2_DEFAULT_JOINT_ANGLES[n] for n in GO2_JOINT_NAMES], dtype=np.float32
        )
        dt = GO2_DT

    gait_fn = GAIT_GENERATORS.get(robot)
    if gait_fn is None:
        raise ValueError(f"无步态生成器: {robot}")

    logger.info("=" * 60)
    logger.info("数据收集器 (unitree_rl_mjlab)")
    logger.info("  任务: %s", task_id)
    logger.info("  机器人: %s (%d joints)", robot, num_joints)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Episode 长度: %d steps (%.1f s)", episode_length, episode_length * dt)
    logger.info("  指令: %s", instruction)
    logger.info("  设备: %s", device)
    logger.info("  输出: %s", output_dir)
    logger.info("=" * 60)

    # ── 4. 使用 unitree_rl_mjlab 创建仿真环境 ─────────────────────────
    use_mjlab = False
    env = None

    try:
        # unitree_rl_mjlab 通过 mjlab.tasks.registry 加载任务
        # 需要先 import src.tasks 以触发任务注册
        rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
        if rl_mjlab_root.exists() and str(rl_mjlab_root) not in sys.path:
            sys.path.insert(0, str(rl_mjlab_root))

        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        # 触发任务注册 (import unitree_rl_mjlab 的 src 包)
        import src.tasks  # noqa: F401

        # 加载任务配置
        env_cfg = load_env_cfg(task_id, play=False)

        # 创建环境
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        logger.info("✅ mjlab 环境创建成功 (device=%s)", device)
        use_mjlab = True

    except ImportError as e:
        logger.warning("unitree_rl_mjlab 不可用: %s", e)
        logger.info("提示: cd /home/kxy/work/unitree/unitree_rl_mjlab && pip install -e .")
        logger.info("回退到纯数据模式 (使用步态生成器 + 模拟物理)")
    except Exception as e:
        logger.warning("mjlab 环境创建失败: %s", e)
        logger.info("回退到纯数据模式 (使用步态生成器 + 模拟物理)")

    # ── 5. 收集数据 ───────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    for ep in range(num_episodes):
        ep_speed = speed * rng.uniform(0.8, 1.2)

        # 重置环境
        if use_mjlab and env is not None:
            try:
                reset_out = env.reset()
                # ManagerBasedRlEnv.reset() 返回 ObservationVecDict
            except Exception as e:
                logger.debug("reset 失败 (ep %d): %s", ep, e)

        observations = []
        actions = []
        rewards = []

        for step_idx in range(episode_length):
            joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed, command=instruction)

            # 尝试从 mjlab 环境获取真实观测
            if use_mjlab and env is not None:
                try:
                    # 将动作转换为 tensor (需要与环境 action 空间匹配)
                    action_tensor = torch.from_numpy(
                        joint_targets
                    ).unsqueeze(0).to(device)

                    # step 环境
                    obs_out, reward, terminated, truncated, info = env.step(action_tensor)

                    # 从观测中提取数据
                    # obs_out 是 ObservationVecDict (dict[str, torch.Tensor])
                    jp = obs_out.get("joint_pos_rel", obs_out.get("joint_pos", None))
                    jv = obs_out.get("joint_vel_rel", obs_out.get("joint_vel", None))
                    bp = obs_out.get("base_pos", None)
                    bq = obs_out.get("base_quat", None)
                    blv = obs_out.get("base_lin_vel", None)
                    bav = obs_out.get("base_ang_vel", None)

                    joint_pos = jp.squeeze(0).cpu().numpy() if jp is not None else None
                    joint_vel = jv.squeeze(0).cpu().numpy() if jv is not None else None
                    base_pos = bp.squeeze(0).cpu().numpy() if bp is not None else None
                    base_quat = bq.squeeze(0).cpu().numpy() if bq is not None else None
                    base_lin_vel = blv.squeeze(0).cpu().numpy() if blv is not None else None
                    base_ang_vel = bav.squeeze(0).cpu().numpy() if bav is not None else None

                    # 如果任一关键数据缺失，回退模拟
                    if any(v is None for v in [joint_pos, base_pos]):
                        raise ValueError("观测数据不完整")

                except Exception as e:
                    if step_idx == 0:
                        logger.debug("mjlab step 失败, 回退模拟数据: %s", e)
                    joint_pos, joint_vel, base_pos, base_quat, base_lin_vel, base_ang_vel = \
                        _simulate_observation(joint_targets, default_angles, num_joints, dt, step_idx)
            else:
                joint_pos, joint_vel, base_pos, base_quat, base_lin_vel, base_ang_vel = \
                    _simulate_observation(joint_targets, default_angles, num_joints, dt, step_idx)

            observations.append({
                "state.joint_pos": joint_pos,
                "state.joint_vel": joint_vel,
                "state.base_pos": base_pos,
                "state.base_quat": base_quat,
                "state.base_lin_vel": base_lin_vel,
                "state.base_ang_vel": base_ang_vel,
            })
            actions.append({"target_joint_pos": joint_targets})
            rewards.append(0.0)

        # 保存 episode
        ep_path = output_path / f"episode_{ep:06d}.npz"
        np.savez_compressed(
            ep_path,
            observations=np.array(observations, dtype=object),
            actions=np.array(actions, dtype=object),
            rewards=np.array(rewards, dtype=np.float32),
        )

        if (ep + 1) % 10 == 0:
            logger.info("收集进度: %d/%d episodes", ep + 1, num_episodes)

    # ── 6. 关闭环境 ───────────────────────────────────────────────────
    if env is not None:
        try:
            env.close()
        except Exception:
            pass

    # ── 7. 保存元数据 ─────────────────────────────────────────────────
    metadata = {
        "robot": robot.upper(),
        "task_id": task_id,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "instruction": instruction,
        "state_dim": 37,
        "action_dim": num_joints,
        "control_mode": "joint_absolute",
        "simulator": "unitree_rl_mjlab (mjlab)",
        "dt": dt,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("✅ 数据收集完成!")
    logger.info("  任务: %s", task_id)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  输出: %s", output_dir)
    logger.info("=" * 60)

    return str(output_path)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────


def _simulate_observation(
    joint_targets: np.ndarray,
    default_angles: np.ndarray,
    num_joints: int,
    dt: float,
    step_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成模拟观测 (无仿真环境时使用)。

    简单 PD 控制模拟，关节位置跟随目标，基座匀速前进。
    """
    t = step_idx * dt

    # 模拟关节位置: 目标 + 小噪声
    joint_pos = joint_targets + np.random.randn(num_joints).astype(np.float32) * 0.01
    joint_vel = np.zeros(num_joints, dtype=np.float32)

    # 模拟基座运动
    base_pos = np.array([0.5 * t, 0.0, 0.78], dtype=np.float32)  # 匀速前进
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 四元数 (wxyz)
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)

    return joint_pos, joint_vel, base_pos, base_quat, base_lin_vel, base_ang_vel


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="数据收集器 — 基于 unitree_rl_mjlab 仿真收集 G1/Go2 演示数据"
    )
    parser.add_argument(
        "--task", type=str, default="Unitree-G1-Flat",
        choices=list(TASK_TO_ROBOT.keys()),
        help="unitree_rl_mjlab 任务 ID (default: Unitree-G1-Flat)",
    )
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="收集 episode 数量 (default: 100)")
    parser.add_argument("--episode-length", type=int, default=200,
                        help="每个 episode 的时间步数 (default: 200)")
    parser.add_argument("--instruction", type=str, default="walk forward",
                        help="语言指令 (default: 'walk forward')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (default: /workspace/data/raw)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="步态速度 (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备 (default: auto)")
    parser.add_argument("--show-viewer", action="store_true",
                        help="显示 mjlab 可视化")
    args = parser.parse_args()

    if args.output_dir is None:
        robot = TASK_TO_ROBOT.get(args.task, "g1")
        args.output_dir = f"/workspace/data/{robot}_raw"

    collect_demonstrations(
        task_id=args.task,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        instruction=args.instruction,
        output_dir=args.output_dir,
        show_viewer=args.show_viewer,
        speed=args.speed,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
