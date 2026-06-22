#!/usr/bin/env python3
"""
数据收集器 — 基于 unitree_rl_mjlab 官方仿真环境收集 G1/Go2 演示数据。

**真实可用的 GR00T 训练数据采集**:
  - 模式 1: --agent scripted   (正弦步态生成器, 用于 CI / smoke test)
  - 模式 2: --agent random     (随机关节目标, 用于 sanity check)
  - 模式 3: --agent trained    ⭐ 用训练好的 PPO 策略回放当专家示教
  - 模式 4: --agent zero       (零动作, 用于验证管线)
  - 模式 5: --agent keyboard   (手动遥操, 可选, 需要 display)

**真实视觉采集**: --video 启用 mjlab rgb_array 渲染, 每个 episode 输出 mp4

**动作空间**:
  - --action-mode absolute     关节目标绝对位置 (与 mjlab JointPositionActionCfg 一致)
  - --action-mode delta        ⭐ 关节目标相对当前增量 (GR00T N1.7 推荐)
  - --action-mode relative_eef 末端执行器位姿增量 (仅 G1 操作任务)

依赖:
  - unitree_rl_mjlab (pip install -e ../unitree_rl_mjlab)
  - mujoco-warp + torch

数据流程:
  mjlab 仿真 + (可选) PPO 策略 → episode_*.npz + episode_*.mp4
                                       ↓
                            convert_to_lerobot.py
                                       ↓
                          LeRobot v2 (parquet + mp4 + modality.json)

使用方法:
    # 1) Smoke test: 脚本化步态, 不开视频, 10 个 episode
    python collect_data.py --agent scripted --num-episodes 10

    # 2) ⭐ 真实训练数据: 用 PPO 策略回放, 开启视频, 100 episode
    python collect_data.py --agent trained \\
        --checkpoint ../unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<run>/model_*.pt \\
        --task Unitree-G1-Flat --num-episodes 100 --video

    # 3) 用 G1 粗糙地形 + 相对动作
    python collect_data.py --agent trained --task Unitree-G1-Rough \\
        --checkpoint <path> --action-mode delta --video
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────


def _asdict_safe(cfg: Any) -> dict[str, Any]:
    """asdict 兼容版: 支持 frozen dataclass, 普通 class, 嵌套 dict 等。

    mjlab 的 RslRlBaseRunnerCfg 是 frozen dataclass, 直接 asdict() 也行,
    但部分版本有嵌套非 dataclass 字段, 这里用更宽松的实现。
    """
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(cfg):
            return asdict(cfg)
    except Exception:
        pass
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def _to_numpy(t: Any) -> np.ndarray | None:
    """torch.Tensor / numpy.ndarray → numpy (squeeze batch dim)."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    arr = np.asarray(t)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


# ──────────────────────────────────────────────────────────────────────────────
# 步态生成器 (脚本化动作，用于在 mjlab 仿真中自动产生演示)
# ──────────────────────────────────────────────────────────────────────────────


def gait_generator_g1(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
    num_joints: int = 29,
) -> np.ndarray:
    """G1 步态生成器 — 产生 29 维 (默认) 或 23 维关节目标。

    修复: 支持 23Dof 变种 (Unitree-G1-23Dof-Flat/Rough)
    简单正弦步态: 双腿交替迈步，手臂自然摆动，腰部保持平衡。
    关节顺序与 unitree_rl_mjlab MJCF (g1.xml / g1_23dof.xml) 一致。

    Args:
        step_idx: 当前时间步
        dt: 时间步长 (秒)
        speed: 行走速度 (影响步频)
        command: 语言指令 (影响方向)
        num_joints: 关节数 (29 = 完整, 23 = 23Dof 变种)

    Returns:
        joint_targets: (num_joints,) float32, 关节目标位置 (rad)
    """
    t = step_idx * dt
    freq = speed * 2.0  # 步频

    if num_joints == 23:
        # 23Dof: 无 waist_roll/waist_pitch, 无 wrist_pitch/wrist_yaw
        # 顺序: 6 左腿 + 6 右腿 + 1 腰 + 5 左臂 + 5 右臂
        targets = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # 左腿 (6)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # 右腿 (6)
            0.0,                                      # waist_yaw (1)
            0.35, 0.18, 0.0, 0.87, 0.0,             # 左臂 (5, 无 wrist_pitch/yaw)
            0.35, -0.18, 0.0, 0.87, 0.0,            # 右臂 (5)
        ], dtype=np.float32)
    else:
        # 29Dof: 完整 (default)
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

    # 腰部 (12 或 12-14)
    if num_joints == 23:
        # 23Dof: 只有 waist_yaw, 不做腰部运动
        pass
    else:
        targets[13] += 0.02 * np.sin(2 * np.pi * freq * t)  # waist_pitch 微动

    # 左臂: 23Dof 起始 idx=13, 29Dof 起始 idx=15
    arm_l_start = 13 if num_joints == 23 else 15
    # 右臂: 23Dof 起始 idx=18, 29Dof 起始 idx=22
    arm_r_start = 18 if num_joints == 23 else 22

    arm_phase = np.sin(2 * np.pi * freq * t)
    targets[arm_l_start] += 0.2 * arm_phase           # shoulder_pitch
    targets[arm_l_start + 3] += -0.1 * max(arm_phase, 0)  # elbow
    targets[arm_r_start] += 0.2 * (-arm_phase)
    targets[arm_r_start + 3] += -0.1 * max(-arm_phase, 0)

    return targets


def gait_generator_go2(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
    **kwargs,  # 兼容 num_joints 参数
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


def gait_generator_generic(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
    num_joints: int = 12,
    default_angles: np.ndarray | None = None,
) -> np.ndarray:
    """通用脚本化步态 — 用于没有专门生成器的机器人 (A2/R1/H1_2/H2/As2).

    简单策略: 在 default 姿态上叠加正弦扰动, 用于 smoke test
    真实训练数据请用 --agent trained + PPO checkpoint
    """
    if default_angles is None:
        targets = np.zeros(num_joints, dtype=np.float32)
    else:
        targets = default_angles.copy()
    t = step_idx * dt
    freq = speed * 1.5
    for i in range(num_joints):
        targets[i] += 0.1 * np.sin(2 * np.pi * freq * t + i * 0.3)
    return targets


GAIT_GENERATORS = {
    "g1": gait_generator_g1,
    "go2": gait_generator_go2,
    "a2": gait_generator_generic,
    "as2": gait_generator_generic,
    "r1": gait_generator_generic,
    "h1_2": gait_generator_generic,
    "h2": gait_generator_generic,
}


# ──────────────────────────────────────────────────────────────────────────────
# 任务 ID → 机器人类型映射
# ──────────────────────────────────────────────────────────────────────────────

# 修复: 对齐 unitree_rl_mjlab 实际注册的任务 ID
#   (src/tasks/velocity/config/{g1,go2,a2,r1,h1_2,h2,as2,g1_23dof}/__init__.py)
TASK_TO_ROBOT: dict[str, str] = {
    # G1 (29 自由度 + 23 自由度)
    "Unitree-G1-Flat": "g1",
    "Unitree-G1-Rough": "g1",
    "Unitree-G1-23Dof-Flat": "g1",
    "Unitree-G1-23Dof-Rough": "g1",
    # Go2 (12 自由度)
    "Unitree-Go2-Flat": "go2",
    "Unitree-Go2-Rough": "go2",
    # A2 (四足)
    "Unitree-A2-Flat": "a2",
    "Unitree-A2-Rough": "a2",
    # R1 (人形)
    "Unitree-R1-Flat": "r1",
    "Unitree-R1-Rough": "r1",
    # H1_2 (人形)
    "Unitree-H1_2-Flat": "h1_2",
    "Unitree-H1_2-Rough": "h1_2",
    # H2 (人形, 修复: 之前缺失)
    "Unitree-H2-Flat": "h2",
    "Unitree-H2-Rough": "h2",
    # As2 (注意: 是 As2 不是 A2, 修复: 之前缺失)
    "Unitree-As2-Flat": "as2",
    "Unitree-As2-Rough": "as2",
}


# ──────────────────────────────────────────────────────────────────────────────# Viser 浏览器 viewer (可选, 用于远程监控数据采集进度)
# ──────────────────────────────────────────────────────────────────────────

class ViserViewer:
    """在浏览器中实时显示数据采集进度 (依赖 viser 库).

    提供:
      - Episode / Step 进度
      - 当前指令
      - Base linear velocity 实时数值
      - 关节目标范围

    启动后打印 URL (如 http://localhost:8080), 浏览器打开即可.
    默认关闭 (--viser 启用).
    """

    def __init__(self, port: int = 8080):
        import viser  # 延迟导入 (viser 是可选依赖)
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.url = f"http://localhost:{port}"

        with self.server.gui.add_folder("📊 进度"):
            self.ep_text = self.server.gui.add_text("Episode", initial_value="-/-", disabled=True)
            self.step_text = self.server.gui.add_text("Step", initial_value="-/-", disabled=True)
            self.instr_text = self.server.gui.add_text("指令", initial_value="(初始化中...)", disabled=True)

        with self.server.gui.add_folder("🤖 状态"):
            self.base_vel_text = self.server.gui.add_text("Base vel (m/s)", initial_value="-", disabled=True)
            self.action_text = self.server.gui.add_text("关节目标范围", initial_value="-", disabled=True)

        logger.info("🌐 viser viewer 已启动: %s (浏览器打开此 URL)", self.url)

    def update(self, ep: int, total_ep: int, step: int, total_step: int,
               instruction: str, base_vel=None, joint_targets=None) -> None:
        self.ep_text.value = f"{ep+1}/{total_ep}"
        self.step_text.value = f"{step}/{total_step}"
        self.instr_text.value = instruction
        if base_vel is not None and len(base_vel) >= 3:
            vx, vy, vz = float(base_vel[0]), float(base_vel[1]), float(base_vel[2])
            self.base_vel_text.value = f"({vx:+.2f}, {vy:+.2f}, {vz:+.2f})"
        if joint_targets is not None and len(joint_targets) > 0:
            jt = np.asarray(joint_targets)
            self.action_text.value = f"[{jt.min():.2f}, {jt.max():.2f}]"

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────# 数据收集
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
    # ─── 新增参数 ───
    agent: str = "scripted",
    checkpoint: str | None = None,
    action_mode: str = "delta",
    enable_video: bool = False,
    video_height: int = 224,
    video_width: int = 224,
    video_fps: int = 0,  # 0 = auto (根据 dt 算), >0 = 显式 fps (GR00T 推荐 30)
    camera_name: str | None = None,
    instruction_pool: list[str] | None = None,
    viser: bool = False,  # 在浏览器里看数据采集进度 (默认关, 需 pip install viser)
    viser_port: int = 8080,
    ee_body: str | None = None,  # relative_eef 模式下的末端 body 名 (默认 G1: left_rubber_hand)
) -> str:
    """在 unitree_rl_mjlab 仿真环境中收集机器人演示数据。

    使用 unitree_rl_mjlab 的 ManagerBasedRlEnv 创建仿真环境，
    配合 (脚本化 / 训练好的 / 随机的) 策略产生演示。
    数据格式兼容 GR00T fine-tune 管线 (LeRobot v2)。

    Args:
        task_id: unitree_rl_mjlab 任务 ID (e.g. "Unitree-G1-Flat")
        num_episodes: 收集 episode 数量
        episode_length: 每个 episode 的时间步数
        instruction: 默认语言指令 (每个 episode 从 instruction_pool 随机选)
        output_dir: 输出目录
        show_viewer: 是否显示可视化 (需要 DISPLAY)
        speed: 步态速度 (仅 scripted 模式)
        seed: 随机种子
        num_envs: 仿真环境数量 (数据收集时强制为 1; 多环境会增加显存)
        device: 计算设备 ("auto", "cuda:0", "cpu")
        agent: 策略类型
            - "scripted": 正弦步态 (无需 checkpoint)
            - "random": 随机关节目标 (无需 checkpoint)
            - "zero": 零动作 (无需 checkpoint)
            - "trained": 用 PPO checkpoint 回放 (必须指定 --checkpoint)
        checkpoint: PPO checkpoint 路径 (.pt 文件), trained 模式必填
        action_mode: 动作空间
            - "absolute": 关节目标绝对位置 (mjlab JointPositionAction 直接吃)
            - "delta": 关节目标相对当前位置的增量 (⭐ GR00T N1.7 推荐)
            - "relative_eef": 末端位姿增量 (G1 操作任务)
        enable_video: 是否采集 RGB 视频 (mjlab offscreen render)
        video_height / video_width: 视频分辨率 (默认 224x224 匹配 GR00T)
        video_fps: 视频帧率 (默认 0 = 根据仿真 dt 自动算, 通常 30~50fps;
                            设为 30 等可显式控制, GR00T 训练推荐 30)
        camera_name: mjlab camera name (None = 用 unitree_rl_mjlab 默认 front_view)
        instruction_pool: 备选指令列表 (每 episode 随机抽一个, 增加数据多样性)

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
            G1_23DOF_NUM_JOINTS, G1_23DOF_DEFAULT_JOINT_ANGLES,
            G1_23DOF_JOINT_NAMES, G1_23DOF_DT,
        )
        # 修复: 23Dof 变种 (Unitree-G1-23Dof-Flat/Rough) 关节数是 23 不是 29
        is_23dof = "23Dof" in task_id
        if is_23dof:
            num_joints = G1_23DOF_NUM_JOINTS
            default_angles = np.array(
                [G1_23DOF_DEFAULT_JOINT_ANGLES[n] for n in G1_23DOF_JOINT_NAMES],
                dtype=np.float32,
            )
            dt = G1_23DOF_DT
        else:
            num_joints = G1_NUM_JOINTS
            default_angles = np.array(
                [G1_DEFAULT_JOINT_ANGLES[n] for n in G1_JOINT_NAMES],
                dtype=np.float32,
            )
            dt = G1_DT
    else:
        # 修复: 非 G1 机器人 (Go2/A2/As2/R1/H1_2/H2) 的 num_joints
        # 从 unitree_rl_mjlab MJCF 统计 (4足 ~12, 人形 24-30)
        from configs.go2_config import GO2_NUM_JOINTS, GO2_DT
        # 关节数查表 (来源: unitree_rl_mjlab/src/assets/robots/*/xmls/*.xml)
        robot_joint_counts = {
            "go2": 12,    # go2.xml: 12 joints
            "a2":  12,    # a2.xml: 12 joints (四足)
            "as2": 12,    # as2.xml: 12 joints
            "r1":  24,    # r1.xml: 24 joints (人形)
            "h1_2": 27,   # h1_2.xml: 27 joints
            "h2":  29,    # h2.xml: 29 joints (近似, 实际可能有 free-joint 折算)
        }
        num_joints = robot_joint_counts.get(robot, GO2_NUM_JOINTS)
        # 通用 default_angles: 全 0 (mjlab reset 时会用实际 HOME_KEYFRAME 覆盖)
        # 注意: 脚本化步态的初始姿态是 0, mjlab 接管后会用 entity 的 default_joint_pos
        # 对于 random/zero agent, 不需要 default_angles
        default_angles = np.zeros(num_joints, dtype=np.float32)
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
    logger.info("  Agent: %s%s", agent,
                 f" (checkpoint={checkpoint})" if checkpoint else "")
    logger.info("  Action mode: %s", action_mode)
    logger.info("  Video: %s (cam=%s, %dx%d @ %dfps)",
                 enable_video, camera_name or "default",
                 video_height, video_width,
                 video_fps if video_fps > 0 else int(round(1.0 / dt)))
    logger.info("  设备: %s", device)
    logger.info("  输出: %s", output_dir)
    logger.info("=" * 60)

    # ── 4. 使用 unitree_rl_mjlab 创建仿真环境 ─────────────────────────
    env_raw = None  # ManagerBasedRlEnv (有 observation_manager / render)
    env_wrapped = None  # RslRlVecEnvWrapper (供 PPO policy 使用)
    policy_fn: Callable | None = None  # trained 模式下: (obs) → action tensor
    use_mjlab = False
    viewer = None  # 修复: 提前初始化, 避免 mjlab 失败时 UnboundLocalError

    try:
        # unitree_rl_mjlab 通过 mjlab.tasks.registry 加载任务
        # 需要先 import src.tasks 以触发任务注册
        import torch as _torch  # noqa: F401 (确认 torch 可用)
        rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
        if rl_mjlab_root.exists() and str(rl_mjlab_root) not in sys.path:
            sys.path.insert(0, str(rl_mjlab_root))

        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        # 触发任务注册 (import unitree_rl_mjlab 的 src 包)
        import src.tasks  # noqa: F401

        # 加载任务配置 — 修正: 官方 API 是 play=True (用于 inference / 收集)
        env_cfg = load_env_cfg(task_id, play=True)
        # 数据收集强制单环境 (multi-env 会破坏 episode 边界)
        env_cfg.scene.num_envs = 1

        # 视频 / 渲染配置
        if enable_video:
            env_cfg.viewer.width = video_width
            env_cfg.viewer.height = video_height
            # 注: mjlab ViewerConfig 不暴露 camera_name 字段
            # 仿真默认相机名以资产定义为准, 这里只能调整 lookat/distance/elevation/azimuth
            # 如需定制, 请在 unitree_rl_mjlab 的任务 env_cfg 中修改 ViewerConfig

        # ─── 创建原始 env (用于获取 per-key obs / state / 渲染) ───
        render_mode = "rgb_array" if enable_video else None
        env_raw = ManagerBasedRlEnv(
            cfg=env_cfg, device=device, render_mode=render_mode
        )
        logger.info("✅ mjlab 环境创建成功 (device=%s, render=%s)",
                    device, render_mode or "off")
        use_mjlab = True

        # ─── viser 浏览器 viewer (可选, --viser 启用) ───────────
        viewer = None
        if viser:
            try:
                viewer = ViserViewer(port=viser_port)
            except ImportError:
                logger.warning("viser 未安装, 跳过浏览器 viewer (pip install viser)")

        # ─── 如果是 trained 模式, 加载 PPO 策略 ───
        if agent == "trained":
            if not checkpoint or not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"--agent trained 必须指定 --checkpoint, 且文件必须存在\n"
                    f"  当前: {checkpoint}\n"
                    f"  提示: 先跑 unitree_rl_mjlab/scripts/train.py 训一个 PPO 模型"
                )

            agent_cfg = load_rl_cfg(task_id)
            # 包装成 VecEnv (policy 需要)
            env_wrapped = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

            # 注: load_runner_cls 访问 _REGISTRY[task_name].runner_cls, 未知任务会 KeyError。
            # mjlab 的 load_runner_cls 本身能返回 None (表示走默认 OnPolicyRunner),
            # 因此只需包一层 try/except 处理未注册任务场景。
            try:
                runner_cls = load_runner_cls(task_id)
            except KeyError:
                runner_cls = None
            runner_cls = runner_cls or MjlabOnPolicyRunner
            # 修复: 统一用 dataclasses.asdict 转换 (兼容 frozen dataclass)
            # 旧 _asdict_safe 在递归时可能漏字段, asdict 才是 mjlab 官方推荐的写法
            from dataclasses import asdict
            try:
                train_cfg_dict = asdict(agent_cfg)
            except Exception:
                train_cfg_dict = _asdict_safe(agent_cfg)
            runner = runner_cls(env_wrapped, train_cfg_dict, log_dir=None, device=device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                        map_location=device)
            policy_fn = runner.get_inference_policy(device=device)
            logger.info("✅ PPO 策略加载成功: %s", Path(checkpoint).name)

    except ImportError as e:
        logger.warning("unitree_rl_mjlab 不可用: %s", e)
        logger.info("提示: cd /home/kxy/work/unitree/unitree_rl_mjlab && pip install -e .")
        logger.info("回退到纯数据模式 (使用 %s 生成器 + 模拟物理)", agent)
    except FileNotFoundError as e:
        logger.error("❌ %s", e)
        raise
    except Exception as e:
        logger.warning("mjlab 环境创建/策略加载失败: %s", e)
        import traceback
        logger.debug("Traceback:\n%s", traceback.format_exc())
        logger.info("回退到纯数据模式 (使用 %s 生成器 + 模拟物理)", agent)

    # ── 5. 收集数据 ───────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 准备可选的视频编码器 (懒加载, 缺 imageio 时优雅降级)
    imageio = None
    if enable_video:
        try:
            import imageio.v2 as imageio  # type: ignore
            logger.info("✅ imageio 可用, 将保存 mp4 视频")
        except ImportError:
            try:
                import imageio  # type: ignore
                logger.info("✅ imageio (v1) 可用, 将保存 mp4 视频")
            except ImportError:
                logger.warning("⚠️ imageio 未安装, 视频将保存为 npz 帧序列 (后期 convert 时再编码)")
                imageio = None

    # 准备指令池
    if instruction_pool:
        logger.info("  指令池: %d 条", len(instruction_pool))
    else:
        instruction_pool = [instruction]

    # 状态维度元数据 (用于 modality.json)
    state_dim = num_joints * 2 + 3 + 4 + 3 + 3  # 71 / 37
    if robot == "g1":
        from configs.g1_config import G1_VIDEO_HEIGHT, G1_VIDEO_WIDTH, G1_VIDEO_KEY
        video_height = video_height or G1_VIDEO_HEIGHT
        video_width = video_width or G1_VIDEO_WIDTH
        video_key = G1_VIDEO_KEY
    else:
        from configs.go2_config import GO2_VIDEO_HEIGHT, GO2_VIDEO_WIDTH, GO2_VIDEO_KEY
        video_height = video_height or GO2_VIDEO_HEIGHT
        video_width = video_width or GO2_VIDEO_WIDTH
        video_key = GO2_VIDEO_KEY

    # relative_eef 模式的 EE body 名 (修复: 不再 hard-code 零向量)
    action_mode_ee_body = ee_body  # None 时 G1 默认 "left_rubber_hand", Go2 不支持
    if action_mode == "relative_eef" and robot != "g1":
        logger.warning("relative_eef 模式仅 G1 有末端执行器, Go2 数据将为零 (用于调试)")
    elif action_mode == "relative_eef" and action_mode_ee_body is None:
        action_mode_ee_body = "left_rubber_hand"
        logger.info("relative_eef 模式默认 EE body: %s (可用 --ee-body 覆盖)",
                    action_mode_ee_body)

    for ep in range(num_episodes):
        ep_speed = speed * rng.uniform(0.8, 1.2)
        ep_instruction = instruction_pool[ep % len(instruction_pool)]
        ep_rng = np.random.RandomState(seed + ep)

        # 重置环境
        current_joint_pos = default_angles.copy()  # 用于 delta 动作的"上一时刻"
        if use_mjlab and env_raw is not None:
            try:
                env_raw.reset()
                # 同步当前 joint_pos (从 critic obs 拿, 比 actor 准)
                obs_dict = _get_per_key_obs(env_raw)
                if obs_dict.get("joint_pos_rel") is not None:
                    current_joint_pos = _to_numpy(obs_dict["joint_pos_rel"]).copy()
                elif obs_dict.get("joint_pos") is not None:
                    current_joint_pos = _to_numpy(obs_dict["joint_pos"]).copy()
            except Exception as e:
                logger.debug("reset 失败 (ep %d): %s", ep, e)

        # ─── 状态变量: 上一时刻 EE 位姿, 用于 relative_eef 计算 delta ──
        prev_ee_pose: np.ndarray | None = None

        observations = []
        actions = []
        rewards = []
        frames = []  # 视频帧缓冲

        for step_idx in range(episode_length):
            # ─── 1) 决定动作 ─────────────────────────────────────
            if agent == "scripted":
                # 修复: 23Dof 变种需要传 num_joints=23 给 gait_generator
                if robot == "g1":
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)
                else:
                    # 通用步态 (A2/As2/R1/H1_2/H2 + Go2): 不传 default_angles
                    # 因为 R1/H1_2/H2 的 HOME_KEYFRAME 维度可能跟 num_joints 不一致
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)
            elif agent == "random":
                # 随机目标在 default ± 0.3 范围内
                joint_targets = (default_angles
                                 + ep_rng.uniform(-0.3, 0.3, num_joints).astype(np.float32))
            elif agent == "zero":
                joint_targets = np.zeros(num_joints, dtype=np.float32)
            elif agent == "trained" and policy_fn is not None:
                # PPO 策略: 用 actor 观测 → 动作 (这是 raw action, mjlab 内部会 *scale)
                try:
                    obs_vec = env_wrapped.unwrapped.observation_manager.compute()["actor"]
                    obs_vec = obs_vec.float()
                    with torch.no_grad():
                        action_t = policy_fn(obs_vec)
                    # action_t shape: (num_envs, action_dim) — 取第 0 个 env
                    joint_targets = _to_numpy(action_t[0])
                except Exception as e:
                    if step_idx == 0:
                        logger.debug("trained 策略推理失败, 回退 scripted: %s", e)
                    # 修复: 同上, fallback 也要传 num_joints
                    if robot == "g1":
                        joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                                command=ep_instruction,
                                                num_joints=num_joints)
                    else:
                        joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                                command=ep_instruction,
                                                num_joints=num_joints)
            else:
                # 修复: 同上
                if robot == "g1":
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)
                else:
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)

            # ─── 2) 推 mjlab 环境 (有的话) ─────────────────────────
            # 注意: mjlab env.step() 内部已经自动处理 terminated env 的 reset
            #       (调用 self._reset_idx(reset_env_ids)), 所以我们不需要再 reset
            obs_dict: dict[str, Any] = {}
            if use_mjlab and env_raw is not None:
                try:
                    import torch as _torch
                    action_tensor = _torch.from_numpy(joint_targets).unsqueeze(0).to(device)
                    obs_out, reward, terminated, truncated, info = env_raw.step(action_tensor)

                    # 检查是否有 env 被自动 reset (terminated/truncated)
                    # mjlab step 已 reset, 但我们需要同步 prev_ee_pose (新 episode 起点)
                    if (terminated.any() or truncated.any()) and action_mode == "relative_eef":
                        prev_ee_pose = None
                except Exception as e:
                    if step_idx == 0:
                        logger.warning("mjlab step 失败, 回退模拟: %s", e)
                    obs_dict = {}
                else:
                    # 重新计算 per-key obs (mjlab step 后 obs_out 是拼接 tensor)
                    obs_dict = _get_per_key_obs(env_raw)

            # ─── 3) 提取 state ──────────────────────────────────────
            joint_pos = _to_numpy(obs_dict.get("joint_pos_rel",
                                obs_dict.get("joint_pos")))
            joint_vel = _to_numpy(obs_dict.get("joint_vel_rel",
                                obs_dict.get("joint_vel")))
            base_pos = _to_numpy(obs_dict.get("base_pos"))
            base_quat = _to_numpy(obs_dict.get("base_quat"))
            base_lin_vel = _to_numpy(obs_dict.get("base_lin_vel"))
            base_ang_vel = _to_numpy(obs_dict.get("base_ang_vel"))

            # 缺数据时用模拟 fallback
            if joint_pos is None or base_pos is None:
                jp, jv, bp, bq, blv, bav = _simulate_observation(
                    joint_targets, default_angles, num_joints, dt, step_idx)
                joint_pos = joint_pos or jp
                joint_vel = joint_vel or jv
                base_pos = base_pos or bp
                base_quat = base_quat or bq
                base_lin_vel = base_lin_vel or blv
                base_ang_vel = base_ang_vel or bav

            # 维度修正: mjlab term `joint_pos` 的 func=mdp.joint_pos_rel, 内容是相对值
            # 修复: 旧代码用 obs_dict.get("joint_pos_rel") is not None → 永远 False
            #       (term 名是 "joint_pos" 不是 "joint_pos_rel"), 改用 obs_dict.get("joint_pos")
            if obs_dict.get("joint_pos") is not None and joint_pos is not None:
                # mjlab 返回的是 rel (=joint_pos - default), 还原为 abs
                joint_pos = joint_pos + default_angles

            # ─── 4) 构造动作 (按 action_mode) ───────────────────────
            if action_mode == "absolute":
                action_record = {
                    "action.joint_position_target": joint_targets.astype(np.float32),
                }
            elif action_mode == "delta":
                delta = (joint_targets - current_joint_pos).astype(np.float32)
                action_record = {
                    "action.joint_position_delta": delta,
                    "action.joint_position_last": current_joint_pos.astype(np.float32),
                }
            elif action_mode == "relative_eef":
                # 修复: 用 mujoco forward kinematics 计算当前末端位姿,
                #      然后减去上一时刻得到 7D delta (xyz + quat_wxyz)
                if action_mode_ee_body is None and robot != "g1":
                    # Go2 没有末端执行器, 退到 zero (这是预期行为)
                    ee_pose_delta = np.zeros(7, dtype=np.float32)
                else:
                    ee_body_name = action_mode_ee_body or "left_rubber_hand"
                    cur_ee_pose = _get_ee_pose(env_raw, ee_body_name, num_envs=1)
                    if cur_ee_pose is None:
                        # FK 失败 (无 mjlab / 无 FK), 退到 zero
                        ee_pose_delta = np.zeros(7, dtype=np.float32)
                    elif prev_ee_pose is None:
                        # 第一帧无 prev, 用 zero delta (不要用 random, 避免破坏训练分布)
                        ee_pose_delta = np.zeros(7, dtype=np.float32)
                    else:
                        # 位姿 delta: xyz 直接相减, quat 用"左乘 prev 的逆"得到增量
                        pos_delta = cur_ee_pose[:3] - prev_ee_pose[:3]
                        # quat_delta = cur * prev^{-1} (wxyz 约定)
                        quat_delta = _quat_diff_wxyz(cur_ee_pose[3:], prev_ee_pose[3:])
                        ee_pose_delta = np.concatenate([pos_delta, quat_delta]).astype(np.float32)
                    # 更新 prev (即使本帧是 zero, 下一帧也要基于此)
                    if cur_ee_pose is not None:
                        prev_ee_pose = cur_ee_pose
                action_record = {
                    "action.ee_pose_delta": ee_pose_delta,
                }
            else:
                raise ValueError(f"未知 action_mode: {action_mode}")

            # ─── 5) 渲染视频 ───────────────────────────────────────
            if enable_video:
                frame = _render_frame(env_raw, ep, step_idx)
                if frame is not None:
                    frames.append(frame)

            # ─── 5.5) viser 浏览器 viewer 更新 (可选) ─────────────
            if viewer is not None and (step_idx % 5 == 0 or step_idx == episode_length - 1):
                try:
                    viewer.update(
                        ep=ep, total_ep=num_episodes,
                        step=step_idx + 1, total_step=episode_length,
                        instruction=ep_instruction,
                        base_vel=base_lin_vel,
                        joint_targets=joint_targets,
                    )
                except Exception as e:
                    if step_idx == 0:
                        logger.debug("viser viewer 更新失败: %s", e)

            # ─── 6) 记录 ───────────────────────────────────────────
            observations.append({
                "state.joint_pos": joint_pos.astype(np.float32),
                "state.joint_vel": joint_vel.astype(np.float32) if joint_vel is not None
                                   else np.zeros(num_joints, dtype=np.float32),
                "state.base_pos": base_pos.astype(np.float32),
                "state.base_quat": base_quat.astype(np.float32) if base_quat is not None
                                   else np.array([1, 0, 0, 0], dtype=np.float32),
                "state.base_lin_vel": base_lin_vel.astype(np.float32) if base_lin_vel is not None
                                      else np.zeros(3, dtype=np.float32),
                "state.base_ang_vel": base_ang_vel.astype(np.float32) if base_ang_vel is not None
                                      else np.zeros(3, dtype=np.float32),
            })
            actions.append(action_record)
            rewards.append(0.0)

            # 更新"上一时刻"位姿 (用于下一步 delta)
            current_joint_pos = joint_targets.astype(np.float32).copy()

        # ─── 保存 episode (.npz) ──────────────────────────────────
        ep_path = output_path / f"episode_{ep:06d}.npz"
        np.savez_compressed(
            ep_path,
            observations=np.array(observations, dtype=object),
            actions=np.array(actions, dtype=object),
            rewards=np.array(rewards, dtype=np.float32),
            instruction=ep_instruction,  # 每 episode 的指令 (供 LeRobot 标注)
        )

        # ─── 保存视频 (.mp4 或 frames.npz) ──────────────────────
        if enable_video and frames:
            vid_path = output_path / f"episode_{ep:06d}.mp4"
            if imageio is not None:
                try:
                    imageio.mimsave(str(vid_path), frames,
                                     fps=video_fps if video_fps > 0 else int(round(1.0 / dt)))
                except Exception as e:
                    logger.warning("mp4 编码失败, 改存 frames.npz: %s", e)
                    np.savez_compressed(output_path / f"episode_{ep:06d}_frames.npz",
                                        frames=np.stack(frames))
            else:
                np.savez_compressed(output_path / f"episode_{ep:06d}_frames.npz",
                                    frames=np.stack(frames))

        if (ep + 1) % 10 == 0:
            extra = f", {len(frames)} frames" if enable_video else ""
            logger.info("收集进度: %d/%d episodes (ep_instruction='%s'%s)",
                        ep + 1, num_episodes, ep_instruction[:30], extra)

    # ── 6. 关闭环境 ───────────────────────────────────────────────────
    if env_raw is not None:
        try:
            env_raw.close()
        except Exception:
            pass

    # 关闭 viser viewer
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass

    # ── 7. 保存元数据 ─────────────────────────────────────────────────
    metadata = {
        "robot": robot.upper(),
        "task_id": task_id,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "instruction": instruction,
        "instruction_pool": instruction_pool,
        "agent": agent,
        "checkpoint": checkpoint,
        "action_mode": action_mode,
        "enable_video": enable_video,
        "video_fps": (video_fps if video_fps > 0 else int(round(1.0 / dt))) if enable_video else 0,
        "video_height": video_height if enable_video else 0,
        "video_width": video_width if enable_video else 0,
        "video_key": video_key if enable_video else None,
        "num_joints": num_joints,
        "state_dim": state_dim,
        "action_dim": num_joints if action_mode != "relative_eef" else 7,
        "control_mode": "joint_" + action_mode,
        "simulator": "unitree_rl_mjlab (mjlab)" if use_mjlab else "mock",
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


def _get_per_key_obs(env_raw: Any) -> dict[str, Any]:
    """从 unitree_rl_mjlab ManagerBasedRlEnv 取 per-key 观测 (dict[str, tensor]).

    mjlab env.step() 返回的 obs 是按 actor/critic 拼接的 tensor,
    这里直接通过 Entity.data 拿分组前的 dict (因为 mjlab ObservationManager
    默认 concatenate_terms=True, 没有 per-key 访问接口).

    修复: 删除对 om.terms / om._terms 的尝试 (mjlab ObservationManager
         无此属性, 是死代码; 直接走 Entity.data 即可).
    """
    out: dict[str, Any] = {}
    try:
        robot = env_raw.unwrapped.scene["robot"]
        rd = robot.data
        # base pose/velocity (mjlab root_link_* 字段, shape (num_envs, D))
        out["base_pos"] = rd.root_link_pos_w
        out["base_quat"] = rd.root_link_quat_w
        out["base_lin_vel"] = rd.root_link_lin_vel_w
        out["base_ang_vel"] = rd.root_link_ang_vel_w
        # joint (mjlab Entity.data.joint_pos 是绝对位置; mjlab term `joint_pos`
        #        func=mdp.joint_pos_rel 返回 rel, 此处用 Entity.data 绝对)
        out["joint_pos"] = rd.joint_pos
        out["joint_vel"] = rd.joint_vel
    except Exception as e:
        logger.debug("Entity.data 失败: %s", e)

    return out



def _render_frame(env_raw: Any, ep: int, step: int) -> np.ndarray | None:
    """从 mjlab env 渲染一帧 RGB 图像 (H, W, 3) uint8.

    多种 API 兼容性尝试:
      1) env.render()                — gym-style (mjlab 已内置 _offline_renderer, 最优)
      2) env.unwrapped.sim.mj_data   — 直接走 mujoco native offscreen (兼容)
      3) viser / native viewer (需 DISPLAY)

    修复: 优先用 env.render() (它内部已封装 mjlab 的 OffscreenRenderer),
         不再手造 mujoco.Renderer (复杂 + 容易出现 model/dim 不一致).
    """
    if env_raw is None:
        return None
    # 方案 1: gym-style render (mjlab 内部用 _offline_renderer)
    try:
        frame = env_raw.render()
        if frame is not None:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 \
                    else arr.clip(0, 255).astype(np.uint8)
            return arr
    except Exception:
        pass
    # 方案 2: mjlab sim.mj_data + mujoco.Renderer (兜底)
    try:
        sim = env_raw.unwrapped.sim
        mj_model = sim.mj_model
        mj_data = sim.mj_data
        import mujoco  # noqa
        from mujoco import Renderer
        viewer_cfg = env_raw.unwrapped.cfg.viewer
        h = int(getattr(viewer_cfg, "height", 224))
        w = int(getattr(viewer_cfg, "width", 224))
        renderer = Renderer(model=mj_model, height=h, width=w)
        # mjlab ViewerConfig 没有 camera_name, 用 entity_name + body_name 推断
        # 默认渲染 root body (-1 = free camera)
        renderer.update_scene(mj_data, camera=-1)
        frame = renderer.render()
        return np.asarray(frame).astype(np.uint8)
    except Exception as e:
        if step == 0:
            logger.debug("sim.render 失败: %s", e)
    return None


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

    # 修复: 用 joint_targets.shape[0] 而不是 num_joints
    # 因为不同机器人的 default_angles 维度可能跟 num_joints 不一致
    # (例如 H2 的 30 关节里只有 29 个 actuated)
    actual_dim = joint_targets.shape[0] if joint_targets is not None else num_joints

    # 模拟关节位置: 目标 + 小噪声
    joint_pos = joint_targets + np.random.randn(actual_dim).astype(np.float32) * 0.01
    joint_vel = np.zeros(actual_dim, dtype=np.float32)

    # 模拟基座运动
    base_pos = np.array([0.5 * t, 0.0, 0.78], dtype=np.float32)  # 匀速前进
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 四元数 (wxyz)
    base_lin_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)

    return joint_pos, joint_vel, base_pos, base_quat, base_lin_vel, base_ang_vel


def _get_ee_pose(
    env_raw: Any,
    body_name: str,
    num_envs: int = 1,
) -> np.ndarray | None:
    """通过 mujoco forward kinematics 获取末端执行器当前位姿 (xyz + quat_wxyz).

    修复: 用于 relative_eef 模式的真实 EE 增量计算 (替代旧的零向量 fallback)

    Args:
        env_raw: unitree_rl_mjlab ManagerBasedRlEnv (需 unwrapped.sim)
        body_name: MJCF 中定义的 body name (e.g. "left_rubber_hand")
        num_envs: 环境数量, 默认 1 (取第 0 个 env)

    Returns:
        np.ndarray (7,) [x, y, z, qw, qx, qy, qz] 或 None (失败时)
    """
    if env_raw is None:
        return None
    try:
        sim = env_raw.unwrapped.sim
        # mjlab 修复: sim.mj_model / sim.mj_data 是原生 mujoco (有 .xpos/.xquat)
        #   sim.data / sim.model 是 Warp DataBridge/ModelBridge, 没有 .xpos
        #   旧代码用 sim._data / sim._model → AttributeError (不存在)
        mj_model = sim.mj_model
        mj_data = sim.mj_data
        # 触发 forward kinematics 更新 (确保 xpos/xquat 是当前状态)
        # 注意: env.step() 已经自动调用 sim.forward(), 这里再调一次是幂等安全
        if hasattr(sim, "forward"):
            sim.forward()
        import mujoco  # noqa
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            logger.debug("EE body 不存在: %s", body_name)
            return None
        # mj_data.xpos/xquat 是一维: shape (nbody,) / (nbody, 4) (单 env)
        pos = np.asarray(mj_data.xpos[body_id], dtype=np.float32)
        quat = np.asarray(mj_data.xquat[body_id], dtype=np.float32)  # wxyz
        return np.concatenate([pos, quat]).astype(np.float32)
    except Exception as e:
        logger.debug("FK 失败 (%s): %s", body_name, e)
        return None


def _quat_diff_wxyz(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """计算两个 wxyz 四元数的"相对旋转": cur * prev^{-1}.

    修复: 用于 relative_eef 模式的姿态增量计算 (避免数值不稳定的"直接相减")

    Args:
        cur: 当前四元数 (4,) wxyz
        prev: 上一时刻四元数 (4,) wxyz

    Returns:
        delta 四元数 (4,) wxyz, 表示从 prev 旋转到 cur 的最小旋转
    """
    # 1) prev 的共轭 (逆, 单位四元数): (w, -x, -y, -z)
    prev_conj = np.array(
        [prev[0], -prev[1], -prev[2], -prev[3]], dtype=np.float32
    )
    # 2) cur ⊗ prev_conj (Hamilton 乘积, wxyz)
    w1, x1, y1, z1 = cur
    w2, x2, y2, z2 = prev_conj
    delta = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
    ], dtype=np.float32)
    # 3) 归一化 (防止累积浮点漂移)
    norm = np.linalg.norm(delta)
    if norm > 1e-8:
        delta = delta / norm
    return delta


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="数据收集器 — 基于 unitree_rl_mjlab 仿真收集 G1/Go2 演示数据 (兼容 GR00T fine-tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1) Smoke test: 脚本化步态, 不开视频
  python collect_data.py --agent scripted --num-episodes 10

  # 2) ⭐ 真实训练数据: 用训练好的 PPO 策略 + 视频
  python collect_data.py --agent trained \\
      --checkpoint ../unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt \\
      --task Unitree-G1-Flat --num-episodes 100 --video

  # 3) 绝对关节目标 (mjlab 直接喂)
  python collect_data.py --agent trained --action-mode absolute --video

  # 4) Go2 四足 + 随机策略 + 视频
  python collect_data.py --task Unitree-Go2-Flat --agent random --video --num-episodes 50

  # 5) 多样化指令 (locomotion skills)
  python collect_data.py --agent trained --instruction-pool "walk forward,turn left,stop,walk backward"
        """,
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
                        help="默认语言指令 (default: 'walk forward')")
    parser.add_argument("--instruction-pool", type=str, default=None,
                        help="备选指令列表 (逗号分隔), 每 episode 随机抽一个")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (default: /workspace/data/{robot}_raw)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="步态速度, 仅 --agent scripted 生效 (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto | cuda:0 | cpu (default: auto)")
    parser.add_argument("--show-viewer", action="store_true",
                        help="显示 mjlab 可视化窗口 (需要 DISPLAY)")
    # ─── 新增选项 ───
    parser.add_argument("--agent", type=str, default="scripted",
                        choices=["scripted", "random", "zero", "trained"],
                        help="策略类型 (default: scripted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PPO checkpoint 路径 (.pt), --agent trained 必填")
    parser.add_argument("--action-mode", type=str, default="delta",
                        choices=["absolute", "delta", "relative_eef"],
                        help="动作空间 (default: delta, ⭐ GR00T N1.7 推荐)")
    parser.add_argument("--video", action="store_true",
                        help="采集 RGB 视频 (mjlab offscreen render, 每 episode 一个 mp4)")
    parser.add_argument("--video-height", type=int, default=224,
                        help="视频高度 (default: 224, GR00T 推荐)")
    parser.add_argument("--video-width", type=int, default=224,
                        help="视频宽度 (default: 224)")
    parser.add_argument("--video-fps", type=int, default=0,
                        help="视频帧率 (default: 0 = auto 根据仿真 dt 算; "
                             "GR00T 训练推荐 30)")
    parser.add_argument("--camera-name", type=str, default=None,
                        help="mjlab camera name (None=默认 front_view; "
                             "修复: 此选项目前无 effect, mjlab ViewerConfig 未暴露 "
                             "camera_name 字段, 需在任务 env_cfg 中修改。")
    parser.add_argument("--viser", action="store_true",
                        help="启用 viser 浏览器可视化 (显示 episode 进度 / 步数 / "
                             "指令 / base velocity; 需 pip install viser)")
    parser.add_argument("--viser-port", type=int, default=8080,
                        help="viser 服务器端口 (default: 8080)")
    parser.add_argument("--ee-body", type=str, default=None,
                        help="relative_eef 模式下的末端 body name (G1 默认 left_rubber_hand; "
                             "Go2 不支持。例: --ee-body right_rubber_hand)")
    args = parser.parse_args()

    if args.output_dir is None:
        robot = TASK_TO_ROBOT.get(args.task, "g1")
        args.output_dir = f"/workspace/data/{robot}_raw"

    # 解析 instruction_pool
    instruction_pool = None
    if args.instruction_pool:
        instruction_pool = [s.strip() for s in args.instruction_pool.split(",") if s.strip()]
        if not instruction_pool:
            instruction_pool = None

    collect_demonstrations(
        task_id=args.task,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        instruction=args.instruction,
        instruction_pool=instruction_pool,
        output_dir=args.output_dir,
        show_viewer=args.show_viewer,
        speed=args.speed,
        seed=args.seed,
        device=args.device,
        agent=args.agent,
        checkpoint=args.checkpoint,
        action_mode=args.action_mode,
        enable_video=args.video,
        video_height=args.video_height,
        video_width=args.video_width,
        video_fps=args.video_fps,
        camera_name=args.camera_name,
        viser=args.viser,
        viser_port=args.viser_port,
        ee_body=args.ee_body,
    )


if __name__ == "__main__":
    main()
