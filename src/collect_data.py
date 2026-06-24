#!/usr/bin/env python3
"""
数据收集器 — 基于 unitree_rl_mjlab 官方仿真环境收集 G1/Go2 演示数据。

支持 4 种 Agent (scripted/random/zero/trained) 和 3 种动作空间 (absolute/delta/relative_eef)。

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

    # 2) 真实训练数据: 用 PPO 策略回放, 开启视频, 100 episode
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
    """asdict 兼容版: 支持 frozen dataclass, 普通 class, 嵌套 dict 等."""
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

    关节顺序与 unitree_rl_mjlab MJCF (g1.xml / g1_23dof.xml) 一致。
    """
    t = step_idx * dt
    freq = speed * 2.0  # 步频

    if num_joints == 23:
        targets = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            0.0,
            0.35, 0.18, 0.0, 0.87, 0.0,
            0.35, -0.18, 0.0, 0.87, 0.0,
        ], dtype=np.float32)
    else:
        targets = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
            0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

    phase_l = np.sin(2 * np.pi * freq * t)
    targets[0] += 0.3 * phase_l
    targets[1] += 0.05 * np.sin(phase_l)
    targets[3] += -0.4 * max(phase_l, 0)

    phase_r = np.sin(2 * np.pi * freq * t + np.pi)
    targets[6] += 0.3 * phase_r
    targets[7] += 0.05 * np.sin(phase_r)
    targets[9] += -0.4 * max(phase_r, 0)

    if num_joints == 23:
        pass
    else:
        targets[13] += 0.02 * np.sin(2 * np.pi * freq * t)

    arm_l_start = 13 if num_joints == 23 else 15
    arm_r_start = 18 if num_joints == 23 else 22

    arm_phase = np.sin(2 * np.pi * freq * t)
    targets[arm_l_start] += 0.2 * arm_phase
    targets[arm_l_start + 3] += -0.1 * max(arm_phase, 0)
    targets[arm_r_start] += 0.2 * (-arm_phase)
    targets[arm_r_start + 3] += -0.1 * max(-arm_phase, 0)

    return targets


def gait_generator_go2(
    step_idx: int,
    dt: float = 0.02,
    speed: float = 0.5,
    command: str = "walk forward",
    **kwargs,
) -> np.ndarray:
    """Go2 步态生成器 — 产生 12 维关节目标。

    关节顺序与 unitree_rl_mjlab MJCF (go2.xml) 一致。
    """
    t = step_idx * dt
    freq = speed * 2.5

    from configs.go2_config import GO2_JOINT_NAMES, GO2_DEFAULT_JOINT_ANGLES
    default = np.array(
        [GO2_DEFAULT_JOINT_ANGLES[n] for n in GO2_JOINT_NAMES], dtype=np.float32
    )

    targets = default.copy()
    phase = np.sin(2 * np.pi * freq * t)

    for i, (hip_idx, thigh_idx, calf_idx) in enumerate([
        (0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11),
    ]):
        p = phase if i % 2 == 0 else -phase
        targets[hip_idx] += 0.15 * p
        targets[thigh_idx] += 0.2 * abs(p)
        targets[calf_idx] += -0.3 * max(p, 0)

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

TASK_TO_ROBOT: dict[str, str] = {
    # mjlab 1.2.0 任务名称格式: Mjlab-Velocity-{Terrain}-Unitree-{Robot}
    "Mjlab-Velocity-Flat-Unitree-G1": "g1",
    "Mjlab-Velocity-Rough-Unitree-G1": "g1",
    "Mjlab-Velocity-Flat-Unitree-G1-23Dof": "g1",
    "Mjlab-Velocity-Rough-Unitree-G1-23Dof": "g1",
    "Mjlab-Velocity-Flat-Unitree-Go2": "go2",
    "Mjlab-Velocity-Rough-Unitree-Go2": "go2",
    "Mjlab-Velocity-Flat-Unitree-A2": "a2",
    "Mjlab-Velocity-Rough-Unitree-A2": "a2",
    "Mjlab-Velocity-Flat-Unitree-R1": "r1",
    "Mjlab-Velocity-Rough-Unitree-R1": "r1",
    "Mjlab-Velocity-Flat-Unitree-H1_2": "h1_2",
    "Mjlab-Velocity-Rough-Unitree-H1_2": "h1_2",
    "Mjlab-Velocity-Flat-Unitree-H2": "h2",
    "Mjlab-Velocity-Rough-Unitree-H2": "h2",
    "Mjlab-Velocity-Flat-Unitree-As2": "as2",
    "Mjlab-Velocity-Rough-Unitree-As2": "as2",
}


# ──────────────────────────────────────────────────────────────────────────────
# Viser 浏览器 3D viewer (可选, 实时查看机器人运动)
# ──────────────────────────────────────────────────────────────────────────────

# 延迟导入 (viser 是可选依赖)
_ViserViewer3D: Any = None


def _get_viser_viewer():
    """延迟加载 AsyncViser3DViewer (避免 viser 未安装时崩溃)."""
    global _ViserViewer3D
    if _ViserViewer3D is None:
        try:
            from viser_3d_viewer import AsyncViser3DViewer
            _ViserViewer3D = AsyncViser3DViewer
        except ImportError:
            logger.warning("viser_3d_viewer 未安装, 3D viewer 不可用")
            _ViserViewer3D = None
    return _ViserViewer3D

    """浏览器可视化: 3D 机器人运动 + 进度面板 + 自适应 FPS.

    自适应策略:
    自适应策略:
      - 有浏览器连接: 按 viser_fps 渲染 (默认 30 FPS)
      - 无浏览器连接: 暂停渲染, 全速采集
    """

    def __init__(self, env=None, port: int = 20006, viser_fps: float = 30.0):
        self._viewer_3d = None
        self._server = None
        self.port = port
        self.url = f"http://localhost:{port}"
        self._viser_fps = viser_fps
        self._fps_adjusted = False  # 是否已根据连接状态调整过 FPS

        Viewer3D = _get_viser_viewer()
        if Viewer3D is not None and env is not None:
            try:
                self._viewer_3d = Viewer3D(env=env, port=port, frame_rate=viser_fps)
                self._viewer_3d.start()
                self.url = self._viewer_3d.url
                logger.info("🌐 Viser 3D viewer 已启动: %s (FPS=%d, 自适应=%s)",
                            self.url, int(viser_fps), "有连接渲染/无连接暂停")
                return
            except Exception as e:
                logger.warning("3D viewer 启动失败, 回退到文本模式: %s", e)

        try:
            import viser
            self._server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
            with self._server.gui.add_folder("📊 进度"):
                self.ep_text = self._server.gui.add_text("Episode", initial_value="-/-", disabled=True)
                self.step_text = self._server.gui.add_text("Step", initial_value="-/-", disabled=True)
                self.instr_text = self._server.gui.add_text("指令", initial_value="(初始化中...)", disabled=True)
            with self._server.gui.add_folder("🤖 状态"):
                self.base_vel_text = self._server.gui.add_text("Base vel (m/s)", initial_value="-", disabled=True)
                self.action_text = self._server.gui.add_text("关节目标范围", initial_value="-", disabled=True)
            logger.info("🌐 Viser 文本 viewer 已启动: %s", self.url)
        except ImportError:
            logger.warning("viser 未安装, 跳过浏览器 viewer")

    def update(self, ep: int, total_ep: int, step: int, total_step: int,
               instruction: str, base_vel=None, joint_targets=None) -> None:
        # 3D viewer: 自适应 FPS (有连接按 viser_fps, 无连接暂停)
        if self._viewer_3d is not None:
            # 首次 update 时根据连接状态调整 FPS
            if not self._fps_adjusted:
                if self._viewer_3d.has_connections:
                    self._viewer_3d.set_fps(self._viser_fps)
                    logger.info("🌐 检测到浏览器连接, 渲染 FPS=%d", int(self._viser_fps))
                else:
                    self._viewer_3d.set_fps(0)  # 暂停渲染, 全速采集
                    logger.info("🌐 无浏览器连接, 暂停渲染以全速采集")
                self._fps_adjusted = True
            self._viewer_3d.update()

        # 文本 viewer (始终更新, 开销极小)
        if self._server is not None:
            try:
                if hasattr(self, 'ep_text'):
                    self.ep_text.value = f"{ep+1}/{total_ep}"
                    self.step_text.value = f"{step}/{total_step}"
                    self.instr_text.value = instruction
                if base_vel is not None and hasattr(self, 'base_vel_text') and len(base_vel) >= 3:
                    vx, vy, vz = float(base_vel[0]), float(base_vel[1]), float(base_vel[2])
                    self.base_vel_text.value = f"({vx:+.2f}, {vy:+.2f}, {vz:+.2f})"
                if joint_targets is not None and hasattr(self, 'action_text') and len(joint_targets) > 0:
                    jt = np.asarray(joint_targets)
                    self.action_text.value = f"[{jt.min():.2f}, {jt.max():.2f}]"
            except Exception:
                pass

    def close(self) -> None:
        if self._viewer_3d is not None:
            self._viewer_3d.stop()
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────# 数据收集
# ──────────────────────────────────────────────────────────────────────────────


def collect_demonstrations(
    task_id: str = "Mjlab-Velocity-Flat-Unitree-G1",
    num_episodes: int = 100,
    episode_length: int = 200,
    instruction: str = "walk forward",
    output_dir: str = "/workspace/data/raw",
    show_viewer: bool = False,
    speed: float = 0.5,
    seed: int = 42,
    num_envs: int = 1,
    device: str = "auto",
    agent: str = "scripted",
    checkpoint: str | None = None,
    action_mode: str = "delta",
    enable_video: bool = False,
    video_height: int = 224,
    video_width: int = 224,
    video_fps: int = 0,
    camera_name: str | None = None,
    instruction_pool: list[str] | None = None,
    viser: bool = True,
    viser_port: int = 20006,
    viser_fps: float = 30.0,
    ee_body: str | None = None,
) -> str:
    if device == "auto":
        try:
            import torch as _torch
            device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
            logger.warning("torch 未安装, 使用 CPU")

    robot = TASK_TO_ROBOT.get(task_id)
    if robot is None:
        raise ValueError(
            f"不支持的任务: {task_id}\n"
            f"可选: {list(TASK_TO_ROBOT.keys())}"
        )
    if robot == "g1":
        from configs.g1_config import (
            G1_NUM_JOINTS, G1_DEFAULT_JOINT_ANGLES, G1_DT, G1_JOINT_NAMES,
            G1_23DOF_NUM_JOINTS, G1_23DOF_DEFAULT_JOINT_ANGLES,
            G1_23DOF_JOINT_NAMES, G1_23DOF_DT,
        )
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
        from configs.go2_config import GO2_NUM_JOINTS, GO2_DT
        robot_joint_counts = {
            "go2": 12, "a2": 12, "as2": 12,
            "r1": 24, "h1_2": 27, "h2": 29,
        }
        num_joints = robot_joint_counts.get(robot, GO2_NUM_JOINTS)
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

    env_raw = None
    env_wrapped = None
    policy_fn: Callable | None = None
    use_mjlab = False
    viewer = None

    try:
        import torch as _torch
        rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
        if rl_mjlab_root.exists() and str(rl_mjlab_root) not in sys.path:
            sys.path.insert(0, str(rl_mjlab_root))

        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        import src.tasks  # noqa: F401

        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = 1

        if enable_video:
            env_cfg.viewer.width = video_width
            env_cfg.viewer.height = video_height

        render_mode = "rgb_array" if enable_video else None
        env_raw = ManagerBasedRlEnv(
            cfg=env_cfg, device=device, render_mode=render_mode
        )
        logger.info("✅ mjlab 环境创建成功 (device=%s, render=%s)",
                    device, render_mode or "off")
        use_mjlab = True

        if viser:
            try:
                viewer = ViserViewer(env=env_raw, port=viser_port, viser_fps=viser_fps)
            except ImportError:
                logger.warning("viser 未安装, 跳过浏览器 viewer")

        if agent == "trained":
            if not checkpoint or not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"--agent trained 必须指定 --checkpoint, 且文件必须存在\n"
                    f"  当前: {checkpoint}\n"
                    f"  提示: 先跑 unitree_rl_mjlab/scripts/train.py 训一个 PPO 模型"
                )

            agent_cfg = load_rl_cfg(task_id)
            env_wrapped = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

            try:
                runner_cls = load_runner_cls(task_id)
            except KeyError:
                runner_cls = None
            runner_cls = runner_cls or MjlabOnPolicyRunner
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    imageio = None
    if enable_video:
        try:
            import imageio.v2 as imageio
            logger.info("✅ imageio 可用, 将保存 mp4 视频")
        except ImportError:
            try:
                import imageio
                logger.info("✅ imageio (v1) 可用, 将保存 mp4 视频")
            except ImportError:
                logger.warning("⚠️ imageio 未安装, 视频将保存为 npz 帧序列")
                imageio = None

    if instruction_pool:
        logger.info("  指令池: %d 条", len(instruction_pool))
    else:
        instruction_pool = [instruction]

    state_dim = num_joints * 2 + 3 + 4 + 3 + 3
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

    action_mode_ee_body = ee_body
    if action_mode == "relative_eef" and robot != "g1":
        logger.warning("relative_eef 模式仅 G1 有末端执行器, Go2 数据将为零")
    elif action_mode == "relative_eef" and action_mode_ee_body is None:
        action_mode_ee_body = "left_rubber_hand"
        logger.info("relative_eef 模式默认 EE body: %s", action_mode_ee_body)

    # ─── 视频流式写入器 (避免帧累积导致 OOM) ─────────────────────
    video_writer = None
    vid_fps = video_fps if video_fps > 0 else int(round(1.0 / dt))
    if enable_video and imageio is not None:
        vid_path = output_path / "episodes.mp4"
        try:
            video_writer = imageio.get_writer(
                str(vid_path), fps=vid_fps,
                codec="libx264", quality=5,
                macro_block_size=16,
            )
            logger.info("视频将流式写入: %s (%dfps)", vid_path, vid_fps)
        except Exception as e:
            logger.warning("imageio.get_writer 失败, 将使用 frames.npz: %s", e)
            video_writer = None

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

        for step_idx in range(episode_length):
            # ─── 1) 决定动作 ─────────────────────────────────────
            if agent == "scripted":
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
                    if robot == "g1":
                        joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                                command=ep_instruction,
                                                num_joints=num_joints)
                    else:
                        joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                                command=ep_instruction,
                                                num_joints=num_joints)
            else:
                if robot == "g1":
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)
                else:
                    joint_targets = gait_fn(step_idx, dt=dt, speed=ep_speed,
                                            command=ep_instruction,
                                            num_joints=num_joints)

            # ─── 2) 推 mjlab 环境 (有的话) ─────────────────────────
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

            # mjlab 返回的是相对值, 还原为绝对值
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

            # ─── 5) 渲染视频 (流式写入, 避免 OOM) ──────────────────
            if enable_video and video_writer is not None:
                frame = _render_frame(env_raw, ep, step_idx)
                if frame is not None:
                    try:
                        video_writer.append_data(frame)
                    except Exception as e:
                        if step_idx == 0:
                            logger.debug("视频帧写入失败: %s", e)

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

        if (ep + 1) % 10 == 0:
            logger.info("收集进度: %d/%d episodes (ep_instruction='%s')",
                        ep + 1, num_episodes, ep_instruction[:30])

    # ── 6. 关闭视频写入器 & 环境 ────────────────────────────────────
    if video_writer is not None:
        try:
            video_writer.close()
            logger.info("视频已保存: %s", output_path / "episodes.mp4")
        except Exception as e:
            logger.warning("视频写入器关闭失败: %s", e)

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
        #   sim.data / sim.model 是 Warp DataBridge/ModelBridge, 没有 .xpos
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

  # 2) 真实训练数据: 用训练好的 PPO 策略 + 视频
  python collect_data.py --agent trained \\
      --checkpoint ../unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt \\
      --task Mjlab-Velocity-Flat-Unitree-G1 --num-episodes 100 --video

  # 3) 绝对关节目标 (mjlab 直接喂)
  python collect_data.py --agent trained --action-mode absolute --video

  # 4) Go2 四足 + 随机策略 + 视频
  python collect_data.py --task Mjlab-Velocity-Flat-Unitree-Go2 --agent random --video --num-episodes 50

  # 5) 多样化指令 (locomotion skills)
  python collect_data.py --agent trained --instruction-pool "walk forward,turn left,stop,walk backward"
        """,
    )
    parser.add_argument(
        "--task", type=str, default="Mjlab-Velocity-Flat-Unitree-G1",
        choices=list(TASK_TO_ROBOT.keys()),
        help="mjlab 任务 ID (default: Mjlab-Velocity-Flat-Unitree-G1)",
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
                        help="动作空间 (default: delta)")
    parser.add_argument("--video", action="store_true",
                        help="采集 RGB 视频 (mjlab offscreen render, 每 episode 一个 mp4)")
    parser.add_argument("--video-height", type=int, default=224,
                        help="视频高度 (default: 224, GR00T 推荐)")
    parser.add_argument("--video-width", type=int, default=224,
                        help="视频宽度 (default: 224)")
    parser.add_argument("--video-fps", type=int, default=0,
                        help="视频帧率 (default: 0 = auto 根据仿真 dt 算; "
                             "GR00T 训练推荐 30)")
parser.add_argument("--camera-name", type=str, default=None, help="mjlab camera name (None=默认 front_view; "
"修复: 此选项目前无 effect, mjlab ViewerConfig 未暴露 "
"camera_name 字段, 需在任务 env_cfg 中修改。")
parser.add_argument("--no-viser", action="store_true", help="禁用 viser 浏览器可视化 (默认启用, 有连接时按 viser-fps 渲染)")
parser.add_argument("--viser-port", type=int, default=20006, help="viser 服务器端口 (default: 20006)")
parser.add_argument("--viser-fps", type=float, default=30.0, help="viser 渲染 FPS (default: 30, 范围 1-30; "
"有浏览器连接时按此 FPS, 无连接时暂停渲染)")
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
        viser=not args.no_viser,
        viser_port=args.viser_port,
        viser_fps=args.viser_fps,
        ee_body=args.ee_body,
    )


if __name__ == "__main__":
    main()
