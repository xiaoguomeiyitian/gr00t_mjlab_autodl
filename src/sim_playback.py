"""
sim_playback.py — 在 MuJoCo 仿真中回放 robot_retargeter 动作并采集训练数据。

与 retarget_to_lerobot.py 的区别：
  - retarget_to_lerobot.py: 纯运动学（解析计算状态 + 单相机渲染）
  - sim_playback.py: 仿真回放（设置 qpos → mj_forward → 多相机渲染 + 真实 qvel）

流程:
  1. 加载 robot_retargeter 的 CSV/NPZ 运动数据
  2. 加载 MuJoCo MJCF 模型
  3. 逐帧设置 qpos，调用 mj_forward 获取真实 qvel
  4. 从仿真中获取多相机图像
  5. 构建与 collect_data.py 相同格式的 episode_*.npz + episode_*.mp4
  6. 可选：直接转换为 LeRobot v2 格式

用法:
    # 基本用法（输出到 sim_raw/）
    python -m src.sim_playback \
        --motion ../robot_retargeter/output_data/robot_motion/xxx_g1.csv \
        --robot g1 \
        --output output/g1_sim_raw \
        --num-episodes 5

    # 直接转换为 LeRobot v2
    python -m src.sim_playback \
        --motion ../robot_retargeter/output_data/robot_motion/xxx_g1.csv \
        --robot g1 \
        --output output/g1_sim_lerobot \
        --to-lerobot

    # 指定 MJCF 模型和 FPS
    python -m src.sim_playback \
        --motion xxx.csv --robot g1 --output output/g1_sim \
        --mjcf path/to/g1.xml --fps 30
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── 在无头服务器上强制使用 osmesa 后端 ───
import os
if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")


# ─────────────────── 机器人配置 ───────────────────
ROBOT_CONFIGS = {
    "g1": {
        "num_joints": 29,
        "state_dim": 71,
        "action_dim": 29,
        "camera_names": ["front", "wrist"],
    },
    "h1": {
        "num_joints": 20,
        "state_dim": 53,
        "action_dim": 20,
        "camera_names": ["front", "wrist"],
    },
    "h1_with_hand": {
        "num_joints": 46,
        "state_dim": 105,
        "action_dim": 46,
        "camera_names": ["front", "wrist"],
    },
    "h1_2": {
        "num_joints": 52,
        "state_dim": 117,
        "action_dim": 52,
        "camera_names": ["front", "wrist"],
    },
    "h2": {
        "num_joints": 32,
        "state_dim": 77,
        "action_dim": 32,
        "camera_names": ["front", "wrist"],
    },
    "go2": {
        "num_joints": 12,
        "state_dim": 37,
        "action_dim": 12,
        "camera_names": ["front", "back"],
    },
}


# ─────────────────── MJCF 模型查找 ───────────────────
def find_robot_mjcf(robot: str) -> str:
    """自动查找机器人 MJCF 模型。"""
    search_roots = [
        Path(__file__).resolve().parent.parent,  # gr00t_mjlab_autodl/
        Path(__file__).resolve().parent.parent.parent,  # unitree/
    ]

    candidates = {
        "g1": [
            "unitree/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
            "robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
            ".venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml",
        ],
        "h1": [
            "unitree/robot_retargeter/asset/robot/h1_description/mjcf/h1.xml",
            "robot_retargeter/asset/robot/h1_description/mjcf/h1.xml",
        ],
        "h1_with_hand": [
            "unitree/robot_retargeter/asset/robot/h1_description/mjcf/h1_with_hand.xml",
            "robot_retargeter/asset/robot/h1_description/mjcf/h1_with_hand.xml",
        ],
        "h1_2": [
            "unitree/robot_retargeter/asset/robot/h1_2_description/h1_2.xml",
            "robot_retargeter/asset/robot/h1_2_description/h1_2.xml",
        ],
        "h2": [
            "unitree/robot_retargeter/asset/robot/h2_description/H2.xml",
            "robot_retargeter/asset/robot/h2_description/H2.xml",
        ],
        "go2": [
            "unitree/robot_retargeter/asset/robot/a2_description/a2.xml",
            "robot_retargeter/asset/robot/a2_description/a2.xml",
        ],
    }

    for root in search_roots:
        for rel in candidates.get(robot, []):
            full_path = root / rel
            if full_path.exists():
                return str(full_path)

    raise FileNotFoundError(
        f"无法找到 {robot} 的 MJCF 模型。请指定 --mjcf 参数。"
    )


# ─────────────────── 运动学仿真回放 ───────────────────
class SimPlayback:
    """在 MuJoCo 中回放运动数据并采集观测。"""

    def __init__(
        self,
        mjcf_path: str,
        robot: str = "g1",
        image_size: tuple = (224, 224),
    ):
        """
        Args:
            mjcf_path: MJCF 模型文件路径
            robot: 机器人类型
            image_size: 渲染图像尺寸 (H, W)
        """
        import mujoco

        self.robot = robot
        self.image_size = image_size
        self.config = ROBOT_CONFIGS[robot]

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 创建渲染器
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.image_size[0],
            width=self.image_size[1],
        )

        # 获取可用相机
        self.camera_names = []
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self.camera_names.append(name)

        # 获取关节名称（排除 free joint）
        self.joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                jnt_type = self.model.jnt_type[i]
                # type 0 = free joint, skip it
                if jnt_type != 0:
                    self.joint_names.append(name)

        # 获取关节 qpos 地址
        self.joint_qpos_indices = []
        self.joint_qvel_indices = []
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type != 0:
                # 获取该 joint 在 qpos 和 qvel 中的起始地址
                qpos_addr = self.model.jnt_qposadr[i]
                qvel_addr = qpos_addr - 1
                self.joint_qpos_indices.append(qpos_addr)
                self.joint_qvel_indices.append(qvel_addr)

        # free joint 的 qpos 地址（前 7 个：pos(3) + quat(4)）
        self.base_pos_idx = 0  # qpos[0:3]
        self.base_quat_idx = 3  # qpos[3:7]

        print(f"  ✅ MuJoCo 仿真初始化: {Path(mjcf_path).name}")
        print(f"     图像尺寸: {self.image_size}")
        print(f"     可用相机: {self.camera_names}")
        print(f"     关节数: {len(self.joint_names)}")

    def step(self, base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray):
        """
        设置机器人状态并执行一步前向运动学。

        Args:
            base_pos: (3,) 基座位置
            base_quat: (4,) 基座四元数 wxyz
            joint_pos: (N,) 关节位置
        """
        import mujoco

        # 设置基座
        self.data.qpos[self.base_pos_idx:self.base_pos_idx + 3] = base_pos
        self.data.qpos[self.base_quat_idx:self.base_quat_idx + 4] = base_quat

        # 设置关节
        for i, idx in enumerate(self.joint_qpos_indices):
            if i < len(joint_pos):
                self.data.qpos[idx] = joint_pos[i]

        # 零速度
        self.data.qvel[:] = 0.0

        # 前向运动学
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """
        获取当前状态向量。

        Returns:
            state: (state_dim,) 状态向量
                [joint_pos(N), joint_vel(N), base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3)]
        """
        num_joints = self.config["num_joints"]

        # 关节位置
        joint_pos = np.zeros(num_joints, dtype=np.float32)
        for i, idx in enumerate(self.joint_qpos_indices):
            if i < num_joints:
                joint_pos[i] = self.data.qpos[idx]

        # 关节速度（从仿真获取）
        joint_vel = np.zeros(num_joints, dtype=np.float32)
        for i, idx in enumerate(self.joint_qpos_indices):
            if i < num_joints:
                # qvel 和 qpos 的索引布局相同（对于 hinge joint）
                joint_vel[i] = self.data.qvel[self.joint_qvel_indices[i]]

        # 基座状态
        base_pos = self.data.qpos[self.base_pos_idx:self.base_pos_idx + 3].astype(np.float32)
        base_quat = self.data.qpos[self.base_quat_idx:self.base_quat_idx + 4].astype(np.float32)

        # 基座速度（从 free joint 的 qvel 获取）
        base_lin_vel = self.data.qvel[0:3].astype(np.float32)
        base_ang_vel = self.data.qvel[3:6].astype(np.float32)

        state = np.concatenate([
            joint_pos,      # (N,)
            joint_vel,      # (N,)
            base_pos,       # (3,)
            base_quat,      # (4,)
            base_lin_vel,   # (3,)
            base_ang_vel,   # (3,)
        ])

        return state.astype(np.float32)

    def render_camera(self, camera_name: Optional[str] = None) -> np.ndarray:
        """
        渲染指定相机的图像。

        Args:
            camera_name: 相机名称（None 则使用第一个可用相机）

        Returns:
            image: (H, W, 3) RGB 图像
        """
        import mujoco

        cam = camera_name if camera_name and camera_name in self.camera_names else None
        self.renderer.update_scene(self.data, camera=cam)
        img = self.renderer.render()
        return img

    def render_all_cameras(self) -> dict:
        """
        渲染所有可用相机的图像。

        Returns:
            images: {"camera_name": (H, W, 3) RGB image}
        """
        images = {}
        for cam_name in self.camera_names:
            images[cam_name] = self.render_camera(cam_name)
        return images


# ─────────────────── 运动数据加载 ───────────────────
def load_motion_data(motion_file: str, fps: Optional[float] = None):
    """
    加载 robot_retargeter 的运动数据。

    Args:
        motion_file: CSV 或 NPZ 文件路径
        fps: 帧率（None 则使用默认值）

    Returns:
        base_pos: (T, 3) 基座位置
        base_quat: (T, 4) 基座四元数 wxyz
        joint_pos: (T, N) 关节位置
        actual_fps: 帧率
    """
    from src.retarget_motion_loader import RetargetMotionLoader

    loader = RetargetMotionLoader(motion_file, fps=fps)
    return loader.load()


# ─────────────────── 数据采集 ───────────────────
def collect_from_sim(
    motion_file: str,
    robot: str = "g1",
    output_dir: str = "output/g1_sim_raw",
    mjcf_path: Optional[str] = None,
    num_episodes: int = 5,
    episode_length: int = 300,
    fps: Optional[float] = None,
    image_size: tuple = (224, 224),
    action_mode: str = "delta",
    task_description: Optional[str] = None,
):
    """
    在仿真中回放运动并采集训练数据。

    Args:
        motion_file: robot_retargeter 的运动文件
        robot: 机器人类型
        output_dir: 输出目录
        mjcf_path: MJCF 模型路径
        num_episodes: episode 数量
        episode_length: 每 episode 步数
        fps: 帧率
        image_size: 图像尺寸
        action_mode: 动作模式
        task_description: 任务描述

    Returns:
        stats: 采集统计信息
    """
    import mujoco

    config = ROBOT_CONFIGS[robot]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找 MJCF
    if mjcf_path is None:
        mjcf_path = find_robot_mjcf(robot)

    # 加载运动数据
    print(f"📂 加载运动数据: {motion_file}")
    base_pos, base_quat, joint_pos, actual_fps = load_motion_data(motion_file, fps=fps)
    T = joint_pos.shape[0]
    dt = 1.0 / actual_fps
    print(f"   帧数: {T}, FPS: {actual_fps}, 时长: {T / actual_fps:.1f}s")

    # 初始化仿真
    sim = SimPlayback(mjcf_path, robot=robot, image_size=image_size)

    # 推断任务描述
    if task_description is None:
        from src.configs.motion_labels import get_motion_label
        task_description = get_motion_label(motion_file)

    print(f"\n🎬 开始仿真回放采集...")
    print(f"   机器人: {robot}")
    print(f"   任务: {task_description}")
    print(f"   Episodes: {num_episodes}")
    print(f"   每 episode 步数: {episode_length}")
    print(f"   动作模式: {action_mode}")
    print(f"   输出: {output_path}")
    print()

    # 计算每 episode 使用的运动片段长度
    # 如果运动数据不够长，循环使用
    if T < episode_length:
        print(f"   ⚠️  运动数据 ({T} 帧) < episode 长度 ({episode_length} 帧)")
        print(f"   将循环使用运动数据")

    stats = {
        "total_steps": 0,
        "start_time": time.time(),
        "episodes": [],
    }

    for ep_idx in range(num_episodes):
        ep_data = _collect_episode(
            sim=sim,
            base_pos=base_pos,
            base_quat=base_quat,
            joint_pos=joint_pos,
            ep_idx=ep_idx,
            output_path=output_path,
            config=config,
            num_joints=config["num_joints"],
            episode_length=episode_length,
            action_mode=action_mode,
            actual_fps=actual_fps,
            task_description=task_description,
        )
        stats["episodes"].append(ep_data)
        stats["total_steps"] += ep_data["steps"]

        elapsed = time.time() - stats["start_time"]
        avg_time = elapsed / (ep_idx + 1)
        eta = avg_time * (num_episodes - ep_idx - 1)

        print(
            f"  ✅ Episode {ep_idx + 1}/{num_episodes}  "
            f"steps={ep_data['steps']}  "
            f"ETA={eta:.0f}s"
        )

    elapsed = time.time() - stats["start_time"]
    print(f"\n📊 采集完成")
    print(f"   总步数: {stats['total_steps']}")
    print(f"   总耗时: {elapsed:.1f}s")
    print(f"   输出目录: {output_path}")

    # 保存 metadata
    _save_metadata(output_path, stats, robot, task_description, action_mode,
                    num_episodes, episode_length, actual_fps, image_size, T, motion_file)

    return stats


def _collect_episode(
    sim: "SimPlayback",
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    ep_idx: int,
    output_path: Path,
    config: dict,
    num_joints: int,
    episode_length: int,
    action_mode: str,
    actual_fps: float,
    task_description: str,
) -> dict:
    """采集单个 episode。"""
    T = joint_pos.shape[0]
    camera_names = config["camera_names"]

    frames = {cam: [] for cam in camera_names}
    states = []
    actions = []
    total_reward = 0.0

    for step in range(episode_length):
        # 循环使用运动数据
        t = step % T

        # 设置仿真状态
        sim.step(base_pos[t], base_quat[t], joint_pos[t])

        # 获取状态
        state = sim.get_state()

        # 计算动作
        if action_mode == "delta":
            t_next = (step + 1) % T
            action = (joint_pos[t_next] - joint_pos[t]).astype(np.float32)
        elif action_mode == "absolute":
            action = joint_pos[t].astype(np.float32)
        else:
            action = np.zeros(num_joints, dtype=np.float32)

        # 渲染多相机
        images = sim.render_all_cameras()

        # 保存数据
        for cam in camera_names:
            if cam in images:
                frames[cam].append(images[cam])
            else:
                frames[cam].append(np.zeros((sim.image_size[0], sim.image_size[1], 3), dtype=np.uint8))

        states.append(state)
        actions.append(action)
        total_reward += 0.0  # 仿真回放无 reward

    steps = len(states)

    # 保存 npz
    npz_path = output_path / f"episode_{ep_idx:04d}.npz"
    np.savez_compressed(
        str(npz_path),
        states=np.stack(states).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.zeros(steps, dtype=np.float32),
        task_name=task_description,
        robot=config.get("robot", "unknown"),
        action_mode=action_mode,
    )

    # 保存 mp4（每个相机一个视频）
    for cam in camera_names:
        if frames[cam]:
            mp4_path = output_path / f"episode_{ep_idx:04d}_{cam}.mp4"
            _save_video(frames[cam], str(mp4_path), fps=actual_fps)

    return {
        "episode": ep_idx,
        "steps": steps,
        "reward": total_reward,
        "npz": str(npz_path),
    }


def _save_video(frames: list, path: str, fps: float = 30.0):
    """保存图像序列为 mp4 视频。"""
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for img in frames:
        if img.ndim == 3 and img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        writer.write(bgr)

    writer.release()


def _save_metadata(output_path, stats, robot, task_description, action_mode,
                   num_episodes, episode_length, fps, image_size, motion_frames, motion_file):
    """保存采集 metadata。"""
    meta = {
        "robot": robot,
        "task": task_description,
        "action_mode": action_mode,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "fps": fps,
        "image_size": list(image_size),
        "state_dim": ROBOT_CONFIGS[robot]["state_dim"],
        "action_dim": ROBOT_CONFIGS[robot]["action_dim"],
        "camera_names": ROBOT_CONFIGS[robot]["camera_names"],
        "source": "sim_playback",
        "source_file": str(motion_file),
        "motion_frames": motion_frames,
        "episodes": stats["episodes"],
        "total_steps": stats["total_steps"],
    }

    meta_path = output_path / "collection_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ─────────────────── CLI ───────────────────
def main():
    parser = argparse.ArgumentParser(
        description="在 MuJoCo 仿真中回放 robot_retargeter 动作并采集训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion", type=str, help="robot_retargeter 运动文件 (CSV/NPZ)")

    parser.add_argument("--robot", type=str, default="g1",
                        choices=list(ROBOT_CONFIGS.keys()),
                        help="机器人类型")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--mjcf", type=str, default=None, help="MJCF 模型路径")
    parser.add_argument("--num-episodes", type=int, default=5, help="采集 episode 数量")
    parser.add_argument("--episode-length", type=int, default=300, help="每 episode 步数")
    parser.add_argument("--fps", type=float, default=None, help="帧率")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                        help="图像尺寸 H W")
    parser.add_argument("--action-mode", type=str, default="delta",
                        choices=["absolute", "delta"], help="动作模式")
    parser.add_argument("--task", type=str, default=None, help="任务描述")

    args = parser.parse_args()

    collect_from_sim(
        motion_file=args.motion,
        robot=args.robot,
        output_dir=args.output,
        mjcf_path=args.mjcf,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        fps=args.fps,
        image_size=tuple(args.image_size),
        action_mode=args.action_mode,
        task_description=args.task,
    )


if __name__ == "__main__":
    main()
