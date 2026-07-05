"""retarget_to_lerobot.py — robot_retargeter 数据 → LeRobot v2 格式。"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.retarget_motion_loader import RetargetMotionLoader
from src.configs.motion_labels import get_motion_label


ROBOT_CONFIGS = {
    "g1": {
        "num_joints": 29,
        "state_dim": 71,
        "action_dim": 29,
        "camera_names": ["front", "wrist"],
        "mjcf_path": None,
    },
    "h1": {
        "num_joints": 20,
        "state_dim": 53,
        "action_dim": 20,
        "camera_names": ["front", "wrist"],
        "mjcf_path": None,
    },
    "h1_with_hand": {
        "num_joints": 46,
        "state_dim": 105,
        "action_dim": 46,
        "camera_names": ["front", "wrist"],
        "mjcf_path": None,
    },
    "h1_2": {
        "num_joints": 52,
        "state_dim": 117,
        "action_dim": 52,
        "camera_names": ["front", "wrist"],
        "mjcf_path": None,
    },
    "h2": {
        "num_joints": 32,
        "state_dim": 77,
        "action_dim": 32,
        "camera_names": ["front", "wrist"],
        "mjcf_path": None,
    },
    "go2": {
        "num_joints": 12,
        "state_dim": 37,
        "action_dim": 12,
        "camera_names": ["front", "back"],
        "mjcf_path": None,
    },
}


def compute_state_vector(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    dt: float,
) -> np.ndarray:
    T, num_joints = joint_pos.shape
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    base_lin_vel = np.gradient(base_pos, dt, axis=0)
    base_ang_vel = compute_angular_velocity(base_quat, dt)

    states = np.concatenate([
        joint_pos,
        joint_vel,
        base_pos,
        base_quat,
        base_lin_vel,
        base_ang_vel,
    ], axis=1)

    return states.astype(np.float32)


def compute_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    T = quat_wxyz.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float32)

    for i in range(1, T - 1):
        q_prev = quat_wxyz[i - 1]
        q_next = quat_wxyz[i + 1]

        w1, x1, y1, z1 = q_prev
        w2, x2, y2, z2 = q_next

        q_conj = np.array([w1, -x1, -y1, -z1])

        w_rel = w2 * q_conj[0] - x2 * q_conj[1] - y2 * q_conj[2] - z2 * q_conj[3]
        x_rel = w2 * q_conj[1] + x2 * q_conj[0] + y2 * q_conj[3] - z2 * q_conj[2]
        y_rel = w2 * q_conj[2] - x2 * q_conj[3] + y2 * q_conj[0] + z2 * q_conj[1]
        z_rel = w2 * q_conj[3] + x2 * q_conj[2] - y2 * q_conj[1] + z2 * q_conj[0]

        # 处理四元数双覆盖：w_rel < 0 时翻转 q_rel 到短弧，避免大旋转方向错误
        if w_rel < 0:
            w_rel, x_rel, y_rel, z_rel = -w_rel, -x_rel, -y_rel, -z_rel

        angle = 2 * np.arctan2(np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2), w_rel)
        axis_norm = np.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)

        if abs(angle) > 1e-8 and axis_norm > 1e-8:
            axis = np.array([x_rel, y_rel, z_rel]) / axis_norm
            ang_vel[i] = axis * (angle / (2 * dt))

    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def compute_delta_actions(joint_pos: np.ndarray) -> np.ndarray:
    actions = np.zeros_like(joint_pos)
    actions[:-1] = joint_pos[1:] - joint_pos[:-1]
    actions[-1] = actions[-2] if len(actions) > 1 else 0
    return actions.astype(np.float32)


def sliding_window_split(
    total_length: int,
    window_size: int = 300,
    overlap: float = 0.5,
) -> list:
    stride = max(1, int(window_size * (1 - overlap)))
    episodes = []

    if total_length <= window_size:
        episodes.append((0, total_length))
        return episodes

    start = 0
    while start + window_size <= total_length:
        episodes.append((start, start + window_size))
        start += stride

    if start < total_length and total_length - start >= window_size // 2:
        episodes.append((total_length - window_size, total_length))

    return episodes


def convert_to_lerobot(
    motion_file: str,
    robot: str = "g1",
    output_dir: str = "output/g1_from_retarget",
    episode_length: int = 300,
    overlap: float = 0.5,
    fps: Optional[float] = None,
    task_description: Optional[str] = None,
    render_videos: bool = True,
    mjcf_path: Optional[str] = None,
):
    if robot not in ROBOT_CONFIGS:
        raise ValueError(f"不支持的机器人: {robot}，可选: {list(ROBOT_CONFIGS.keys())}")

    config = ROBOT_CONFIGS[robot]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 转换: {motion_file}")
    print(f"   机器人: {robot}")
    print(f"   输出: {output_path}")
    print()

    loader = RetargetMotionLoader(motion_file, fps=fps)
    base_pos, base_quat, joint_pos, actual_fps = loader.load()
    dt = 1.0 / actual_fps

    T = joint_pos.shape[0]
    print(f"   总帧数: {T}, FPS: {actual_fps}, 时长: {T / actual_fps:.1f}s")
    print()

    print("📊 计算状态向量...")
    states = compute_state_vector(base_pos, base_quat, joint_pos, dt)
    print(f"   states: {states.shape}")

    print("🔧 计算 delta 动作...")
    actions = compute_delta_actions(joint_pos)
    print(f"   actions: {actions.shape}")

    if task_description is None:
        task_description = get_motion_label(motion_file)
    print(f"   任务描述: {task_description}")
    print()

    actual_episode_length = min(episode_length, T)
    if actual_episode_length < episode_length:
        print(f"   ⚠️  数据帧数 ({T}) < 请求的 episode 长度 ({episode_length})")
        print(f"   自动调整为: {actual_episode_length} 帧")
    print("✂️ 切分 episode...")
    episodes = sliding_window_split(T, window_size=actual_episode_length, overlap=overlap)
    print(f"   切分为 {len(episodes)} 个 episode")
    print()

    meta_dir = output_path / "meta"
    data_dir = output_path / "data" / "chunk-000"
    videos_dir = output_path / "videos" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    num_joints = config["num_joints"]
    modality = {
        "state": {
            "joint_pos": {"start": 0, "end": num_joints},
            "joint_vel": {"start": num_joints, "end": 2 * num_joints},
            "base_pos": {"start": 2 * num_joints, "end": 2 * num_joints + 3},
            "base_quat": {"start": 2 * num_joints + 3, "end": 2 * num_joints + 7},
            "base_lin_vel": {"start": 2 * num_joints + 7, "end": 2 * num_joints + 10},
            "base_ang_vel": {"start": 2 * num_joints + 10, "end": 2 * num_joints + 13},
        },
        "action": {
            "joint_position_delta": {"start": 0, "end": num_joints},
        },
        "video": {
            cam: {"original_key": f"observation.images.{cam}"}
            for cam in config["camera_names"]
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task_description": task_description}) + "\n")

    print("📝 处理 episodes...")
    all_rows = []
    episodes_info = []
    total_frames = 0

    # 在循环外构造一次 renderer，避免每 episode 重复加载 MJCF
    shared_renderer = None
    if render_videos:
        try:
            from src.mujoco_renderer import MujocoRenderer
            shared_renderer = MujocoRenderer(
                mjcf_path=mjcf_path or config.get("mjcf_path"),
                robot=robot,
                image_size=(224, 224),
            )
        except Exception as e:
            print(f"   ⚠️  MuJoCo 渲染器初始化失败，将使用占位视频: {e}")
            shared_renderer = None

    for ep_idx, (start, end) in enumerate(episodes):
        ep_states = states[start:end]
        ep_actions = actions[start:end]
        ep_steps = end - start

        episodes_info.append({
            "episode_index": ep_idx,
            "length": ep_steps,
            "task_index": 0,
        })

        for step in range(ep_steps):
            row = {
                "observation.state": ep_states[step].tolist(),
                "action": ep_actions[step].tolist(),
                "timestamp": step / actual_fps,
                "annotation.human.action.task_description": task_description,
                "task_index": 0,
                "episode_index": ep_idx,
                "index": total_frames + step,
                "next.reward": 0.0,
                "next.done": (step == ep_steps - 1),
            }
            all_rows.append(row)

        total_frames += ep_steps

        if render_videos:
            ep_joint_pos = joint_pos[start:end]
            ep_base_pos = base_pos[start:end]
            ep_base_quat = base_quat[start:end]

            video_path = videos_dir / f"episode_{ep_idx:06d}.mp4"

            if shared_renderer is not None:
                try:
                    shared_renderer.render_motion(
                        joint_pos=ep_joint_pos,
                        output_path=str(video_path),
                        base_pos=ep_base_pos,
                        base_quat=ep_base_quat,
                        camera_name=None,
                        fps=actual_fps,
                    )
                except Exception as e:
                    print(f"   ⚠️  视频渲染失败 (episode {ep_idx}): {e}")
                    _create_placeholder_video(str(video_path), ep_steps, int(actual_fps))
            else:
                _create_placeholder_video(str(video_path), ep_steps, int(actual_fps))

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episodes) - 1:
            print(f"  ✅ Episode {ep_idx + 1}/{len(episodes)}")

    df = pd.DataFrame(all_rows)
    parquet_path = data_dir / "episode_000000.parquet"
    df.to_parquet(str(parquet_path), index=False)
    print(f"  📄 Parquet: {len(df)} rows → {parquet_path}")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot,
        "num_episodes": len(episodes),
        "total_frames": total_frames,
        "task_label": task_description,
        "fps": actual_fps,
        "rejected": False,
        "source": "robot_retargeter",
        "source_file": str(motion_file),
        "task_description": task_description,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print()
    print("📊 转换完成")
    print(f"   Episodes: {len(episodes)}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {actual_fps}")
    print(f"   输出: {output_path}")
    print()
    print("   下一步:")
    print(f"   ./start.sh upload {robot} {output_path}")
    print(f"   ./start.sh train {robot}")

    return info


def _create_placeholder_video(path: str, num_frames: int, fps: int):
    """创建占位视频（纯色帧）。"""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for _ in range(min(num_frames, 10)):  # 最多 10 帧
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [64, 64, 64]  # 灰色
        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="将 robot_retargeter 的运动数据转换为 GR00T LeRobot v2 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=str, help="robot_retargeter qpos CSV 文件路径")
    source.add_argument("--npz", type=str, help="robot_retargeter NPZ 文件路径")

    parser.add_argument("--robot", type=str, default="g1", choices=list(ROBOT_CONFIGS.keys()))
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--episode-length", type=int, default=300, help="每 episode 帧数")
    parser.add_argument("--overlap", type=float, default=0.5, help="episode 重叠比例")
    parser.add_argument("--fps", type=float, default=None, help="帧率")
    parser.add_argument("--task", type=str, default=None, help="任务描述（None 则自动推断）")
    parser.add_argument("--no-video", action="store_true", help="不渲染视频")
    parser.add_argument("--mjcf", type=str, default=None, help="MJCF 模型路径")

    args = parser.parse_args()

    motion_file = args.csv or args.npz

    convert_to_lerobot(
        motion_file=motion_file,
        robot=args.robot,
        output_dir=args.output,
        episode_length=args.episode_length,
        overlap=args.overlap,
        fps=args.fps,
        task_description=args.task,
        render_videos=not args.no_video,
        mjcf_path=args.mjcf,
    )


if __name__ == "__main__":
    main()
