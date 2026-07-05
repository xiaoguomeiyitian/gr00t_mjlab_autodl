"""convert_to_lerobot.py — 原始采集数据 → LeRobot v2 格式。"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def convert_to_lerobot(
    input_dir: str,
    output_dir: str,
    robot: str = "g1",
    task_description: str = "perform the locomotion task",
    fps: int = 30,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    meta_file = input_path / "collection_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            collection_meta = json.load(f)
    else:
        collection_meta = {}

    _robot_defaults = {
        "g1":           {"state_dim": 71, "action_dim": 29, "num_joints": 29, "camera_names": ["front", "wrist"]},
        "h1":           {"state_dim": 53, "action_dim": 20, "num_joints": 20, "camera_names": ["front", "wrist"]},
        "h1_with_hand": {"state_dim": 105, "action_dim": 46, "num_joints": 46, "camera_names": ["front", "wrist"]},
        "h1_2":         {"state_dim": 117, "action_dim": 52, "num_joints": 52, "camera_names": ["front", "wrist"]},
        "h2":           {"state_dim": 77, "action_dim": 32, "num_joints": 32, "camera_names": ["front", "wrist"]},
        "go2":          {"state_dim": 37, "action_dim": 12, "num_joints": 12, "camera_names": ["front", "back"]},
    }
    defaults = _robot_defaults.get(robot, _robot_defaults["g1"])
    state_dim = collection_meta.get("state_dim", defaults["state_dim"])
    action_dim = collection_meta.get("action_dim", defaults["action_dim"])
    num_joints = collection_meta.get("num_joints", defaults["num_joints"])
    camera_names = collection_meta.get("camera_names", defaults["camera_names"])

    npz_files = sorted(input_path.glob("episode_*.npz"))
    # mp4 可能是单相机 (episode_000000.mp4) 或多相机 (episode_000000_front.mp4)
    all_mp4_files = sorted(input_path.glob("episode_*.mp4"))

    if not npz_files:
        raise FileNotFoundError(f"未找到 episode_*.npz 文件: {input_path}")

    # 按 episode 索引分组 mp4
    def _episode_idx_from_name(name: str) -> int:
        # episode_000000.mp4 / episode_000000_front.mp4 → 0
        stem = Path(name).stem
        parts = stem.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return -1

    mp4_by_ep = {}
    for mp4 in all_mp4_files:
        ep_idx = _episode_idx_from_name(mp4.name)
        if ep_idx < 0:
            continue
        mp4_by_ep.setdefault(ep_idx, []).append(mp4)

    print(f"📦 转换为 LeRobot v2 格式")
    print(f"   输入: {input_path} ({len(npz_files)} episodes)")
    print(f"   输出: {output_path}")
    print(f"   机器人: {robot}")

    meta_dir = output_path / "meta"
    data_dir = output_path / "data" / "chunk-000"
    videos_dir = output_path / "videos" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    modality = _build_modality_json(robot, num_joints, state_dim, camera_names)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # tasks.jsonl：字段名必须是 "task"（与 LeRobot v2 / GR00T 一致），不是 "task_description"
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    all_rows = []
    episodes_info = []
    video_count = 0
    parquet_idx = 0  # 全局帧索引，跨 episode 连续递增

    for ep_idx, npz_file in enumerate(npz_files):
        data = np.load(str(npz_file))
        states = data["states"]
        actions = data["actions"]
        rewards = data.get("rewards", np.zeros(len(states)))
        ep_steps = len(states)

        # episodes.jsonl：tasks 字段是任务描述列表（与 LeRobot v2 一致）
        episodes_info.append({
            "episode_index": ep_idx,
            "tasks": [task_description],
            "length": ep_steps,
        })

        # 按 episode 索引显式查找 mp4，避免 sorted 索引错配
        ep_mp4s = mp4_by_ep.get(ep_idx, [])
        # 视频目录结构：videos/chunk-000/observation.images.<cam>/episode_000000.mp4
        # 与 LeRobot v2 / GR00T 官方一致（modality.json 的 original_key 指向此路径）
        for cam in camera_names:
            cam_video_dir = videos_dir / f"observation.images.{cam}"
            cam_video_dir.mkdir(parents=True, exist_ok=True)
            video_name = f"episode_{ep_idx:06d}.mp4"
            target = cam_video_dir / video_name
            # 找匹配该相机的 mp4
            cam_mp4 = next((m for m in ep_mp4s if m.stem.endswith(f"_{cam}")), None)
            if cam_mp4 and cam_mp4.exists():
                shutil.copy2(str(cam_mp4), str(target))
                video_count += 1
            elif ep_mp4s and cam == camera_names[0]:
                # 第一个相机回退到无后缀单文件
                single = next((m for m in ep_mp4s if "_" not in m.stem.split("episode_")[1]), None)
                if single and single.exists():
                    shutil.copy2(str(single), str(target))
                    video_count += 1
                else:
                    _create_placeholder_video(str(target), ep_steps, fps, camera_names)
            else:
                _create_placeholder_video(str(target), ep_steps, fps, camera_names)

        # 每个 episode 单独一个 parquet（与 LeRobot v2 一致）
        ep_rows = []
        for step in range(ep_steps):
            row = {
                "observation.state": states[step].tolist(),
                "action": actions[step].tolist(),
                "timestamp": step / fps,
                # annotation 列存 task_index（int），指向 tasks.jsonl，不是字符串
                "annotation.human.action.task_description": 0,
                "task_index": 0,
                "episode_index": ep_idx,
                "index": parquet_idx,
                "frame_index": step,
                "next.reward": float(rewards[step]),
                "next.done": (step == ep_steps - 1),
            }
            ep_rows.append(row)
            parquet_idx += 1

        ep_df = pd.DataFrame(ep_rows)
        ep_parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(str(ep_parquet_path), index=False)

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(npz_files) - 1:
            print(f"  ✅ Episode {ep_idx + 1}/{len(npz_files)}  steps={ep_steps}  → {ep_parquet_path.name}")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")

    total_steps = sum(ep["length"] for ep in episodes_info)
    # info.json：补充 LeRobot v2 标准字段，同时保留本项目维度字段
    info = {
        # LeRobot v2 标准字段
        "codebase_version": "v2.0",
        "robot_type": robot,
        "total_episodes": len(npz_files),
        "total_frames": total_steps,
        "total_tasks": 1,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
            "action": {"dtype": "float32", "shape": [action_dim], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        # 本项目额外维度字段（lerobot_loader 依赖）
        "state_dim": state_dim,
        "action_dim": action_dim,
        "num_joints": num_joints,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n📊 转换完成")
    print(f"   Episodes: {len(npz_files)}")
    print(f"   Total frames: {total_steps}")
    print(f"   Videos: {video_count}")
    print(f"   输出: {output_path}")


def _build_modality_json(robot: str, num_joints: int, state_dim: int, camera_names: list) -> dict:
    slices = {
        "joint_pos": {"start": 0, "end": num_joints},
        "joint_vel": {"start": num_joints, "end": 2 * num_joints},
    }
    offset = 2 * num_joints
    for key in ["base_pos", "base_quat", "base_lin_vel", "base_ang_vel"]:
        dim = {"base_pos": 3, "base_quat": 4, "base_lin_vel": 3, "base_ang_vel": 3}[key]
        slices[key] = {"start": offset, "end": offset + dim}
        offset += dim

    action_slices = {"joint_position_delta": {"start": 0, "end": num_joints}}

    video = {}
    for cam_name in camera_names:
        video[cam_name] = {"original_key": f"observation.images.{cam_name}"}

    return {
        "state": slices,
        "action": action_slices,
        "video": video,
        # annotation 指向 parquet 中的 task_index 列（int 索引到 tasks.jsonl）
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def _create_placeholder_video(path: str, num_frames: int, fps: int, camera_names: list):
    h, w = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        text = f"F{i}"
        cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="将原始数据转换为 LeRobot v2 格式")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"])
    parser.add_argument("--task-description", type=str, default="perform the locomotion task")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    convert_to_lerobot(
        input_dir=args.input_dir, output_dir=args.output_dir,
        robot=args.robot, task_description=args.task_description, fps=args.fps,
    )


if __name__ == "__main__":
    main()
