#!/usr/bin/env python3
"""
数据格式转换 — npz → LeRobot v2 格式

读取 collect_data.py 产出的:
  - episode_*.npz         (state + action + reward + 每 episode instruction)
  - episode_*.mp4         (RGB 视频, 可选)
  - episode_*_frames.npz  (frames 序列, 当 imageio 不可用时的 fallback)

转换为 GR00T fine-tune 所需的 LeRobot v2 标准格式:
  output_dir/
  ├── meta/
  │   ├── modality.json   # 模态配置 (state/action/video 维度)
  │   ├── episodes.jsonl  # 每 episode 元数据 (含 task_index)
  │   ├── tasks.jsonl     # 任务描述列表 (去重)
  │   └── info.json       # 数据集元信息
  ├── data/
  │   └── chunk-000/
  │       └── file-000.parquet  # 所有 step 的 state/action 数据
  └── videos/
      └── chunk-000/
          └── episode_*.mp4     # 复制的视频 (GR00T LeRobot v2 期望此结构)

此脚本与仿真引擎完全无关, 只关心 npz 文件中的数据结构。

使用方法:
    # 默认转换 (从 metadata.json 读取 action_mode 等参数)
    python convert_to_lerobot.py --robot g1 \\
        --data-dir /workspace/data/g1_raw \\
        --output-dir /workspace/data/g1_lerobot

    # 显式指定 action_mode
    python convert_to_lerobot.py --robot g1 \\
        --action-mode delta \\
        --data-dir /workspace/data/g1_raw \\
        --output-dir /workspace/data/g1_lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _to_python_list(v: Any) -> list | None:
    """numpy / tensor → python list. None passthrough."""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [(_to_python_list(x) if hasattr(x, "tolist") else x) for x in v]
    return v


def _load_episode_npz(ep_path: Path) -> dict[str, Any]:
    """读 npz, 容错处理 dtype=object 数组."""
    data = np.load(ep_path, allow_pickle=True)
    out: dict[str, Any] = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype == object:
            out[key] = [_to_python_list(x) if not isinstance(x, str) else x for x in arr]
        else:
            out[key] = arr
    return out


def _try_load_video(ep_idx: int, data_path: Path, videos_dst: Path,
                    video_height: int, video_width: int, video_fps: int) -> str | None:
    """查找并复制 episode 的视频到 LeRobot 期望的位置.

    优先级:
      1) episode_NNNNNN.mp4 (imageio 输出)
      2) episode_NNNNNN_frames.npz → 重新编码为 mp4

    Returns:
        相对路径 (用于 parquet 中的 video 字段), 或 None (无视频)
    """
    ep_id = f"{ep_idx:06d}"
    mp4_src = data_path / f"episode_{ep_id}.mp4"
    frames_src = data_path / f"episode_{ep_id}_frames.npz"

    videos_dst.mkdir(parents=True, exist_ok=True)
    mp4_dst = videos_dst / f"episode_{ep_id}.mp4"

    if mp4_src.exists():
        if not mp4_dst.exists() or mp4_dst.stat().st_size != mp4_src.stat().st_size:
            shutil.copy2(mp4_src, mp4_dst)
        return f"videos/chunk-000/episode_{ep_id}.mp4"

    if frames_src.exists():
        # 重新编码 frames → mp4
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                logger.warning(
                    "frames.npz 存在但 imageio 未装, 跳过 episode %s 视频", ep_id
                )
                return None
        try:
            frames = np.load(frames_src)["frames"]
            imageio.mimsave(str(mp4_dst), list(frames), fps=video_fps)
            return f"videos/chunk-000/episode_{ep_id}.mp4"
        except Exception as e:
            logger.warning("frames → mp4 编码失败 (ep %s): %s", ep_id, e)
            return None

    return None


def convert(
    robot: str = "g1",
    data_dir: str = "/workspace/data/g1_raw",
    output_dir: str = "/workspace/data/g1_lerobot",
    action_mode: str | None = None,
    skip_video: bool = False,
) -> str:
    """将收集的 npz (+mp4) 数据转换为 LeRobot v2 格式.

    Args:
        robot: "g1" 或 "go2"
        data_dir: collect_data.py 输出目录
        output_dir: LeRobot v2 输出目录
        action_mode: 强制覆盖 (None = 从 metadata.json 读, 默认 "absolute")
        skip_video: 不复制/不编码视频

    Returns:
        output_dir: 字符串路径
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("需要安装 pandas 和 pyarrow: pip install pandas pyarrow")
        sys.exit(1)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取采集时的元数据
    meta_path = data_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        logger.info("读取 metadata.json: robot=%s, task=%s, action_mode=%s, video=%s",
                    metadata.get("robot"), metadata.get("task_id"),
                    metadata.get("action_mode"), metadata.get("enable_video"))
    else:
        logger.warning("未找到 metadata.json, 使用默认值")
        metadata = {
            "robot": robot.upper(),
            "task_id": "Unitree-G1-Flat",
            "instruction": "walk forward",
            "instruction_pool": ["walk forward"],
            "action_mode": "absolute",
            "enable_video": False,
            "video_fps": 50,
            "video_height": 224,
            "video_width": 224,
            "num_joints": 29 if robot == "g1" else 12,
            "dt": 0.02,
            "state_dim": (29 if robot == "g1" else 12) * 2 + 3 + 4 + 3 + 3,
        }

    # 确定 action_mode
    action_mode = action_mode or metadata.get("action_mode", "absolute")
    if action_mode not in ("absolute", "delta", "relative_eef"):
        logger.warning("未知 action_mode '%s', 回退 absolute", action_mode)
        action_mode = "absolute"

    has_video = (not skip_video) and bool(metadata.get("enable_video"))
    video_fps = int(metadata.get("video_fps", 50))
    video_height = int(metadata.get("video_height", 224))
    video_width = int(metadata.get("video_width", 224))

    # ── 1. 写入 modality.json ─────────────────────────────────────────
    if robot == "g1":
        from configs.g1_config import get_g1_modality_config
        modality = get_g1_modality_config(action_mode=action_mode)
        video_key = "video.front_view"
    else:
        from configs.go2_config import get_go2_modality_config
        modality = get_go2_modality_config(action_mode=action_mode)
        video_key = "video.front_view"

    # 如果源数据无视频, 从 modality 里去掉 video 块
    if not has_video and "video" in modality:
        logger.info("源数据无视频, modality.json 不包含 video 字段")
        del modality["video"]

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)
    logger.info("保存 modality.json (action_mode=%s)", action_mode)

    # ── 2. 找出所有 episode npz ─────────────────────────────────────────
    episodes = sorted(data_path.glob("episode_*.npz"))
    if not episodes:
        logger.error("未找到 episode_*.npz 文件: %s", data_path)
        sys.exit(1)
    logger.info("找到 %d 个 episodes", len(episodes))

    # ── 3. 处理视频 (复制到 videos/chunk-000/ 下, GR00T 期望路径) ───
    videos_dst = output_path / "videos" / "chunk-000"
    video_relpaths: dict[int, str | None] = {}
    if has_video:
        logger.info("复制/编码视频到 %s ...", videos_dst)
        for ep_idx in range(len(episodes)):
            rel = _try_load_video(ep_idx, data_path, videos_dst,
                                  video_height, video_width, video_fps)
            video_relpaths[ep_idx] = rel
        n_videos = sum(1 for v in video_relpaths.values() if v is not None)
        logger.info("视频处理完成: %d / %d 个 episode 有视频", n_videos, len(episodes))

    # ── 4. 转换每 episode 到 parquet rows ─────────────────────────────
    data_records: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}  # instruction → task_index
    tasks_list: list[dict[str, Any]] = []

    for ep_idx, ep_path in enumerate(episodes):
        try:
            data = _load_episode_npz(ep_path)
        except Exception as e:
            logger.warning("跳过损坏的 %s: %s", ep_path.name, e)
            continue

        observations = data.get("observations")
        actions = data.get("actions")
        rewards = data.get("rewards")
        ep_instruction_arr = data.get("instruction")
        # instruction 可能是: 0-d 数组 / 1-d 数组 / Python str / numpy 标量
        # 用 .size (或 len) 都得先把它转成可定长判断的形态
        if ep_instruction_arr is None:
            ep_instruction = None
        elif isinstance(ep_instruction_arr, str):
            ep_instruction = ep_instruction_arr if ep_instruction_arr else None
        else:
            # 尝试用 numpy 通用接口, 避免 0-d 数组触发 len() 报错
            try:
                arr = np.asarray(ep_instruction_arr)
                if arr.size == 0:
                    ep_instruction = None
                else:
                    # 0-d 数组 → 标量; 1-d → 取第一个
                    if arr.ndim == 0:
                        ep_instruction = str(arr.item())
                    else:
                        ep_instruction = str(arr.flatten()[0])
            except Exception:
                ep_instruction = str(ep_instruction_arr)

        if not ep_instruction:
            ep_instruction = metadata.get("instruction", "walk forward")

        # 注册 task
        if ep_instruction not in task_to_index:
            task_to_index[ep_instruction] = len(tasks_list)
            tasks_list.append({"task_index": len(tasks_list), "task": ep_instruction})

        if observations is None or actions is None:
            logger.warning("Episode %d 缺 observations/actions, 跳过", ep_idx)
            continue

        # 统一 array 化
        if isinstance(observations, list):
            n_steps = len(observations)
        else:
            n_steps = len(rewards) if rewards is not None else len(actions)

        for step_idx in range(n_steps):
            # obs 解析
            obs_raw = observations[step_idx]
            if isinstance(obs_raw, dict):
                obs = obs_raw
            elif hasattr(obs_raw, "item"):
                obs = obs_raw.item()
            else:
                obs = {}

            # action 解析
            act_raw = actions[step_idx]
            if isinstance(act_raw, dict):
                act = act_raw
            elif hasattr(act_raw, "item"):
                act = act_raw.item()
            else:
                act = {}

            record: dict[str, Any] = {
                "episode_index": ep_idx,
                "frame_index": step_idx,
                "timestamp": float(step_idx * metadata.get("dt", 0.02)),
                "task_index": task_to_index[ep_instruction],
                "reward": float(rewards[step_idx]) if rewards is not None else 0.0,
                "done": bool(step_idx == n_steps - 1),
                # GR00T 期望的 language annotation key
                "annotation.language.action_text": ep_instruction,
            }

            # ── state.* 字段 ───────────────────────────────────────
            for k, v in obs.items():
                if not isinstance(k, str) or not k.startswith("state."):
                    continue
                if v is None:
                    continue
                if isinstance(v, np.ndarray):
                    record[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    record[k] = list(v)
                else:
                    try:
                        record[k] = float(v)
                    except Exception:
                        record[k] = v

            # ── action.* 字段 (按 action_mode) ────────────────────
            # 注意: 不能用 `or` 串联 act.get(), 否则 numpy 数组会触发
            #       "The truth value of an array with more than one element is ambiguous"
            if action_mode == "absolute":
                v = act.get("action.joint_position_target")
                if v is None:
                    v = act.get("target_joint_pos")
                if v is not None and isinstance(v, np.ndarray):
                    record["action.joint_position_target"] = v.tolist()
            elif action_mode == "delta":
                v = act.get("action.joint_position_delta")
                if v is not None and isinstance(v, np.ndarray):
                    record["action.joint_position_delta"] = v.tolist()
                last = act.get("action.joint_position_last")
                if last is not None and isinstance(last, np.ndarray):
                    record["action.joint_position_last"] = last.tolist()
            elif action_mode == "relative_eef":
                v = act.get("action.ee_pose_delta")
                if v is not None and isinstance(v, np.ndarray):
                    record["action.ee_pose_delta"] = v.tolist()

            # ── 视频路径 (每条 step 写同一个 mp4 路径, GR00T 会抽帧)
            if has_video and video_relpaths.get(ep_idx) is not None:
                record[video_key] = video_relpaths[ep_idx]

            data_records.append(record)

        episode_metadata.append({
            "episode_index": ep_idx,
            "length": n_steps,
            "task": ep_instruction,
            "task_index": task_to_index[ep_instruction],
        })
        if (ep_idx + 1) % 50 == 0:
            logger.info("  转换进度: %d / %d episodes", ep_idx + 1, len(episodes))

    if not data_records:
        logger.error("没有可写入的数据, 请检查 npz 文件")
        sys.exit(1)

    # ── 5. 写 Parquet ─────────────────────────────────────────────────
    df = pd.DataFrame(data_records)
    # 确保数值列是 list 而非 numpy (parquet 兼容性)
    for col in df.columns:
        if col in ("episode_index", "frame_index", "task_index"):
            df[col] = df[col].astype(int)
        elif col == "reward":
            df[col] = df[col].astype(float)
    data_dir_out = output_path / "data" / "chunk-000"
    data_dir_out.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir_out / "file-000.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("保存 Parquet: %s (%d 行, %d 列)",
                parquet_path, len(df), len(df.columns))

    # ── 6. 写 episodes.jsonl ──────────────────────────────────────────
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in episode_metadata:
            f.write(json.dumps(ep) + "\n")
    logger.info("保存 episodes.jsonl (%d episodes, %d unique tasks)",
                len(episode_metadata), len(tasks_list))

    # ── 7. 写 tasks.jsonl ─────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in tasks_list:
            f.write(json.dumps(t) + "\n")
    logger.info("保存 tasks.jsonl (%d tasks):", len(tasks_list))
    for t in tasks_list:
        logger.info("  [%d] %s", t["task_index"], t["task"][:60])

    # ── 8. 写 info.json (GR00T LeRobot 期望) ──────────────────────────
    info = {
        "robot_type": robot.upper(),
        "total_episodes": len(episode_metadata),
        "total_frames": len(data_records),
        "fps": video_fps if has_video else int(round(1.0 / metadata.get("dt", 0.02))),
        "features": {
            **{k: {"dtype": "float32", "shape": v.get("shape", [])}
               for k, v in modality.get("state", {}).items()},
            **{k: {"dtype": "float32", "shape": v.get("shape", [])}
               for k, v in modality.get("action", {}).items()},
            **({video_key: {"dtype": "video", "shape": [video_height, video_width, 3],
                           "fps": video_fps}} if has_video else {}),
            "annotation.language.action_text": {"dtype": "str", "shape": []},
            "task_index": {"dtype": "int64", "shape": []},
        },
        "source_metadata": metadata,  # 完整保存采集时元数据
        "created_at": metadata.get("created_at"),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info("保存 info.json")

    # ── 9. 汇总 ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅ 转换完成!")
    logger.info("  输入:   %s (%d episodes)", data_dir, len(episodes))
    logger.info("  输出:   %s", output_dir)
    logger.info("  Frames: %d", len(data_records))
    logger.info("  Tasks:  %d unique", len(tasks_list))
    logger.info("  Video:  %s", "✅" if has_video else "❌ (no video in source)")
    logger.info("=" * 60)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="npz → LeRobot v2 格式转换 (GR00T fine-tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "go2"],
                        help="机器人类型 (default: g1)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="输入: collect_data.py 输出目录 (含 episode_*.npz)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出: LeRobot v2 格式目录")
    parser.add_argument("--action-mode", type=str, default=None,
                        choices=["absolute", "delta", "relative_eef"],
                        help="动作空间 (None = 从 metadata.json 读)")
    parser.add_argument("--skip-video", action="store_true",
                        help="不复制/不编码视频 (用于纯本体感知训练)")
    args = parser.parse_args()

    convert(
        robot=args.robot,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        action_mode=args.action_mode,
        skip_video=args.skip_video,
    )


if __name__ == "__main__":
    main()
