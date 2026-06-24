#!/usr/bin/env python3
"""数据格式转换 — npz → LeRobot v2 (Isaac-GR00T 兼容)

读取 collect_data.py 产出的:
  - episode_*.npz         (state + action + reward + 每 episode instruction)
  - episode_*.mp4         (RGB 视频, 可选)
  - episode_*_frames.npz  (frames 序列, 当 imageio 不可用时的 fallback)

转换为 GR00T fine-tune 所需的 LeRobot v2 标准格式 (与 Isaac-GR00T
demo_data/cube_to_bowl_5 完全一致):

  output_dir/
  ├── meta/
  │   ├── info.json              # LeRobot v2 dataset metadata
  │   ├── modality.json          # GR00T start/end slice schema
  │   ├── episodes.jsonl         # 每 episode: index, tasks, length
  │   ├── tasks.jsonl            # task_index → task string
  │   └── stats.json             # 由 gr00t/data/stats.py 生成 (本脚本不写)
  ├── data/
  │   └── chunk-000/
  │       └── file-000.parquet   # 所有 step 的数据 (state/action 拼接为单列)
  └── videos/
      └── chunk-000/
          └── observation.images.front_view/
              └── episode_NNNNNN.mp4

parquet 列结构 (关键):
  - observation.state : float32[71]  (G1) / float32[37]  (Go2)   ← 拼接
  - action            : float32[29]  (G1) / float32[12]  (Go2)   ← 拼接
  - task_index        : int64[1]     (索引 tasks.jsonl)
  - frame_index       : int64[1]
  - episode_index     : int64[1]
  - index             : int64[1]     (全局 step 序号)
  - timestamp         : float32[1]
  - observation.images.front_view : str  (相对路径, GR00T 内部抽帧)

注意: state/action 都是**拼接的单列**, 通过 modality.json 的 start/end 切片
被 GR00T 数据加载器分解为各子字段。

使用方法:
    # 默认 (从 metadata.json 读 action_mode)
    python convert_to_lerobot.py --robot g1 \\
        --data-dir data/g1_raw \\
        --output-dir data/g1_lerobot

    # 显式指定 action_mode
    python convert_to_lerobot.py --robot g1 \\
        --action-mode delta \\
        --data-dir data/g1_raw \\
        --output-dir data/g1_lerobot

转换完成后, 必须在云端或本地跑一次 stats 生成:
    python gr00t/data/stats.py \\
        --dataset-path data/g1_lerobot \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path <path-to>/g1_new_embodiment_config.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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


# ──────────────────────────────────────────────────────────────────────────────
# 常量: 与 GR00T demo data 一致
# ──────────────────────────────────────────────────────────────────────────────

# LeRobot data path 模板 (与 demo_data/cube_to_bowl_5/info.json 一致)
# 关键: loader 用 episode_index 读取 parquet, 不能用 file_index
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
# GR00T 期望的视频路径: {video_key} 会被 modality.json video.<key>.original_key 替换
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000
LEROBOT_CODEBASE_VERSION = "v2.1"


# state 拼接顺序: 与 configs/g1_config.py _build_state_layout 一致
STATE_KEYS_ORDER = [
    "joint_pos",
    "joint_vel",
    "base_pos",
    "base_quat",
    "base_lin_vel",
    "base_ang_vel",
]


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────


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


def _get_state_vector(obs: dict, num_joints: int) -> np.ndarray:
    """从单步 obs dict 提取并按 STATE_KEYS_ORDER 拼接 state 向量.

    缺失的字段用零填充 (G1: 71 维, Go2: 37 维, 计算自 g1_config._build_state_layout).
    """
    layout_dims = {
        "joint_pos":    num_joints,
        "joint_vel":    num_joints,
        "base_pos":     3,
        "base_quat":    4,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
    }
    parts = []
    for key in STATE_KEYS_ORDER:
        v = obs.get(f"state.{key}")
        if v is None:
            v = np.zeros(layout_dims[key], dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.shape[0] != layout_dims[key]:
                logger.warning("state.%s 维度不对: got %d, expected %d, 用 0 填充",
                               key, v.shape[0], layout_dims[key])
                v = np.zeros(layout_dims[key], dtype=np.float32)
        parts.append(v)
    return np.concatenate(parts).astype(np.float32)


def _get_action_vector(act: dict, action_mode: str, num_joints: int) -> np.ndarray:
    """从单步 act dict 提取 action 向量 (按 action_mode).

    与 modality.json action 块中 start/end 的拼接顺序一致 (参见
    configs/g1_config.py: _build_action_layout).
    """
    if action_mode == "absolute":
        # action: { "joint_position_target": (num_joints,) }
        v = act.get("action.joint_position_target")
        if v is None:
            v = act.get("target_joint_pos")
        if v is None:
            v = np.zeros(num_joints, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] != num_joints:
            v = np.zeros(num_joints, dtype=np.float32)
        return v.astype(np.float32)

    elif action_mode == "delta":
        # action: { "joint_position_delta": (num_joints,) }
        v = act.get("action.joint_position_delta")
        if v is None:
            v = np.zeros(num_joints, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] != num_joints:
            v = np.zeros(num_joints, dtype=np.float32)
        return v.astype(np.float32)

    elif action_mode == "relative_eef":
        # action: { "ee_pose_delta": (7,) }  pos(3) + quat(4)
        v = act.get("action.ee_pose_delta")
        if v is None:
            v = np.zeros(7, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] != 7:
            v = np.zeros(7, dtype=np.float32)
        return v.astype(np.float32)

    else:
        raise ValueError(f"未知 action_mode: {action_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# 视频处理
# ──────────────────────────────────────────────────────────────────────────────


def _try_load_video(
    ep_idx: int,
    data_path: Path,
    videos_chunk_dir: Path,
    video_height: int,
    video_width: int,
    video_fps: int,
    video_subdir: str,
) -> str | None:
    """复制/编码 episode 的视频到 LeRobot 期望的 videos/chunk-XXX/<original_key>/ 下.

    优先级:
      1) episode_NNNNNN.mp4 (imageio 输出) — 直接复制
      2) episode_NNNNNN_frames.npz → 重新编码为 mp4

    Returns:
        相对路径 (写入 parquet 前会被替换为 observation.images.front_view/<file>),
        或 None (无视频).
    """
    ep_id = f"{ep_idx:06d}"
    mp4_src = data_path / f"episode_{ep_id}.mp4"
    frames_src = data_path / f"episode_{ep_id}_frames.npz"

    # 视频子目录名 = modality.json video.front_view.original_key 的值
    video_dst_dir = videos_chunk_dir / video_subdir
    video_dst_dir.mkdir(parents=True, exist_ok=True)
    mp4_dst = video_dst_dir / f"episode_{ep_id}.mp4"

    if mp4_src.exists():
        if not mp4_dst.exists() or mp4_dst.stat().st_size != mp4_src.stat().st_size:
            shutil.copy2(mp4_src, mp4_dst)
        return f"videos/chunk-000/{video_subdir}/episode_{ep_id}.mp4"

    if frames_src.exists():
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
            return f"videos/chunk-000/{video_subdir}/episode_{ep_id}.mp4"
        except Exception as e:
            logger.warning("frames → mp4 编码失败 (ep %s): %s", ep_id, e)
            return None

    return None


# ──────────────────────────────────────────────────────────────────────────────
# 主转换函数
# ──────────────────────────────────────────────────────────────────────────────


def convert(
    robot: str = "g1",
    data_dir: str = "/workspace/data/g1_raw",
    output_dir: str = "/workspace/data/g1_lerobot",
    action_mode: str | None = None,
    skip_video: bool = False,
) -> str:
    """将 collect_data.py 输出的 npz (+mp4) 数据转换为 LeRobot v2 格式.

    Args:
        robot: "g1" 或 "go2"
        data_dir: collect_data.py 输出目录
        output_dir: LeRobot v2 输出目录
        action_mode: 强制覆盖 (None = 从 metadata.json 读, 默认 "delta")
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
            "task_id": "Mjlab-Velocity-Flat-Unitree-G1" if robot == "g1" else "Mjlab-Velocity-Flat-Unitree-Go2",
            "instruction": "walk forward",
            "instruction_pool": ["walk forward"],
            "action_mode": "delta",
            "enable_video": False,
            "video_fps": 30,
            "video_height": 224,
            "video_width": 224,
            "num_joints": 29 if robot == "g1" else 12,
            "dt": 0.02,
        }

    # 确定 action_mode / robot 元数据
    action_mode = action_mode or metadata.get("action_mode", "delta")  # ← 修复: 默认 delta (与 01 脚本对齐)
    if action_mode not in ("absolute", "delta", "relative_eef"):
        logger.warning("未知 action_mode '%s', 回退 delta", action_mode)
        action_mode = "delta"

    num_joints = 29 if robot == "g1" else 12
    has_video = (not skip_video) and bool(metadata.get("enable_video"))
    video_fps = int(metadata.get("video_fps", 30))
    video_height = int(metadata.get("video_height", 224))
    video_width = int(metadata.get("video_width", 224))
    # 与 g1_config.G1_VIDEO_KEY 保持一致
    video_subdir = "observation.images.front_view"

    # 维度计算
    if robot == "g1":
        from configs.g1_config import get_g1_state_dim, get_g1_action_dim
        state_dim = get_g1_state_dim()
        action_dim = get_g1_action_dim(action_mode)
    else:
        from configs.go2_config import get_go2_state_dim, get_go2_action_dim
        state_dim = get_go2_state_dim()
        action_dim = get_go2_action_dim(action_mode)

    logger.info("=" * 60)
    logger.info("LeRobot v2 转换 (GR00T 兼容)")
    logger.info("  机器人:        %s (%d joints)", robot, num_joints)
    logger.info("  动作空间:      %s", action_mode)
    logger.info("  state_dim:     %d (拼接后)", state_dim)
    logger.info("  action_dim:    %d (拼接后)", action_dim)
    logger.info("  视频:          %s", "✅" if has_video else "❌")
    logger.info("=" * 60)

    # ── 1. 写入 modality.json (GR00T start/end schema) ─────────────────
    if robot == "g1":
        from configs.g1_config import get_g1_modality_config
        modality = get_g1_modality_config(
            action_mode=action_mode, include_video=has_video,
        )
    else:
        from configs.go2_config import get_go2_modality_config
        modality = get_go2_modality_config(
            action_mode=action_mode, include_video=has_video,
        )

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)
    logger.info("✅ 写入 meta/modality.json")

    # ── 1.5 一致性校验: modality.action 关键 key 必须与 action_mode 语义匹配 ──
    # 防止采集端用 --action-mode delta, 但误用 absolute config (或反之)
    expected_action_keys = {
        "absolute":     ["joint_position_target"],
        "delta":        ["joint_position_delta"],
        "relative_eef": ["ee_pose_delta"],
    }
    actual_action_keys = list(modality.get("action", {}).keys())
    expected = expected_action_keys.get(action_mode, [])
    if expected and not any(k in actual_action_keys for k in expected):
        # ← 修复: 这是软警告, 不阻断流程, 但训练会失败. 给用户明确提示
        logger.warning(
            "⚠️  modality.action 关键 key 与 action_mode 不一致! "
            "action_mode=%s 期望含 %s, 实际含 %s. "
            "训练时 LeRobot loader 会找不到对应 joint group, 请检查 ModalityConfig.",
            action_mode, expected, actual_action_keys,
        )

    # ── 2. 找出所有 episode npz ─────────────────────────────────────────
    episodes = sorted(data_path.glob("episode_*.npz"))
    if not episodes:
        logger.error("未找到 episode_*.npz 文件: %s", data_path)
        sys.exit(1)
    logger.info("找到 %d 个 episodes", len(episodes))

    # ── 3. 处理视频 ────────────────────────────────────────────────────
    videos_chunk_dir = output_path / "videos" / "chunk-000"
    video_relpaths: dict[int, str | None] = {}
    if has_video:
        logger.info("复制/编码视频到 %s ...", videos_chunk_dir)
        for ep_idx in range(len(episodes)):
            rel = _try_load_video(
                ep_idx, data_path, videos_chunk_dir,
                video_height, video_width, video_fps, video_subdir,
            )
            video_relpaths[ep_idx] = rel
        n_videos = sum(1 for v in video_relpaths.values() if v is not None)
        logger.info("视频处理完成: %d / %d 个 episode 有视频", n_videos, len(episodes))
        if n_videos == 0:
            logger.warning("所有 episode 都没有视频, 但 metadata 声明 enable_video=true. "
                           "回退到无视频模式.")
            has_video = False
            # 重写 modality.json (去掉 video 块)
            if robot == "g1":
                from configs.g1_config import get_g1_modality_config
                modality = get_g1_modality_config(
                    action_mode=action_mode, include_video=False,
                )
            else:
                from configs.go2_config import get_go2_modality_config
                modality = get_go2_modality_config(
                    action_mode=action_mode, include_video=False,
                )
            with open(meta_dir / "modality.json", "w") as f:
                json.dump(modality, f, indent=2, ensure_ascii=False)

    # ── 4. 转换每 episode 到 parquet rows ─────────────────────────────
    data_records: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    tasks_list: list[dict[str, Any]] = []
    global_index = 0  # 全局 step 序号 (对应 parquet 'index' 列)

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

        # 解 instruction
        if ep_instruction_arr is None:
            ep_instruction = None
        elif isinstance(ep_instruction_arr, str):
            ep_instruction = ep_instruction_arr if ep_instruction_arr else None
        else:
            try:
                arr = np.asarray(ep_instruction_arr)
                if arr.size == 0:
                    ep_instruction = None
                elif arr.ndim == 0:
                    ep_instruction = str(arr.item())
                else:
                    ep_instruction = str(arr.flatten()[0])
            except Exception:
                ep_instruction = str(ep_instruction_arr)

        if not ep_instruction:
            ep_instruction = metadata.get("instruction", "walk forward")

        if ep_instruction not in task_to_index:
            task_to_index[ep_instruction] = len(tasks_list)
            tasks_list.append({"task_index": len(tasks_list), "task": ep_instruction})

        if observations is None or actions is None:
            logger.warning("Episode %d 缺 observations/actions, 跳过", ep_idx)
            continue

        # 修复: n_steps 优先级应为 observations > actions > rewards
        #   - observations 是关键 (决定 parquet 写入多少 frame)
        #   - actions 是次优 (理论上应与 observations 同长)
        #   - rewards 仅辅助 (旧代码用 rewards 优先, 错)
        if isinstance(observations, list):
            n_steps = len(observations)
        elif isinstance(actions, list):
            n_steps = len(actions)
        elif rewards is not None:
            n_steps = len(rewards)
        else:
            n_steps = 0

        for step_idx in range(n_steps):
            obs_raw = observations[step_idx]
            if isinstance(obs_raw, dict):
                obs = obs_raw
            elif hasattr(obs_raw, "item"):
                obs = obs_raw.item()
            else:
                obs = {}

            act_raw = actions[step_idx]
            if isinstance(act_raw, dict):
                act = act_raw
            elif hasattr(act_raw, "item"):
                act = act_raw.item()
            else:
                act = {}

            # ── 拼接 state / action 为 1D 数组 ───────────────────
            state_vec = _get_state_vector(obs, num_joints)
            action_vec = _get_action_vector(act, action_mode, num_joints)

            record: dict[str, Any] = {
                "observation.state": state_vec,                   # float32[state_dim]
                "action":            action_vec,                  # float32[action_dim]
                "task_index":    task_to_index[ep_instruction],
                "frame_index":   step_idx,
                "episode_index": ep_idx,
                "index":         global_index,
                "timestamp":     float(step_idx * metadata.get("dt", 0.02)),
            }

            # 视频相对路径 (per-row 都写同一个, GR00T 内部抽帧)
            if has_video and video_relpaths.get(ep_idx) is not None:
                record[video_subdir] = video_relpaths[ep_idx]

            data_records.append(record)
            global_index += 1

        episode_metadata.append({
            "episode_index": ep_idx,
            "tasks": [ep_instruction],   # 注意: GR00T 用 list (per LeRobot v2 spec)
            "length": n_steps,
        })
        if (ep_idx + 1) % 50 == 0:
            logger.info("  转换进度: %d / %d episodes", ep_idx + 1, len(episodes))

    if not data_records:
        logger.error("没有可写入的数据, 请检查 npz 文件")
        sys.exit(1)

    # ── 5. 写 Parquet ─────────────────────────────────────────────────
    df = pd.DataFrame(data_records)
    # 把 numpy 数组列显式转 list (parquet 兼容)
    for col in ("observation.state", "action"):
        if col in df.columns:
            df[col] = df[col].apply(lambda a: np.asarray(a, dtype=np.float32).tolist())
    for col in ("task_index", "frame_index", "episode_index", "index"):
        if col in df.columns:
            df[col] = df[col].astype("int64")
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("float32")
    if video_subdir in df.columns:
        df[video_subdir] = df[video_subdir].astype("object")

    # ── 每个 episode 一个 parquet, 匹配官方 LeRobot v2 格式 ────────────────
    # GR00T 的 lerobot_episode_loader 通过 data_path 模板按 episode_index 读取:
    #   parquet_filename = data_path_pattern.format(episode_chunk=..., episode_index=...)
    # 因此 data/chunk-XXX/episode_NNNNNN.parquet 是必须的命名格式。
    n_episodes_written = 0
    for ep_idx in range(len(episodes)):
        ep_df = df[df["episode_index"] == ep_idx].copy()
        if ep_df.empty:
            continue
        chunk_idx = ep_idx // CHUNKS_SIZE
        chunk_dir = output_path / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_parquet_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(ep_parquet_path, index=False)
        n_episodes_written += 1
    logger.info("✅ 写入 %d 个 episode parquet (chunk-000/episode_NNNNNN.parquet)",
                n_episodes_written)

    # ── 6. 写 episodes.jsonl (LeRobot v2: tasks=list, length=int) ────
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in episode_metadata:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    logger.info("✅ 写入 meta/episodes.jsonl (%d episodes)", len(episode_metadata))

    # ── 7. 写 tasks.jsonl ─────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in tasks_list:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("✅ 写入 meta/tasks.jsonl (%d unique tasks)", len(tasks_list))
    for t in tasks_list:
        logger.info("  [%d] %s", t["task_index"], t["task"][:60])

    # ── 8. 写 info.json (LeRobot v2 + GR00T) ──────────────────────────
    # state/action 都是拼接单列; 维度从 modality.json 推算
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action":            {"dtype": "float32", "shape": [action_dim]},
        "task_index":        {"dtype": "int64",   "shape": [1], "names": None},
        "frame_index":       {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index":     {"dtype": "int64",   "shape": [1], "names": None},
        "index":             {"dtype": "int64",   "shape": [1], "names": None},
        "timestamp":         {"dtype": "float32", "shape": [1], "names": None},
    }
    if has_video:
        features[video_subdir] = {
            "dtype": "video",
            "shape": [video_height, video_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height":      video_height,
                "video.width":       video_width,
                "video.codec":       "h264",
                "video.pix_fmt":     "yuv420p",
                "video.is_depth_map": False,
                "video.fps":         video_fps,
                "video.channels":    3,
                "has_audio":         False,
            },
        }

    fps_value = video_fps if has_video else int(round(1.0 / metadata.get("dt", 0.02)))

    # 修复: total_chunks 应根据实际 chunk 数计算 (CHUNKS_SIZE=1000, 大数据集可能多 chunk)
    total_chunks = max(
        1, math.ceil(len(episode_metadata) / CHUNKS_SIZE)
    )

    info: dict[str, Any] = {
        "codebase_version":   LEROBOT_CODEBASE_VERSION,
        "robot_type":         robot.upper(),
        "total_episodes":     len(episode_metadata),
        "total_frames":       len(data_records),
        "total_tasks":        len(tasks_list),
        "total_chunks":       total_chunks,
        "total_videos":       len(episodes) if has_video else 0,
        "chunks_size":        CHUNKS_SIZE,
        "fps":                fps_value,
        "splits":             {"train": f"0:{len(episode_metadata)}"},
        "data_path":          DATA_PATH_TEMPLATE,
        "video_path":         VIDEO_PATH_TEMPLATE if has_video else None,
        "features":           features,
        "source_metadata":    metadata,  # 保留采集时的完整 metadata
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info("✅ 写入 meta/info.json (codebase_version=%s, fps=%d, features=%d 项)",
                LEROBOT_CODEBASE_VERSION, fps_value, len(features))

    # ── 9. 汇总 ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅ 转换完成!")
    logger.info("  输入:   %s (%d episodes)", data_dir, len(episodes))
    logger.info("  输出:   %s", output_dir)
    logger.info("  Frames: %d (total)", len(data_records))
    logger.info("  Tasks:  %d unique", len(tasks_list))
    logger.info("  Video:  %s (%d eps)",
                "✅" if has_video else "❌",
                len(episodes) if has_video else 0)
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  重要: 训练前必须生成 stats.json + relative_stats.json:")
    logger.info("  在云端 (有 Isaac-GR00T 仓库) 跑:")
    logger.info("    source /root/Isaac-GR00T/.venv/bin/activate")
    logger.info("    cd /root/Isaac-GR00T")
    if robot == "g1":
        cfg = "g1_new_embodiment_config" if action_mode == "delta" else "g1_new_embodiment_config_absolute"
    else:
        cfg = "go2_new_embodiment_config" if action_mode == "delta" else "go2_new_embodiment_config_absolute"
    logger.info("    python3 gr00t/data/stats.py \\")
    logger.info("        --dataset-path %s \\", output_path)
    logger.info("        --embodiment-tag NEW_EMBODIMENT \\")
    logger.info("        --modality-config-path /root/gr00t_mjlab_autodl/src/configs/%s.py", cfg)
    logger.info("")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="npz → LeRobot v2 格式转换 (Isaac-GR00T fine-tune)",
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
