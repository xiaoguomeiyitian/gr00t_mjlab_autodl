#!/usr/bin/env python3
"""
数据格式转换 — npz → LeRobot v2 格式

读取 collect_data.py 产出的 episode_*.npz，转换为 GR00T fine-tune 所需的
LeRobot v2 标准格式 (parquet + modality.json + episodes.jsonl)。

此脚本与仿真引擎完全无关，只关心 npz 文件中的数据结构。

使用方法:
    python convert_to_lerobot.py --robot g1 \
        --data-dir /workspace/data/g1_raw \
        --output-dir /workspace/data/g1_lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def convert(
    robot: str = "g1",
    data_dir: str = "/workspace/data/g1_raw",
    output_dir: str = "/workspace/data/g1_lerobot",
) -> str:
    """将收集的 npz 数据转换为 LeRobot v2 格式。

    LeRobot v2 目录结构:
        output_dir/
        ├── meta/
        │   ├── modality.json      # 模态配置
        │   ├── episodes.jsonl     # episode 元数据
        │   └── tasks.jsonl        # 任务描述
        ├── data/
        │   └── chunk-000/
        │       └── file-000.parquet  # 所有 episode 数据
        └── videos/                # (可选) 视频
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("需要安装 pandas 和 pyarrow: pip install pandas pyarrow")
        sys.exit(1)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取元数据
    meta_path = data_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        logger.warning("未找到 metadata.json, 使用默认值")
        metadata = {"instruction": "walk forward", "robot": robot.upper()}

    instruction = metadata.get("instruction", "walk forward")

    # ── 1. 保存 modality.json ────────────────────────────────────────────
    if robot == "g1":
        from configs.g1_config import get_g1_modality_config
        modality = get_g1_modality_config()
    else:
        from configs.go2_config import get_go2_modality_config
        modality = get_go2_modality_config()

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)
    logger.info("保存 modality.json")

    # ── 2. 转换每个 episode ──────────────────────────────────────────────
    episodes = sorted(data_path.glob("episode_*.npz"))
    if not episodes:
        logger.error("未找到 episode_*.npz 文件: %s", data_dir)
        sys.exit(1)

    logger.info("转换 %d 个 episodes...", len(episodes))

    data_records: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []

    for ep_idx, ep_path in enumerate(episodes):
        data = np.load(ep_path, allow_pickle=True)
        observations = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]

        for step_idx in range(len(rewards)):
            # 解析 observation
            obs_raw = observations[step_idx]
            if isinstance(obs_raw, dict):
                obs = obs_raw
            elif hasattr(obs_raw, "item"):
                obs = obs_raw.item()
            else:
                obs = {}

            # 解析 action
            act_raw = actions[step_idx]
            if isinstance(act_raw, dict):
                act = act_raw
            elif hasattr(act_raw, "item"):
                act = act_raw.item()
            else:
                act = {"target_joint_pos": np.zeros(metadata.get("action_dim", 12))}

            record: dict[str, Any] = {
                "episode_index": ep_idx,
                "frame_index": step_idx,
                "timestamp": step_idx * metadata.get("dt", 0.02),
                "reward": float(rewards[step_idx]),
                "done": step_idx == len(rewards) - 1,
            }

            # 添加状态
            for k, v in obs.items():
                if k.startswith("state."):
                    if isinstance(v, np.ndarray):
                        record[k] = v.tolist()
                    else:
                        record[k] = v

            # 添加动作
            if isinstance(act, dict) and "target_joint_pos" in act:
                action_val = act["target_joint_pos"]
                if isinstance(action_val, np.ndarray):
                    record["action.joint_position_target"] = action_val.tolist()
                else:
                    record["action.joint_position_target"] = action_val

            data_records.append(record)

        episode_metadata.append({
            "episode_index": ep_idx,
            "length": len(rewards),
            "task": instruction,
        })

    # ── 3. 保存 Parquet ──────────────────────────────────────────────────
    df = pd.DataFrame(data_records)
    data_dir_path = output_path / "data" / "chunk-000"
    data_dir_path.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir_path / "file-000.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("保存 Parquet: %s (%d 行)", parquet_path, len(df))

    # ── 4. 保存 episodes.jsonl ───────────────────────────────────────────
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in episode_metadata:
            f.write(json.dumps(ep) + "\n")
    logger.info("保存 episodes.jsonl: %d episodes", len(episode_metadata))

    # ── 5. 保存 tasks.jsonl ──────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    logger.info("保存 tasks.jsonl")

    logger.info("=" * 60)
    logger.info("✅ 转换完成!")
    logger.info("  输入: %s (%d episodes)", data_dir, len(episodes))
    logger.info("  输出: %s", output_dir)
    logger.info("  数据行数: %d", len(df))
    logger.info("=" * 60)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="npz → LeRobot v2 格式转换")
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "go2"])
    parser.add_argument("--data-dir", type=str, required=True,
                        help="输入: 收集的 npz 数据目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出: LeRobot v2 格式目录")
    args = parser.parse_args()

    convert(robot=args.robot, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
