"""
LeRobot Episode Loader — 独立实现，不依赖 Isaac-GR00T。

读取 LeRobot v2 格式数据集（parquet + mp4），提取图像、状态、动作。
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


class LeRobotEpisodeLoader:
    """LeRobot v2 数据集加载器。"""

    def __init__(
        self,
        dataset_path: str,
        modality_configs: Optional[dict] = None,  # 兼容旧接口，未使用
        image_size: tuple = (224, 224),
    ):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.meta_dir = self.dataset_path / "meta"
        self.data_dir = self.dataset_path / "data"
        self.videos_dir = self.dataset_path / "videos"

        self._load_meta()

    def _load_meta(self):
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                self.info = json.load(f)
        else:
            self.info = {}

        episodes_path = self.meta_dir / "episodes.jsonl"
        self.episodes = []
        if episodes_path.exists():
            with open(episodes_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.episodes.append(json.loads(line))

        tasks_path = self.meta_dir / "tasks.jsonl"
        self.tasks = []
        if tasks_path.exists():
            with open(tasks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        # 兼容 LeRobot v2 标准 "task" 字段与旧版 "task_description"
                        task_text = entry.get("task", entry.get("task_description", ""))
                        self.tasks.append({
                            "task_index": entry.get("task_index", 0),
                            "task": task_text,
                            "task_description": task_text,  # 保留旧字段兼容
                        })

        modality_path = self.meta_dir / "modality.json"
        if modality_path.exists():
            with open(modality_path, encoding="utf-8") as f:
                self.modality = json.load(f)
        else:
            self.modality = {}

        # 从 info.json 读取维度，避免硬编码
        self.state_dim = int(self.info.get("state_dim", 0)) or None
        self.action_dim = int(self.info.get("action_dim", 0)) or None

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> "LeRobotEpisode":
        if idx < 0 or idx >= len(self.episodes):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self.episodes)})")
        return LeRobotEpisode(
            self.data_dir, self.videos_dir, self.episodes[idx],
            self.image_size, self.state_dim, self.action_dim,
            tasks=self.tasks,
        )


class LeRobotEpisode:
    """单个 LeRobot episode。"""

    def __init__(
        self,
        data_dir: Path,
        videos_dir: Path,
        episode_meta: dict,
        image_size: tuple,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        tasks: Optional[list] = None,
    ):
        self.data_dir = data_dir
        self.videos_dir = videos_dir
        self.meta = episode_meta
        self.image_size = image_size
        self.episode_index = episode_meta.get("episode_index", 0)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.tasks = tasks or []

        self._load_data()

    def _load_data(self):
        """加载该 episode 对应的 parquet 行（按 episode_index 过滤）。"""
        # 优先匹配 LeRobot v2 chunk 结构：data/chunk-*/episode_*.parquet
        data_files = sorted(self.data_dir.rglob("episode_*.parquet"))
        if not data_files:
            data_files = sorted(self.data_dir.rglob("data-*.parquet"))
        if not data_files:
            data_files = sorted(self.data_dir.glob("*.parquet"))

        if data_files:
            # 合并所有 parquet，按 episode_index 过滤
            dfs = [pd.read_parquet(p) for p in data_files]
            full_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if "episode_index" in full_df.columns:
                self.df = full_df[full_df["episode_index"] == self.episode_index].reset_index(drop=True)
            else:
                # 无 episode_index 列时退化为取第一个文件
                self.df = pd.read_parquet(data_files[0])
        else:
            self.df = pd.DataFrame()

        self._video_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self):
        return self.df.columns.tolist()

    def _load_video_frame(self, video_key: str, frame_idx: int) -> Optional[np.ndarray]:
        """从 mp4 视频文件读取指定帧。

        支持的目录结构（按优先级）：
          - videos/chunk-000/observation.images.<cam>/episode_000000.mp4  (LeRobot v2 标准)
          - videos/chunk-000/<cam>/episode_000000.mp4
          - videos/chunk-000/<cam>_episode_000000.mp4  (旧扁平命名)
        """
        ep_name = f"episode_{self.episode_index:06d}.mp4"
        candidates = [
            self.videos_dir / f"observation.images.{video_key}" / ep_name,  # LeRobot v2 标准
            self.videos_dir / video_key / ep_name,
            self.videos_dir / f"{video_key}_episode_{self.episode_index:06d}.mp4",
            self.videos_dir / f"episode_{self.episode_index:06d}_{video_key}.mp4",  # 旧扁平带后缀
        ]
        # rglob 兜底
        if not any(p.exists() for p in candidates):
            matches = list(self.videos_dir.rglob(ep_name))
            for m in matches:
                if video_key in str(m) or video_key.replace(".", "_") in str(m):
                    candidates.insert(0, m)
                    break
            if not matches:
                matches = list(self.videos_dir.rglob("episode_*.mp4"))
                if matches:
                    candidates = matches

        for path in candidates:
            if not path.exists():
                continue
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # cv2 读取为 BGR，转 RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frame(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Frame index {idx} out of range")

        row = self.df.iloc[idx]

        # 提取图像：从视频文件按帧读取
        images = {}
        image_cols = [c for c in self.df.columns
                      if "image" in c.lower() or "video" in c.lower()]
        for col in image_cols:
            # 列名形如 observation.images.front
            video_key = col.split(".")[-1] if "." in col else col
            frame_idx = int(row.get("frame_index", idx))
            img = self._load_video_frame(video_key, frame_idx)
            if img is None:
                # parquet 中可能存了占位，或无视频
                placeholder = row[col]
                if isinstance(placeholder, np.ndarray) and placeholder.size > 0:
                    img = placeholder
                else:
                    img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            images[col] = img

        # 提取状态：LeRobot v2 中 state 通常存为单列 observation.state（list）
        state = np.zeros(self._state_dim or 0, dtype=np.float32)
        if "observation.state" in self.df.columns:
            val = row["observation.state"]
            state = np.atleast_1d(np.array(val, dtype=np.float32))
        elif any(c.startswith("state.") for c in self.df.columns):
            state_cols = [c for c in self.df.columns if c.startswith("state.")]
            state = np.concatenate(
                [np.atleast_1d(np.array(row[c], dtype=np.float32)) for c in state_cols]
            ).astype(np.float32)

        # 提取 GT 动作
        gt_action = np.zeros(self._action_dim or 0, dtype=np.float32)
        if "action" in self.df.columns:
            val = row["action"]
            gt_action = np.atleast_1d(np.array(val, dtype=np.float32))
        elif any(c.startswith("action.") for c in self.df.columns):
            action_cols = [c for c in self.df.columns if c.startswith("action.")]
            gt_action = np.concatenate(
                [np.atleast_1d(np.array(row[c], dtype=np.float32)) for c in action_cols]
            ).astype(np.float32)

        return {"images": images, "state": state, "gt_action": gt_action}

    def get_task_description(self, idx: int = 0) -> str:
        """根据 parquet 行的 task_index 取任务描述（指向 tasks.jsonl）。"""
        if not self.tasks:
            return ""
        if idx < 0 or idx >= len(self.df):
            return self.tasks[0].get("task", "")
        row = self.df.iloc[idx]
        task_idx = int(row.get("task_index", 0))
        if 0 <= task_idx < len(self.tasks):
            return self.tasks[task_idx].get("task", "")
        return self.tasks[0].get("task", "")
