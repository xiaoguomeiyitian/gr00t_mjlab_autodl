"""
LeRobot Episode Loader — 独立实现，不依赖 Isaac-GR00T。

读取 LeRobot v2 格式数据集（parquet + mp4），提取图像、状态、动作。
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class LeRobotEpisodeLoader:
    """LeRobot v2 数据集加载器。"""

    def __init__(
        self,
        dataset_path: str,
        modality_configs: dict = None,
        image_size: tuple = (224, 224),
    ):
        """
        Args:
            dataset_path: 数据集根目录（含 meta/ 和 data/）
            modality_configs: 模态配置（可选，自动推断）
            image_size: 图像尺寸 (H, W)
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.meta_dir = self.dataset_path / "meta"
        self.data_dir = self.dataset_path / "data"
        self.videos_dir = self.dataset_path / "videos"

        # 读取元数据
        self._load_meta()

    def _load_meta(self):
        """加载 LeRobot 元数据"""
        # info.json
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                self.info = json.load(f)
        else:
            self.info = {}

        # episodes.jsonl
        episodes_path = self.meta_dir / "episodes.jsonl"
        self.episodes = []
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.episodes.append(json.loads(line))

        # tasks.jsonl
        tasks_path = self.meta_dir / "tasks.jsonl"
        self.tasks = []
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.tasks.append(json.loads(line))

        # modality.json
        modality_path = self.meta_dir / "modality.json"
        if modality_path.exists():
            with open(modality_path) as f:
                self.modality = json.load(f)
        else:
            self.modality = {}

    def __len__(self) -> int:
        """返回 episode 数量"""
        return len(self.episodes)

    def __getitem__(self, idx: int) -> "LeRobotEpisode":
        """获取指定 episode"""
        if idx < 0 or idx >= len(self.episodes):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self.episodes)})")
        return LeRobotEpisode(self.data_dir, self.videos_dir, self.episodes[idx], self.image_size)


class LeRobotEpisode:
    """单个 LeRobot episode。"""

    def __init__(self, data_dir: Path, videos_dir: Path, episode_meta: dict, image_size: tuple):
        self.data_dir = data_dir
        self.videos_dir = videos_dir
        self.meta = episode_meta
        self.image_size = image_size
        self.episode_index = episode_meta.get("episode_index", 0)

        # 加载 parquet 数据
        self._load_data()

    def _load_data(self):
        """加载 parquet 文件"""
        # 找到对应的 chunk 文件
        data_files = sorted(self.data_dir.glob("chunk-*/data-*.parquet"))
        if not data_files:
            # 尝试其他命名
            data_files = sorted(self.data_dir.glob("*.parquet"))

        if data_files:
            # 读取第一个（简化：假设单个 chunk）
            self.df = pd.read_parquet(data_files[0])
        else:
            self.df = pd.DataFrame()

        # 视频缓存
        self._video_cache = {}

    def __len__(self) -> int:
        """返回帧数"""
        return len(self.df)

    @property
    def columns(self):
        """返回 DataFrame 列名"""
        return self.df.columns.tolist()

    def get_frame(self, idx: int) -> dict:
        """
        获取单帧数据。

        Returns:
            {
                "images": {"camera_key": np.ndarray},
                "state": np.ndarray,
                "gt_action": np.ndarray,
            }
        """
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Frame index {idx} out of range")

        row = self.df.iloc[idx]

        # 提取图像
        images = {}
        for col in self.df.columns:
            if "image" in col.lower() or "video" in col.lower():
                img = row[col]
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                if img.shape[:2] != self.image_size:
                    img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                images[col] = img

        # 提取状态
        state_cols = [c for c in self.df.columns if c.startswith("state.")]
        if state_cols:
            state = np.vstack([row[c] for c in state_cols]).flatten().astype(np.float32)
        else:
            state = np.zeros(17, dtype=np.float32)

        # 提取 GT 动作
        action_cols = [c for c in self.df.columns if c.startswith("action.")]
        if action_cols:
            gt_action = np.vstack([row[c] for c in action_cols]).flatten().astype(np.float32)
        else:
            gt_action = np.zeros(17, dtype=np.float32)

        return {"images": images, "state": state, "gt_action": gt_action}
