"""测试 LeRobotEpisodeLoader — LeRobot v2 数据集加载器。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.lerobot_loader import LeRobotEpisodeLoader


def _make_minimal_lerobot_dataset(root, num_frames=5, state_dim=71, action_dim=29):
    """构造一个最小可用的 LeRobot v2 数据集目录结构（不含视频文件）。"""
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump({"state_dim": state_dim, "action_dim": action_dim}, f)

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_index": 0, "length": num_frames}) + "\n")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task_description": "walk"}) + "\n")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_frames):
        rows.append({
            "observation.state": [float(x) for x in np.random.randn(state_dim)],
            "action": [float(x) for x in np.random.randn(action_dim)],
            "episode_index": 0,
            "frame_index": i,
            "timestamp": float(i) / 30.0,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / "episode_000000.parquet", index=False)

    return root


class TestLeRobotEpisodeLoader:
    """LeRobotEpisodeLoader 测试。"""

    @pytest.fixture
    def dataset(self, temp_dir):
        _make_minimal_lerobot_dataset(temp_dir)
        return temp_dir

    def test_init_and_len(self, dataset):
        loader = LeRobotEpisodeLoader(str(dataset))
        assert len(loader) == 1
        assert loader.state_dim == 71
        assert loader.action_dim == 29

    def test_getitem_returns_episode(self, dataset):
        loader = LeRobotEpisodeLoader(str(dataset))
        episode = loader[0]
        assert episode is not None
        assert len(episode) == 5
        assert episode.episode_index == 0

    def test_get_frame_shapes(self, dataset):
        loader = LeRobotEpisodeLoader(str(dataset))
        episode = loader[0]
        frame = episode.get_frame(0)
        assert isinstance(frame, dict)
        assert "images" in frame
        assert "state" in frame
        assert "gt_action" in frame
        assert frame["state"].shape == (71,)
        assert frame["gt_action"].shape == (29,)
        assert frame["state"].dtype == np.float32
        assert frame["gt_action"].dtype == np.float32

    def test_get_frame_index_out_of_range(self, dataset):
        loader = LeRobotEpisodeLoader(str(dataset))
        episode = loader[0]
        with pytest.raises(IndexError):
            episode.get_frame(5)
        with pytest.raises(IndexError):
            episode.get_frame(-1)

    def test_episode_index_out_of_range(self, dataset):
        loader = LeRobotEpisodeLoader(str(dataset))
        with pytest.raises(IndexError):
            _ = loader[1]
        with pytest.raises(IndexError):
            _ = loader[-1]

    def test_no_video_placeholder_images(self, dataset):
        """无视频文件时，images 应返回占位（zeros）而非报错。"""
        loader = LeRobotEpisodeLoader(str(dataset))
        episode = loader[0]
        frame = episode.get_frame(0)
        # 数据集中没有 image/video 列，images 应为空 dict
        assert isinstance(frame["images"], dict)

    def test_multi_episode_filtering(self, temp_dir):
        """多 episode 时按 episode_index 过滤。"""
        _make_minimal_lerobot_dataset(temp_dir, num_frames=5)
        # 追加第二个 episode 的行
        data_dir = temp_dir / "data" / "chunk-000"
        df = pd.read_parquet(data_dir / "episode_000000.parquet")
        df2 = pd.DataFrame([
            {
                "observation.state": [0.0] * 71,
                "action": [0.0] * 29,
                "episode_index": 1,
                "frame_index": i,
                "timestamp": float(i) / 30.0,
            }
            for i in range(3)
        ])
        df_all = pd.concat([df, df2], ignore_index=True)
        df_all.to_parquet(data_dir / "episode_000000.parquet", index=False)

        # 更新 episodes.jsonl
        with open(temp_dir / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"episode_index": 0, "length": 5}) + "\n")
            f.write(json.dumps({"episode_index": 1, "length": 3}) + "\n")

        loader = LeRobotEpisodeLoader(str(temp_dir))
        assert len(loader) == 2
        ep0 = loader[0]
        ep1 = loader[1]
        assert len(ep0) == 5
        assert len(ep1) == 3


class TestLeRobotVideoLoading:
    """视频帧读取测试。"""

    def _make_dataset_with_video(self, root, num_frames=3, state_dim=71, action_dim=29):
        """构造含视频文件的 LeRobot 数据集。"""
        import cv2
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"state_dim": state_dim, "action_dim": action_dim}, f)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": num_frames}) + "\n")
        with open(meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task_description": "walk"}) + "\n")

        data_dir = root / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(num_frames):
            rows.append({
                "observation.state": [float(x) for x in np.random.randn(state_dim)],
                "action": [float(x) for x in np.random.randn(action_dim)],
                "observation.images.front": [0],  # 占位列
                "episode_index": 0, "frame_index": i, "timestamp": i / 30.0,
            })
        pd.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet", index=False)

        # 创建视频文件
        videos_dir = root / "videos" / "chunk-000" / "front"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_path = videos_dir / "episode_000000.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (64, 64))
        for i in range(num_frames):
            frame = np.full((64, 64, 3), i * 80, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return root

    def test_video_frame_loading(self, temp_dir):
        """能从 mp4 读取视频帧。"""
        self._make_dataset_with_video(temp_dir, num_frames=3)
        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        frame = ep.get_frame(0)
        assert "front" in frame["images"]
        img = frame["images"]["front"]
        assert img.shape == (224, 224, 3)  # image_size 默认 224x224
        assert img.dtype == np.uint8

    def test_video_frame_index(self, temp_dir):
        """不同 frame_index 读取不同帧。"""
        self._make_dataset_with_video(temp_dir, num_frames=3)
        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        f0 = ep.get_frame(0)["images"]["front"]
        f2 = ep.get_frame(2)["images"]["front"]
        # 帧 0 和帧 2 像素值不同
        assert not np.array_equal(f0, f2)

    def test_missing_video_placeholder(self, temp_dir):
        """视频文件不存在时返回占位零张量。"""
        meta_dir = temp_dir / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"state_dim": 71, "action_dim": 29}, f)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 2}) + "\n")
        data_dir = temp_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        rows = []
        for i in range(2):
            rows.append({
                "observation.state": [0.0] * 71,
                "action": [0.0] * 29,
                "observation.images.front": None,  # None 占位
                "episode_index": 0, "frame_index": i, "timestamp": i / 30.0,
            })
        pd.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet", index=False)

        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        frame = ep.get_frame(0)
        img = frame["images"]["front"]
        assert img.shape == (224, 224, 3)
        np.testing.assert_array_equal(img, 0)


class TestLeRobotMultiColumnState:
    """state.*/action.* 多列分支测试。"""

    def test_state_multi_column(self, temp_dir):
        """无 observation.state 列但有 state.* 列时拼接。"""
        meta_dir = temp_dir / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"state_dim": 71, "action_dim": 29}, f)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 2}) + "\n")
        data_dir = temp_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        rows = []
        for i in range(2):
            rows.append({
                "state.joint_pos": [float(x) for x in np.random.randn(29)],
                "state.base_pos": [0.0, 0.0, 0.8],
                "action.joint_pos_delta": [float(x) for x in np.random.randn(29)],
                "episode_index": 0, "frame_index": i, "timestamp": i / 30.0,
            })
        pd.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet", index=False)

        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        frame = ep.get_frame(0)
        # state.joint_pos(29) + state.base_pos(3) = 32
        assert frame["state"].shape == (32,)
        assert frame["gt_action"].shape == (29,)

    def test_columns_property(self, temp_dir):
        """columns 属性返回 parquet 列名。"""
        _make_minimal_lerobot_dataset(temp_dir)
        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        cols = ep.columns
        assert "observation.state" in cols
        assert "action" in cols

    def test_empty_dataset(self, temp_dir):
        """无 parquet 文件时 episode 为空。"""
        meta_dir = temp_dir / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "info.json", "w") as f:
            json.dump({"state_dim": 71, "action_dim": 29}, f)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 0}) + "\n")
        # 不创建 data 目录
        loader = LeRobotEpisodeLoader(str(temp_dir))
        ep = loader[0]
        assert len(ep) == 0
        with pytest.raises(IndexError):
            ep.get_frame(0)

    def test_no_info_json_defaults(self, temp_dir):
        """无 info.json 时 state_dim/action_dim 为 None。"""
        meta_dir = temp_dir / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": 2}) + "\n")
        data_dir = temp_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        rows = []
        for i in range(2):
            rows.append({
                "observation.state": [0.0] * 71,
                "action": [0.0] * 29,
                "episode_index": 0, "frame_index": i, "timestamp": i / 30.0,
            })
        pd.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet", index=False)

        loader = LeRobotEpisodeLoader(str(temp_dir))
        assert loader.state_dim is None
        assert loader.action_dim is None
        # get_frame 仍能从 parquet 列读取
        frame = loader[0].get_frame(0)
        assert frame["state"].shape == (71,)
