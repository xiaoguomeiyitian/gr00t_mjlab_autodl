"""测试格式转换模块。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.convert_to_lerobot import convert_to_lerobot, _build_modality_json, _create_placeholder_video


class TestBuildModalityJson:
    """_build_modality_json 测试。"""

    def test_g1_modality(self):
        modality = _build_modality_json("g1", 29, 71, ["front", "wrist"])
        assert "state" in modality
        assert "action" in modality
        assert "video" in modality
        assert "annotation" in modality
        assert modality["state"]["joint_pos"] == {"start": 0, "end": 29}
        assert modality["state"]["joint_vel"] == {"start": 29, "end": 58}
        assert "front" in modality["video"]
        assert "wrist" in modality["video"]

    def test_go2_modality(self):
        modality = _build_modality_json("go2", 12, 37, ["front", "back"])
        assert modality["state"]["joint_pos"] == {"start": 0, "end": 12}
        assert modality["state"]["joint_vel"] == {"start": 12, "end": 24}
        assert "front" in modality["video"]
        assert "back" in modality["video"]

    def test_state_slices_contiguous(self):
        modality = _build_modality_json("g1", 29, 71, ["front"])
        state = modality["state"]
        sorted_keys = sorted(state.keys(), key=lambda k: state[k]["start"])
        for i in range(len(sorted_keys) - 1):
            assert state[sorted_keys[i]]["end"] == state[sorted_keys[i + 1]]["start"]


class TestCreatePlaceholderVideo:
    """_create_placeholder_video 测试。"""

    def test_creates_file(self, temp_dir):
        path = str(temp_dir / "test.mp4")
        _create_placeholder_video(path, num_frames=5, fps=30, camera_names=["front"])
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestConvertToLeRobot:
    """convert_to_lerobot 集成测试。"""

    @pytest.fixture
    def raw_dir(self, temp_dir):
        """创建模拟原始数据目录。"""
        raw = temp_dir / "g1_raw"
        raw.mkdir()

        # 创建 2 个 episode
        for ep_idx in range(2):
            np.savez_compressed(
                str(raw / f"episode_{ep_idx:04d}.npz"),
                states=np.random.randn(10, 71).astype(np.float32),
                actions=np.random.randn(10, 29).astype(np.float32),
                rewards=np.random.randn(10).astype(np.float32),
            )
            # 创建 mp4
            _create_placeholder_video(
                str(raw / f"episode_{ep_idx:04d}.mp4"),
                num_frames=10, fps=30, camera_names=["front"],
            )

        # 创建 metadata
        meta = {
            "robot": "g1",
            "task": "test",
            "action_mode": "delta",
            "num_episodes": 2,
            "episode_length": 10,
            "fps": 30,
            "image_size": [32, 32],
            "state_dim": 71,
            "action_dim": 29,
            "num_joints": 29,
            "camera_names": ["front", "wrist"],
        }
        with open(raw / "collection_meta.json", "w") as f:
            json.dump(meta, f)

        return raw

    def test_creates_directory_structure(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        assert (output_dir / "meta").is_dir()
        assert (output_dir / "data" / "chunk-000").is_dir()
        assert (output_dir / "videos" / "chunk-000").is_dir()

    def test_creates_all_meta_files(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        assert (output_dir / "meta" / "info.json").exists()
        assert (output_dir / "meta" / "episodes.jsonl").exists()
        assert (output_dir / "meta" / "tasks.jsonl").exists()
        assert (output_dir / "meta" / "modality.json").exists()

    def test_info_json_content(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["robot_type"] == "g1"
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 20
        assert info["fps"] == 30

    def test_episodes_jsonl(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        with open(output_dir / "meta" / "episodes.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 2
        ep0 = json.loads(lines[0])
        assert ep0["episode_index"] == 0
        assert ep0["length"] == 10

    def test_parquet_created(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        parquet_files = list((output_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_parquet_content(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        import pandas as pd
        df = pd.read_parquet(str(output_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        assert len(df) == 20  # 2 episodes × 10 steps
        assert "observation.state" in df.columns
        assert "action" in df.columns
        assert "episode_index" in df.columns
        assert len(df.iloc[0]["observation.state"]) == 71
        assert len(df.iloc[0]["action"]) == 29

    def test_videos_copied(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
        )
        video_files = list((output_dir / "videos" / "chunk-000").glob("*.mp4"))
        assert len(video_files) == 2

    def test_missing_input_dir(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            convert_to_lerobot(
                input_dir="/nonexistent/path",
                output_dir=str(temp_dir / "out"),
                robot="g1",
            )

    def test_no_npz_files(self, temp_dir):
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            convert_to_lerobot(
                input_dir=str(empty_dir),
                output_dir=str(temp_dir / "out"),
                robot="g1",
            )

    def test_custom_task_description(self, raw_dir, temp_dir):
        output_dir = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw_dir),
            output_dir=str(output_dir),
            robot="g1",
            task_description="pick up the red cube",
        )
        with open(output_dir / "meta" / "tasks.jsonl") as f:
            task = json.loads(f.readline())
        assert task["task_description"] == "pick up the red cube"
