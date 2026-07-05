"""测试 sim_playback 模块（不依赖 MuJoCo 的纯逻辑部分）。"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.sim_playback import (
    ROBOT_CONFIGS,
    find_robot_mjcf,
    _save_video,
    _save_metadata,
    _collect_episode,
    load_motion_data,
)


class TestRobotConfigs:
    """ROBOT_CONFIGS 一致性测试。"""

    def test_all_robots_present(self):
        expected = {"g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"}
        assert set(ROBOT_CONFIGS.keys()) == expected

    def test_state_dim_equals_2x_joints_plus_13(self):
        """state_dim = num_joints*2 + base_pos(3) + base_quat(4) + lin_vel(3) + ang_vel(3)。"""
        for robot, cfg in ROBOT_CONFIGS.items():
            assert cfg["state_dim"] == cfg["num_joints"] * 2 + 13, f"{robot} state_dim 错误"

    def test_action_dim_equals_num_joints(self):
        for robot, cfg in ROBOT_CONFIGS.items():
            assert cfg["action_dim"] == cfg["num_joints"], f"{robot} action_dim 错误"

    def test_camera_names_present(self):
        for robot, cfg in ROBOT_CONFIGS.items():
            assert len(cfg["camera_names"]) >= 1, f"{robot} 无相机"

    def test_go2_uses_back_camera(self):
        assert "back" in ROBOT_CONFIGS["go2"]["camera_names"]
        assert "front" in ROBOT_CONFIGS["go2"]["camera_names"]

    def test_g1_uses_wrist_camera(self):
        assert "wrist" in ROBOT_CONFIGS["g1"]["camera_names"]


class TestFindRobotMjcf:
    """find_robot_mjcf 路径查找测试。"""

    def test_not_found_raises(self):
        """不存在的机器人名应抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="nonexistent_robot"):
            find_robot_mjcf("nonexistent_robot")

    def test_returns_str(self):
        """返回值应为 str 类型（找到时）或抛异常。"""
        # 不存在的机器人会抛异常，这里验证类型约定
        try:
            result = find_robot_mjcf("g1")
            assert isinstance(result, str)
        except FileNotFoundError:
            pass  # 本地无 MJCF 文件时可接受


class TestSaveVideo:
    """_save_video 测试。"""

    def test_creates_video_file(self, temp_dir):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        path = str(temp_dir / "test.mp4")
        _save_video(frames, path, fps=30.0)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_empty_frames_noop(self, temp_dir):
        path = str(temp_dir / "empty.mp4")
        _save_video([], path, fps=30.0)
        # 空帧列表不应创建文件或创建空文件
        assert not Path(path).exists() or Path(path).stat().st_size == 0

    def test_bgr_passthrough(self, temp_dir):
        """ndim==3 且 shape[2]==3 的帧会转 BGR。"""
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        path = str(temp_dir / "bgr.mp4")
        _save_video(frames, path, fps=30.0)
        assert Path(path).exists()


class TestSaveMetadata:
    """_save_metadata 测试。"""

    def test_writes_valid_json(self, temp_dir):
        stats = {"total_steps": 100, "episodes": [{"episode": 0, "steps": 100}]}
        _save_metadata(
            output_path=temp_dir, stats=stats, robot="g1",
            task_description="walk", action_mode="delta",
            num_episodes=1, episode_length=100, fps=30.0,
            image_size=(224, 224), motion_frames=100,
            motion_file="/tmp/test.csv",
        )
        meta_path = temp_dir / "collection_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["robot"] == "g1"
        assert meta["task"] == "walk"
        assert meta["action_mode"] == "delta"
        assert meta["state_dim"] == 71
        assert meta["action_dim"] == 29
        assert meta["camera_names"] == ["front", "wrist"]
        assert meta["source"] == "sim_playback"
        assert meta["total_steps"] == 100
        assert meta["motion_frames"] == 100

    def test_go2_metadata(self, temp_dir):
        _save_metadata(
            output_path=temp_dir, stats={"total_steps": 0, "episodes": []},
            robot="go2", task_description="trot", action_mode="delta",
            num_episodes=1, episode_length=50, fps=30.0,
            image_size=(224, 224), motion_frames=50,
            motion_file="/tmp/x.csv",
        )
        meta = json.loads((temp_dir / "collection_meta.json").read_text())
        assert meta["state_dim"] == 37
        assert meta["action_dim"] == 12


class TestCollectEpisode:
    """_collect_episode 的 delta/absolute 逻辑（mock sim）。"""

    def _make_mock_sim(self, num_joints=29, image_size=(32, 32)):
        sim = MagicMock()
        sim.image_size = image_size
        sim.step = MagicMock()
        sim.get_state = MagicMock(return_value=np.zeros(71, dtype=np.float32))
        sim.render_all_cameras = MagicMock(
            return_value={"front": np.zeros(image_size + (3,), dtype=np.uint8),
                          "wrist": np.zeros(image_size + (3,), dtype=np.uint8)}
        )
        return sim

    def test_delta_mode(self, temp_dir):
        """delta 模式：action = joint_pos[t+1] - joint_pos[t]。"""
        T = 10
        num_joints = 29
        joint_pos = np.random.randn(T, num_joints).astype(np.float32)
        base_pos = np.zeros((T, 3), dtype=np.float32)
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0

        sim = self._make_mock_sim()
        config = ROBOT_CONFIGS["g1"]
        result = _collect_episode(
            sim=sim, base_pos=base_pos, base_quat=base_quat, joint_pos=joint_pos,
            ep_idx=0, output_path=temp_dir, config=config,
            num_joints=num_joints, episode_length=5,
            action_mode="delta", actual_fps=30.0,
            task_description="walk",
        )
        assert result["steps"] == 5
        assert result["episode"] == 0
        # 验证 npz
        npz = np.load(temp_dir / "episode_0000.npz")
        actions = npz["actions"]
        # 第 0 步 delta = joint_pos[1] - joint_pos[0]
        np.testing.assert_array_almost_equal(actions[0], joint_pos[1] - joint_pos[0])
        # 验证 mp4 生成
        assert (temp_dir / "episode_0000_front.mp4").exists()
        assert (temp_dir / "episode_0000_wrist.mp4").exists()

    def test_absolute_mode(self, temp_dir):
        """absolute 模式：action = joint_pos[t]。"""
        T = 8
        num_joints = 29
        joint_pos = np.random.randn(T, num_joints).astype(np.float32)
        base_pos = np.zeros((T, 3), dtype=np.float32)
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0

        sim = self._make_mock_sim()
        config = ROBOT_CONFIGS["g1"]
        result = _collect_episode(
            sim=sim, base_pos=base_pos, base_quat=base_quat, joint_pos=joint_pos,
            ep_idx=1, output_path=temp_dir, config=config,
            num_joints=num_joints, episode_length=3,
            action_mode="absolute", actual_fps=30.0,
            task_description="walk",
        )
        npz = np.load(temp_dir / "episode_0001.npz")
        actions = npz["actions"]
        np.testing.assert_array_almost_equal(actions[0], joint_pos[0])
        np.testing.assert_array_almost_equal(actions[2], joint_pos[2])

    def test_loop_boundary_delta_zero(self, temp_dir):
        """循环边界处（t_next 回绕到 0）delta 置 0。"""
        T = 5
        num_joints = 4
        joint_pos = np.random.randn(T, num_joints).astype(np.float32)
        base_pos = np.zeros((T, 3), dtype=np.float32)
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0

        sim = self._make_mock_sim()
        config = {"camera_names": ["front"], "num_joints": num_joints}
        # episode_length = T，最后一步 t=T-1=4, t_next=0 回绕 → delta=0
        _collect_episode(
            sim=sim, base_pos=base_pos, base_quat=base_quat, joint_pos=joint_pos,
            ep_idx=0, output_path=temp_dir, config=config,
            num_joints=num_joints, episode_length=T,
            action_mode="delta", actual_fps=30.0,
            task_description="test",
        )
        npz = np.load(temp_dir / "episode_0000.npz")
        actions = npz["actions"]
        # 最后一步（step=T-1=4）delta 应为 0
        np.testing.assert_array_almost_equal(actions[T - 1], np.zeros(num_joints))

    def test_invalid_mode_zeros(self, temp_dir):
        """未知 action_mode 返回零动作。"""
        T = 3
        num_joints = 4
        joint_pos = np.random.randn(T, num_joints).astype(np.float32)
        base_pos = np.zeros((T, 3), dtype=np.float32)
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0

        sim = self._make_mock_sim()
        config = {"camera_names": ["front"], "num_joints": num_joints}
        _collect_episode(
            sim=sim, base_pos=base_pos, base_quat=base_quat, joint_pos=joint_pos,
            ep_idx=0, output_path=temp_dir, config=config,
            num_joints=num_joints, episode_length=2,
            action_mode="invalid", actual_fps=30.0,
            task_description="test",
        )
        npz = np.load(temp_dir / "episode_0000.npz")
        np.testing.assert_array_almost_equal(npz["actions"], np.zeros((2, num_joints)))


class TestLoadMotionData:
    """load_motion_data 委托给 RetargetMotionLoader。"""

    def test_csv_round_trip(self, temp_dir):
        csv = temp_dir / "motion.csv"
        T, nj = 5, 4
        data = np.random.randn(T, 7 + nj).astype(np.float32)
        # RetargetMotionLoader._load_csv 用 skiprows=1，需写一行 header
        header = ",".join([f"col{i}" for i in range(7 + nj)])
        np.savetxt(str(csv), data, delimiter=",", header=header, comments="")
        base_pos, base_quat, joint_pos, fps = load_motion_data(str(csv), fps=30.0)
        assert base_pos.shape == (T, 3)
        assert base_quat.shape == (T, 4)
        assert joint_pos.shape == (T, nj)
        assert fps == 30.0
