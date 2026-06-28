"""测试数据采集模块。"""

import numpy as np
import pytest

from src.collect_data import DataCollector, ROBOT_CONFIGS, _MockEnv


class TestRobotConfigs:
    """机器人配置常量测试。"""

    def test_g1_config(self):
        assert ROBOT_CONFIGS["g1"]["num_joints"] == 29
        assert ROBOT_CONFIGS["g1"]["state_dim"] == 71
        assert ROBOT_CONFIGS["g1"]["action_dim"] == 29
        assert "front" in ROBOT_CONFIGS["g1"]["camera_names"]
        assert "wrist" in ROBOT_CONFIGS["g1"]["camera_names"]

    def test_go2_config(self):
        assert ROBOT_CONFIGS["go2"]["num_joints"] == 12
        assert ROBOT_CONFIGS["go2"]["state_dim"] == 37
        assert ROBOT_CONFIGS["go2"]["action_dim"] == 12

    def test_all_configs_have_required_keys(self):
        required = ["task", "num_joints", "state_dim", "action_dim", "camera_names"]
        for robot, config in ROBOT_CONFIGS.items():
            for key in required:
                assert key in config, f"{robot} missing {key}"


class TestDataCollector:
    """DataCollector 测试。"""

    def test_init_g1(self):
        collector = DataCollector(robot="g1")
        assert collector.robot == "g1"
        assert collector.config["num_joints"] == 29

    def test_init_go2(self):
        collector = DataCollector(robot="go2")
        assert collector.robot == "go2"
        assert collector.config["num_joints"] == 12

    def test_init_invalid_robot(self):
        with pytest.raises(ValueError, match="不支持的机器人"):
            DataCollector(robot="invalid")

    def test_init_custom_params(self):
        collector = DataCollector(
            robot="g1",
            action_mode="absolute",
            num_episodes=10,
            episode_length=100,
            fps=15,
            image_size=(112, 112),
        )
        assert collector.action_mode == "absolute"
        assert collector.num_episodes == 10
        assert collector.episode_length == 100
        assert collector.fps == 15
        assert collector.image_size == (112, 112)

    def test_extract_state_shape(self):
        collector = DataCollector(robot="g1")
        obs = {
            "qpos": np.ones(29),
            "qvel": np.ones(29),
            "base_pos": np.ones(3),
            "base_quat": np.array([1, 0, 0, 0]),
            "base_lin_vel": np.ones(3),
            "base_ang_vel": np.ones(3),
        }
        state = collector._extract_state(obs)
        assert state.shape == (71,)

    def test_extract_state_go2(self):
        collector = DataCollector(robot="go2")
        obs = {
            "qpos": np.ones(12),
            "qvel": np.ones(12),
            "base_pos": np.ones(3),
            "base_quat": np.array([1, 0, 0, 0]),
            "base_lin_vel": np.ones(3),
            "base_ang_vel": np.ones(3),
        }
        state = collector._extract_state(obs)
        assert state.shape == (37,)

    def test_extract_state_missing_keys(self):
        """缺失 key 时用零填充（base_quat 默认 [1,0,0,0]）。"""
        collector = DataCollector(robot="g1")
        state = collector._extract_state({})
        assert state.shape == (71,)
        # base_quat 默认值为 [1, 0, 0, 0]
        np.testing.assert_array_equal(state[61:65], [1, 0, 0, 0])

    def test_extract_image(self):
        collector = DataCollector(robot="g1")
        obs = {
            "images": {
                "front": np.ones((224, 224, 3), dtype=np.uint8),
                "wrist": np.ones((224, 224, 3), dtype=np.uint8) * 2,
            }
        }
        images = collector._extract_image(obs)
        assert "front" in images
        assert "wrist" in images
        assert images["front"].shape == (224, 224, 3)

    def test_extract_image_resize(self):
        """测试图像 resize。"""
        collector = DataCollector(robot="g1", image_size=(112, 112))
        obs = {
            "images": {
                "front": np.ones((224, 224, 3), dtype=np.uint8),
            }
        }
        images = collector._extract_image(obs)
        assert images["front"].shape[:2] == (112, 112)

    def test_generate_action_delta(self):
        collector = DataCollector(robot="g1", action_mode="delta")
        state = np.zeros(71)
        action = collector._generate_action(state, 0)
        assert action.shape == (29,)
        assert action.dtype == np.float32

    def test_generate_action_absolute(self):
        collector = DataCollector(robot="g1", action_mode="absolute")
        state = np.zeros(71)
        action = collector._generate_action(state, 0)
        assert action.shape == (29,)

    def test_generate_action_relative_eef(self):
        collector = DataCollector(robot="g1", action_mode="relative_eef")
        state = np.zeros(71)
        action = collector._generate_action(state, 0)
        assert action.shape == (29,)

    def test_generate_action_invalid_mode(self):
        collector = DataCollector(robot="g1", action_mode="invalid")
        with pytest.raises(ValueError, match="未知动作模式"):
            collector._generate_action(np.zeros(71), 0)

    def test_run_produces_output(self, temp_dir):
        """完整运行产生 npz + mp4 文件。"""
        collector = DataCollector(
            robot="g1",
            num_episodes=2,
            episode_length=5,
            image_size=(32, 32),
        )
        output_dir = str(temp_dir / "raw")
        stats = collector.run(output_dir=output_dir)

        assert stats["total_steps"] == 10
        assert len(stats["episodes"]) == 2

        import os
        files = os.listdir(output_dir)
        assert any(f.endswith(".npz") for f in files)
        assert any(f.endswith(".mp4") for f in files)
        assert "collection_meta.json" in files

    def test_metadata_content(self, temp_dir):
        """验证 metadata 内容。"""
        collector = DataCollector(
            robot="g1",
            num_episodes=1,
            episode_length=3,
            image_size=(32, 32),
        )
        output_dir = str(temp_dir / "raw")
        collector.run(output_dir=output_dir)

        import json, os
        with open(os.path.join(output_dir, "collection_meta.json")) as f:
            meta = json.load(f)
        assert meta["robot"] == "g1"
        assert meta["num_episodes"] == 1
        assert meta["state_dim"] == 71
        assert meta["action_dim"] == 29


class TestMockEnv:
    """Mock 环境测试。"""

    def test_reset(self):
        config = ROBOT_CONFIGS["g1"]
        env = _MockEnv(config, np.random.RandomState(42))
        obs = env.reset()
        assert "qpos" in obs
        assert "images" in obs

    def test_step(self):
        config = ROBOT_CONFIGS["g1"]
        env = _MockEnv(config, np.random.RandomState(42))
        env.reset()
        obs, reward, done, info = env.step(np.zeros(29))
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_done_after_max_steps(self):
        config = ROBOT_CONFIGS["g1"]
        env = _MockEnv(config, np.random.RandomState(42))
        env.reset()
        for _ in range(300):
            obs, reward, done, info = env.step(np.zeros(29))
        assert done
