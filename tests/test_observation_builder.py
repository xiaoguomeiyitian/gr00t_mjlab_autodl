"""测试观测构建模块。"""

import numpy as np
import pytest

from src.observation_builder import ObservationBuilder


class TestObservationBuilder:
    """ObservationBuilder 测试。"""

    def test_default_init(self):
        builder = ObservationBuilder()
        assert builder.state_dim == 71
        assert builder.image_size == (224, 224)
        assert builder.language_instruction == "perform the task"

    def test_custom_init(self):
        builder = ObservationBuilder(
            camera_keys=["cam1", "cam2"],
            state_dim=37,
            image_size=(112, 112),
            language_instruction="walk forward",
        )
        assert builder.camera_keys == ["cam1", "cam2"]
        assert builder.state_dim == 37
        assert builder.image_size == (112, 112)

    def test_build_basic(self):
        builder = ObservationBuilder(camera_keys=["front"])
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)

        assert "video" in obs
        assert "state" in obs
        assert "language" in obs
        assert "front" in obs["video"]

    def test_build_state_shape(self):
        builder = ObservationBuilder(camera_keys=["front"], state_dim=71)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.ones(71, dtype=np.float32)
        obs = builder.build(images, state)

        assert obs["state"].shape == (1, 71)
        np.testing.assert_array_equal(obs["state"][0], state)

    def test_build_language_override(self):
        builder = ObservationBuilder(camera_keys=["front"])
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state, language="pick up cube")
        assert obs["language"] == "pick up cube"

    def test_build_default_language(self):
        builder = ObservationBuilder(camera_keys=["front"], language_instruction="default task")
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert obs["language"] == "default task"

    def test_build_missing_camera(self):
        """缺失相机时跳过。"""
        builder = ObservationBuilder(camera_keys=["front", "wrist"])
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert "front" in obs["video"]
        # wrist 不存在，不应报错

    def test_build_image_resize(self):
        """测试图像 resize。"""
        builder = ObservationBuilder(camera_keys=["front"], image_size=(112, 112))
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert obs["video"]["front"].shape[:2] == (112, 112)

    def test_build_multiple_cameras(self):
        builder = ObservationBuilder(camera_keys=["front", "wrist"])
        images = {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist": np.ones((224, 224, 3), dtype=np.uint8) * 255,
        }
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert "front" in obs["video"]
        assert "wrist" in obs["video"]

    def test_build_go2_state(self):
        builder = ObservationBuilder(camera_keys=["front"], state_dim=37)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.random.randn(37).astype(np.float32)
        obs = builder.build(images, state)
        assert obs["state"].shape == (1, 37)
