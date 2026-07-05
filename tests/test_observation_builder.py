"""测试观测构建模块（符合 GR00T check_observation 契约）。"""

import numpy as np
import pytest

from src.observation_builder import ObservationBuilder, state_slices_from_config


# G1 state_slices（与 g1_modality_config.py 对齐）
G1_SLICES = {
    "joint_pos": (0, 29),
    "joint_vel": (29, 58),
    "base_pos": (58, 61),
    "base_quat": (61, 65),
    "base_lin_vel": (65, 68),
    "base_ang_vel": (68, 71),
}


class TestObservationBuilder:
    """ObservationBuilder 测试。"""

    def test_default_init(self):
        builder = ObservationBuilder()
        assert builder.state_dim == 71
        assert builder.image_size == (224, 224)
        assert builder.language_instruction == "perform the task"
        assert builder.language_key == "annotation.human.task_description"
        assert builder.num_obs_steps == 1

    def test_custom_init(self):
        builder = ObservationBuilder(
            camera_keys=["cam1", "cam2"],
            state_dim=37,
            image_size=(112, 112),
            language_instruction="walk forward",
            state_slices={"joint_pos": (0, 12), "joint_vel": (12, 24)},
            language_key="annotation.human.task_description",
            num_obs_steps=2,
        )
        assert builder.camera_keys == ["cam1", "cam2"]
        assert builder.state_dim == 37
        assert builder.image_size == (112, 112)
        assert builder.num_obs_steps == 2

    def test_build_basic_structure(self):
        """build 产出符合 check_observation 的三层嵌套结构。"""
        builder = ObservationBuilder(camera_keys=["front"], state_dim=71, state_slices=G1_SLICES)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)

        # 顶层三个 modality
        assert set(obs.keys()) == {"video", "state", "language"}
        # 每个都是 dict
        assert isinstance(obs["video"], dict)
        assert isinstance(obs["state"], dict)
        assert isinstance(obs["language"], dict)
        assert "front" in obs["video"]

    def test_build_video_dims(self):
        """video 每个相机是 (B, T, H, W, 3) uint8。"""
        builder = ObservationBuilder(camera_keys=["front"], state_dim=71, state_slices=G1_SLICES)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)

        v = obs["video"]["front"]
        assert v.ndim == 5, f"video 应为 5 维 (B,T,H,W,3)，实际 {v.shape}"
        assert v.shape == (1, 1, 224, 224, 3)
        assert v.dtype == np.uint8

    def test_build_state_split(self):
        """state 按 state_slices 拆分成多个子键，每个 (B, T, D) float32。"""
        builder = ObservationBuilder(camera_keys=["front"], state_dim=71, state_slices=G1_SLICES)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.ones(71, dtype=np.float32)
        obs = builder.build(images, state)

        # 应有 6 个 state 子键
        assert set(obs["state"].keys()) == set(G1_SLICES.keys())
        # 每个子键维度正确
        assert obs["state"]["joint_pos"].shape == (1, 1, 29)
        assert obs["state"]["joint_vel"].shape == (1, 1, 29)
        assert obs["state"]["base_pos"].shape == (1, 1, 3)
        assert obs["state"]["base_quat"].shape == (1, 1, 4)
        # dtype
        for k, arr in obs["state"].items():
            assert arr.dtype == np.float32, f"{k} dtype 应为 float32"
            assert arr.ndim == 3, f"{k} 应为 3 维 (B,T,D)"
        # 值正确（state 全 1）
        np.testing.assert_array_equal(obs["state"]["joint_pos"][0, 0], np.ones(29, dtype=np.float32))

    def test_build_language_format(self):
        """language 是 {language_key: [[str]]}，形状 (B, T)。"""
        builder = ObservationBuilder(camera_keys=["front"], state_dim=71, state_slices=G1_SLICES)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state, language="pick up cube")

        lang = obs["language"]
        assert isinstance(lang, dict)
        assert "annotation.human.task_description" in lang
        val = lang["annotation.human.task_description"]
        # 外层 B=1，中层 T=1，内层 1 个 str
        assert isinstance(val, list)
        assert len(val) == 1  # B
        assert isinstance(val[0], list)
        assert len(val[0]) == 1  # T
        assert val[0][0] == "pick up cube"

    def test_build_default_language(self):
        builder = ObservationBuilder(
            camera_keys=["front"], state_dim=71, state_slices=G1_SLICES,
            language_instruction="default task",
        )
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert obs["language"]["annotation.human.task_description"][0][0] == "default task"

    def test_build_missing_camera(self):
        """缺失相机时用零张量填充并警告。"""
        builder = ObservationBuilder(camera_keys=["front", "wrist"], state_dim=71, state_slices=G1_SLICES)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        with pytest.warns(UserWarning, match="缺失"):
            obs = builder.build(images, state)
        assert "front" in obs["video"]
        assert "wrist" in obs["video"]
        # 缺失相机填充为 (B, T, H, W, 3)
        assert obs["video"]["wrist"].shape == (1, 1, 224, 224, 3)

    def test_build_image_resize(self):
        """测试图像 resize 到 image_size。"""
        builder = ObservationBuilder(
            camera_keys=["front"], state_dim=71, state_slices=G1_SLICES,
            image_size=(112, 112),
        )
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        # resize 后 H, W = 112, 112
        assert obs["video"]["front"].shape == (1, 1, 112, 112, 3)

    def test_build_multiple_cameras(self):
        builder = ObservationBuilder(camera_keys=["front", "wrist"], state_dim=71, state_slices=G1_SLICES)
        images = {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist": np.ones((224, 224, 3), dtype=np.uint8) * 255,
        }
        state = np.zeros(71, dtype=np.float32)
        obs = builder.build(images, state)
        assert "front" in obs["video"]
        assert "wrist" in obs["video"]
        # 两个相机 batch size 一致
        assert obs["video"]["front"].shape[0] == obs["video"]["wrist"].shape[0]

    def test_build_go2_state(self):
        """go2 用 state_slices_from_config 拆分。"""
        slices = state_slices_from_config("go2")
        builder = ObservationBuilder(camera_keys=["front"], state_dim=37, state_slices=slices)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.random.randn(37).astype(np.float32)
        obs = builder.build(images, state)
        assert obs["state"]["joint_pos"].shape == (1, 1, 12)
        assert obs["state"]["base_quat"].shape == (1, 1, 4)

    def test_build_state_2d_input(self):
        """state 支持 (T, D) 输入，video 单帧广播到 num_obs_steps。"""
        builder = ObservationBuilder(
            camera_keys=["front"], state_dim=71, state_slices=G1_SLICES,
            num_obs_steps=3,
        )
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.ones((3, 71), dtype=np.float32)
        obs = builder.build(images, state)
        # state T=3
        assert obs["state"]["joint_pos"].shape == (1, 3, 29)
        # video 单帧广播到 T=3
        assert obs["video"]["front"].shape == (1, 3, 224, 224, 3)

    def test_state_slices_from_config(self):
        """state_slices_from_config 返回各机器人正确切片。"""
        g1 = state_slices_from_config("g1")
        assert g1["joint_pos"] == (0, 29)
        assert g1["base_ang_vel"] == (68, 71)

        go2 = state_slices_from_config("go2")
        assert go2["joint_pos"] == (0, 12)
        assert go2["base_quat"] == (27, 31)

        h1 = state_slices_from_config("h1")
        assert h1["joint_pos"] == (0, 20)

    def test_contract_compliance(self):
        """模拟 Gr00tPolicy.check_observation 的关键校验。"""
        builder = ObservationBuilder(
            camera_keys=["front", "wrist"], state_dim=71, state_slices=G1_SLICES,
            num_obs_steps=1,
        )
        images = {
            "front": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        state = np.random.randn(71).astype(np.float32)
        obs = builder.build(images, state, language="walk")

        # 模拟 check_observation 的断言
        for modality in ["video", "state", "language"]:
            assert modality in obs
            assert isinstance(obs[modality], dict)

        # video: (B, T, H, W, 3) uint8
        for cam in ["front", "wrist"]:
            v = obs["video"][cam]
            assert v.dtype == np.uint8
            assert v.ndim == 5
            assert v.shape[1] == 1  # T
            assert v.shape[-1] == 3  # C

        # state: (B, T, D) float32
        for sk in G1_SLICES.keys():
            s = obs["state"][sk]
            assert s.dtype == np.float32
            assert s.ndim == 3
            assert s.shape[1] == 1  # T

        # language: list[list[list[str]]]
        lk = "annotation.human.task_description"
        assert lk in obs["language"]
        val = obs["language"][lk]
        assert isinstance(val, list)
        assert isinstance(val[0], list)
        assert isinstance(val[0][0], str)
