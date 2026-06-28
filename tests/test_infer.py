"""测试本地推理模块。"""

import numpy as np
import pytest

from src.infer import ActionChunkBuffer, GR00TLocalInference


class TestActionChunkBuffer:
    """ActionChunkBuffer 测试。"""

    def test_initial_state(self):
        buf = ActionChunkBuffer()
        assert buf.is_empty
        assert buf.pop() is None

    def test_push_pop_single(self):
        buf = ActionChunkBuffer()
        actions = np.array([[1.0, 2.0], [3.0, 4.0]])
        buf.push(actions)

        a1 = buf.pop()
        np.testing.assert_array_equal(a1, [1.0, 2.0])
        assert not buf.is_empty

        a2 = buf.pop()
        np.testing.assert_array_equal(a2, [3.0, 4.0])
        assert buf.is_empty

    def test_pop_exhausted(self):
        buf = ActionChunkBuffer()
        buf.push(np.array([[1.0]]))
        buf.pop()
        assert buf.pop() is None

    def test_clear(self):
        buf = ActionChunkBuffer()
        buf.push(np.array([[1.0], [2.0]]))
        buf.clear()
        assert buf.is_empty
        assert buf.pop() is None

    def test_push_overwrite(self):
        """新 push 覆盖旧数据。"""
        buf = ActionChunkBuffer()
        buf.push(np.array([[1.0], [2.0]]))
        buf.push(np.array([[10.0]]))
        a = buf.pop()
        np.testing.assert_array_equal(a, [10.0])

    def test_multidim_actions(self):
        buf = ActionChunkBuffer()
        actions = np.random.randn(16, 29).astype(np.float32)
        buf.push(actions)
        for i in range(16):
            a = buf.pop()
            np.testing.assert_array_almost_equal(a, actions[i])
        assert buf.is_empty


class TestGR00TLocalInference:
    """GR00TLocalInference 测试（不需要实际模型）。"""

    def test_detect_quant_mode_int4_lut(self, temp_dir):
        """检测 INT4 LUT 量化模式。"""
        # 创建 .quant 文件
        (temp_dir / "model.quant").touch()
        (temp_dir / "config.json").write_text("{}")

        # 只测试 _detect_quant_mode，不加载模型
        inference = object.__new__(GR00TLocalInference)
        inference.model_path = temp_dir
        mode = inference._detect_quant_mode()
        assert mode == "int4_lut"

    def test_detect_quant_mode_none(self, temp_dir):
        """检测无量化模式。"""
        (temp_dir / "config.json").write_text("{}")

        inference = object.__new__(GR00TLocalInference)
        inference.model_path = temp_dir
        mode = inference._detect_quant_mode()
        assert mode == "none"

    def test_detect_quant_mode_bnb4(self, temp_dir):
        """检测 BitsAndBytes 4bit 模式。"""
        import json
        config = {"quantization_config": {"load_in_4bit": True}}
        (temp_dir / "config.json").write_text(json.dumps(config))

        inference = object.__new__(GR00TLocalInference)
        inference.model_path = temp_dir
        mode = inference._detect_quant_mode()
        assert mode == "int4_bnb"

    def test_detect_quant_mode_bnb8(self, temp_dir):
        """检测 BitsAndBytes 8bit 模式。"""
        import json
        config = {"quantization_config": {"load_in_8bit": True}}
        (temp_dir / "config.json").write_text(json.dumps(config))

        inference = object.__new__(GR00TLocalInference)
        inference.model_path = temp_dir
        mode = inference._detect_quant_mode()
        assert mode == "int8"

    def test_build_observation(self, temp_dir):
        """测试观测构建。"""
        inference = object.__new__(GR00TLocalInference)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        obs = inference._build_observation(images, state, "walk forward")
        assert "video" in obs
        assert "state" in obs
        assert "language" in obs
        assert obs["language"] == "walk forward"
        assert obs["state"].shape == (1, 71)

    def test_extract_action_ndarray_2d(self, temp_dir):
        """从 2D ndarray 提取动作。"""
        inference = object.__new__(GR00TLocalInference)
        action_data = np.random.randn(16, 29).astype(np.float32)
        action = inference._extract_action(action_data)
        assert action.shape == (29,)
        np.testing.assert_array_almost_equal(action, action_data[0])

    def test_extract_action_ndarray_1d(self, temp_dir):
        """从 1D ndarray 提取动作。"""
        inference = object.__new__(GR00TLocalInference)
        action_data = np.random.randn(29).astype(np.float32)
        action = inference._extract_action(action_data)
        assert action.shape == (29,)

    def test_extract_action_dict(self, temp_dir):
        """从 dict 提取动作。"""
        inference = object.__new__(GR00TLocalInference)
        action_data = {"action": np.random.randn(16, 29).astype(np.float32)}
        action = inference._extract_action(action_data)
        assert action.shape == (29,)

    def test_extract_action_list(self, temp_dir):
        """从 list 提取动作。"""
        inference = object.__new__(GR00TLocalInference)
        action_data = [np.random.randn(29).astype(np.float32) for _ in range(4)]
        action = inference._extract_action(action_data)
        assert action.shape == (29,)

    def test_extract_action_fallback(self, temp_dir):
        """未知格式返回零向量。"""
        inference = object.__new__(GR00TLocalInference)
        action = inference._extract_action("invalid")
        assert action.shape == (29,)
        np.testing.assert_array_equal(action, np.zeros(29))
