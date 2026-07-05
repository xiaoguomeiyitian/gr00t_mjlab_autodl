"""测试本地推理模块。"""

import numpy as np

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
        """检测 INT4 LUT 量化模式（safetensors 内含 .quant key）。"""
        from safetensors.numpy import save_file
        model_dir = temp_dir / "model"
        model_dir.mkdir(exist_ok=True)
        weights = {
            "linear.weight.quant": np.zeros((4, 2), dtype=np.uint8),
            "linear.weight.absmax": np.ones((4, 1), dtype=np.float32),
        }
        save_file(weights, str(model_dir / "model.safetensors"))

        inference = object.__new__(GR00TLocalInference)
        inference.model_path = model_dir
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
        """测试观测构建（符合 check_observation 契约）。"""
        inference = object.__new__(GR00TLocalInference)
        inference.robot = "g1"
        inference.num_obs_steps = 1
        inference._obs_builder = None
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        obs = inference._build_observation(images, state, "walk forward")
        assert "video" in obs
        assert "state" in obs
        assert "language" in obs
        # 新契约：state/language 都是 dict
        assert isinstance(obs["state"], dict)
        assert isinstance(obs["language"], dict)
        # state 按 G1 切片拆分
        assert "joint_pos" in obs["state"]
        assert obs["state"]["joint_pos"].shape == (1, 1, 29)
        # language 是 {key: [[str]]}
        lk = "annotation.human.task_description"
        assert lk in obs["language"]
        assert obs["language"][lk][0][0] == "walk forward"

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
        """未知格式返回零向量（长度为 action_horizon）。"""
        inference = object.__new__(GR00TLocalInference)
        inference.action_horizon = 16
        action = inference._extract_action("invalid")
        assert action.ndim == 1
        assert action.shape[0] >= 1


class TestPredictBuffer:
    """predict 方法的缓冲逻辑测试（mock policy）。"""

    def _make_inference_with_mock_policy(self, action_chunk):
        """构造一个绕过 __init__ 的 GR00TLocalInference，policy 返回固定 chunk。"""
        inference = object.__new__(GR00TLocalInference)
        inference._closed = False
        inference.quant_mode = "none"
        inference.device = "cpu"
        inference.action_horizon = 16
        inference.robot = "g1"
        inference.num_obs_steps = 1
        inference._action_buffer = ActionChunkBuffer()
        inference._obs_builder = None

        class MockPolicy:
            def get_action(self, obs):
                return action_chunk, {}

        inference.policy = MockPolicy()
        return inference

    def test_predict_buffer_hit(self):
        """use_buffer=True 时，第二次 predict 应命中缓冲（source=buffer）。"""
        chunk = np.random.randn(16, 29).astype(np.float32)
        inf = self._make_inference_with_mock_policy(chunk)

        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        # 第一次：从模型取，缓存剩余 15 步
        action1, info1 = inf.predict(images, state, use_buffer=True)
        assert info1["source"] == "model"
        assert action1.shape == (29,)

        # 第二次：应命中缓冲
        action2, info2 = inf.predict(images, state, use_buffer=True)
        assert info2["source"] == "buffer"
        np.testing.assert_array_almost_equal(action2, chunk[1])

    def test_predict_no_buffer(self):
        """use_buffer=False 时不缓存，每次都从模型取。"""
        chunk = np.random.randn(16, 29).astype(np.float32)
        inf = self._make_inference_with_mock_policy(chunk)

        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        action1, info1 = inf.predict(images, state, use_buffer=False)
        action2, info2 = inf.predict(images, state, use_buffer=False)
        assert info1["source"] == "model"
        assert info2["source"] == "model"
        # 缓冲应仍为空
        assert inf._action_buffer.is_empty

    def test_predict_dict_result(self):
        """policy 返回 dict 时也能正确缓冲。"""
        chunk = np.random.randn(16, 29).astype(np.float32)
        inf = self._make_inference_with_mock_policy(chunk)
        # 改 mock policy 返回 dict
        class MockPolicyDict:
            def get_action(self, obs):
                return {"action": chunk}, {}
        inf.policy = MockPolicyDict()

        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        action1, info1 = inf.predict(images, state, use_buffer=True)
        assert info1["source"] == "model"
        assert action1.shape == (29,)

        action2, info2 = inf.predict(images, state, use_buffer=True)
        assert info2["source"] == "buffer"

    def test_reset_buffer(self):
        """reset_buffer 清空缓冲。"""
        chunk = np.random.randn(16, 29).astype(np.float32)
        inf = self._make_inference_with_mock_policy(chunk)
        images = {"front": np.zeros((224, 224, 3), dtype=np.uint8)}
        state = np.zeros(71, dtype=np.float32)

        inf.predict(images, state, use_buffer=True)
        assert not inf._action_buffer.is_empty
        inf.reset_buffer()
        assert inf._action_buffer.is_empty
