"""测试 viz 推理模块（不依赖 viser/mujoco 的纯逻辑部分）。"""

import numpy as np


class TestViserInferActionExtraction:
    """ViserInferLoop._extract_action_vector 测试。"""

    def _make_loop(self):
        from src.viz.viser_infer import ViserInferLoop
        return object.__new__(ViserInferLoop)

    def test_dict_btd(self):
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)
        assert v.dtype == np.float32

    def test_dict_td(self):
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_dict_d(self):
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_ndarray_btd(self):
        loop = self._make_loop()
        action = np.random.randn(1, 16, 29).astype(np.float32)
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_ndarray_td(self):
        loop = self._make_loop()
        action = np.random.randn(8, 29).astype(np.float32)
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_ndarray_d(self):
        loop = self._make_loop()
        action = np.random.randn(29).astype(np.float32)
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_tuple_action_info(self):
        """(action_dict, info) tuple 防御性处理。"""
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector((action, {"info": 1}))
        assert v.shape == (29,)

    def test_tuple_ndarray_info(self):
        """(ndarray, info) tuple 不应被拆（ndarray 是 action 本身）。"""
        loop = self._make_loop()
        action = np.random.randn(1, 16, 29).astype(np.float32)
        # ndarray 作为 tuple 第一项，不应递归
        v = loop._extract_action_vector((action, {}))
        assert v.shape == (29,)

    def test_dict_joint_position_key(self):
        loop = self._make_loop()
        action = {"joint_position": np.random.randn(1, 8, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_dict_joint_position_delta_backcompat(self):
        """旧 key joint_position_delta 仍可识别（向后兼容）。"""
        loop = self._make_loop()
        action = {"joint_position_delta": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_dict_action_key(self):
        loop = self._make_loop()
        action = {"action": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_dict_fallback_first_value(self):
        """无已知 key 时取第一个值。"""
        loop = self._make_loop()
        action = {"unknown_key": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_squeeze_high_dim(self):
        """(B,T,D) → (D,)。"""
        loop = self._make_loop()
        arr = np.random.randn(2, 16, 29).astype(np.float32)
        v = loop._squeeze_action(arr)
        assert v.shape == (29,)

    def test_squeeze_1d(self):
        """(D,) 不变。"""
        loop = self._make_loop()
        arr = np.random.randn(29).astype(np.float32)
        v = loop._squeeze_action(arr)
        assert v.shape == (29,)


class TestMuJoCoInferActionExtraction:
    """MuJoCoInferLoop._extract_action_vector 测试。"""

    def _make_loop(self):
        from src.viz.mujoco_infer import MuJoCoInferLoop
        return object.__new__(MuJoCoInferLoop)

    def test_dict_btd(self):
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_ndarray_td(self):
        loop = self._make_loop()
        action = np.random.randn(16, 29).astype(np.float32)
        v = loop._extract_action_vector(action)
        assert v.shape == (29,)

    def test_tuple_action_info(self):
        loop = self._make_loop()
        action = {"joint_pos": np.random.randn(1, 16, 29).astype(np.float32)}
        v = loop._extract_action_vector((action, {}))
        assert v.shape == (29,)


class TestModalityConfigParsing:
    """viz connect() 的 modality_config 解析逻辑测试。"""

    def _parse_video_keys(self, modality_config):
        """复用 viser_infer.connect() 的解析逻辑。"""
        video_keys = ["exterior_image_1_left"]
        if isinstance(modality_config, dict):
            video_cfg = modality_config.get("video", {})
            if isinstance(video_cfg, dict):
                mk = video_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    video_keys = mk
            else:
                mk = getattr(video_cfg, "modality_keys", None)
                if mk:
                    video_keys = list(mk)
        return video_keys

    def test_dict_with_modality_keys(self):
        cfg = {"video": {"delta_indices": [0], "modality_keys": ["front", "wrist"], "action_configs": None}}
        assert self._parse_video_keys(cfg) == ["front", "wrist"]

    def test_dict_missing_modality_keys(self):
        cfg = {"video": {"delta_indices": [0]}}
        assert self._parse_video_keys(cfg) == ["exterior_image_1_left"]

    def test_dict_empty_modality_keys(self):
        cfg = {"video": {"modality_keys": []}}
        assert self._parse_video_keys(cfg) == ["exterior_image_1_left"]

    def test_dict_video_missing(self):
        cfg = {"state": {}}
        assert self._parse_video_keys(cfg) == ["exterior_image_1_left"]

    def test_not_dict(self):
        assert self._parse_video_keys(None) == ["exterior_image_1_left"]
        assert self._parse_video_keys("string") == ["exterior_image_1_left"]

    def test_object_with_modality_keys(self):
        """ModalityConfig 对象（本地直连场景）。"""
        class FakeMC:
            modality_keys = ["cam1", "cam2"]
        cfg = {"video": FakeMC()}
        assert self._parse_video_keys(cfg) == ["cam1", "cam2"]

    def test_go2_camera_names(self):
        cfg = {"video": {"modality_keys": ["front", "back"]}}
        assert self._parse_video_keys(cfg) == ["front", "back"]


class TestViserInferDefaults:
    """ViserInferLoop 默认值测试。"""

    def test_default_embodiment_tag(self):
        from src.viz.viser_infer import ViserInferLoop
        import inspect
        sig = inspect.signature(ViserInferLoop.__init__)
        assert sig.parameters["embodiment_tag"].default == "NEW_EMBODIMENT"

    def test_default_host_port(self):
        from src.viz.viser_infer import ViserInferLoop
        import inspect
        sig = inspect.signature(ViserInferLoop.__init__)
        assert sig.parameters["host"].default == "127.0.0.1"
        assert sig.parameters["port"].default == 5555


class TestMuJoCoInferDefaults:
    """MuJoCoInferLoop 默认值测试。"""

    def test_default_embodiment_tag(self):
        from src.viz.mujoco_infer import MuJoCoInferLoop
        import inspect
        sig = inspect.signature(MuJoCoInferLoop.__init__)
        assert sig.parameters["embodiment_tag"].default == "NEW_EMBODIMENT"
