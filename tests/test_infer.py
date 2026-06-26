#!/usr/bin/env python3.12
"""测试 infer.py:
   - 初始化参数 (instruction 透传)
   - 设备/量化模式自动检测
   - _build_policy_observation
   - Action chunking 队列管理
   - RELATIVE 模式 delta 累加
"""
import sys, os, pytest, numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: 用 mock 隔离 mjlab / gr00t 依赖
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_torch():
    """Mock torch 模块, 让 device='auto' 能选择 cpu."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": fake_torch}):
        yield fake_torch


@pytest.fixture
def infer_instance(mock_torch, tmp_path):
    """构造一个不触发 GR00T 模型加载的 GR00TLocalInference 实例."""
    from infer import GR00TLocalInference
    # model_path 只要存在即可 (load() 才真正去读模型)
    fake_model = tmp_path / "fake_model"
    fake_model.mkdir()
    return GR00TLocalInference(
        model_path=str(fake_model),
        robot="g1",
        quantize="none",
        device="auto",
        instruction="walk forward",
    )


class TestInstructionPropagation:
    """instruction 参数必须正确透传到实例。"""

    def test_default_instruction_stored(self, infer_instance):
        assert infer_instance.instruction == "walk forward"

    def test_custom_instruction_stored(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model),
            robot="go2",
            instruction="turn left and walk slowly",
        )
        assert inst.instruction == "turn left and walk slowly"


class TestDeviceAutoDetection:
    """device='auto' → 根据 torch.cuda.is_available() 选 cuda/cpu."""

    def test_auto_picks_cpu_when_no_cuda(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), device="auto")
        assert inst.device == "cpu"

    def test_auto_picks_cuda_when_available(self, tmp_path):
        from infer import GR00TLocalInference
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            fake_model = tmp_path / "fake_model"
            fake_model.mkdir()
            inst = GR00TLocalInference(model_path=str(fake_model), device="auto")
            assert inst.device == "cuda"

    def test_explicit_device_preserved(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), device="cuda:0")
        assert inst.device == "cuda:0"


class TestQuantizeAutoDetection:
    """quantize='auto' → 从 model_path 推断 int4/int8/none."""

    def test_int4_path_detected(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "g1_gr00t_int4"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), quantize="auto")
        assert inst.quantize == "4bit"

    def test_int8_path_detected(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "g1_gr00t_int8"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), quantize="auto")
        assert inst.quantize == "8bit"

    def test_no_marker_means_full_precision(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "g1_gr00t_bf16"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), quantize="auto")
        assert inst.quantize == "none"

    def test_explicit_quantize_preserved(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), quantize="4bit")
        assert inst.quantize == "4bit"


class TestPolicyObservationBuild:
    """_build_policy_observation — 单步 obs → Gr00tPolicy batched 输入."""

    def test_video_dimension_expansion(self, infer_instance):
        """frame (H,W,3) → (1,1,H,W,3)."""
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        obs = self._make_mock_obs(frame)
        infer_instance._policy = MagicMock()
        infer_instance._policy.language_key = "annotation.human.task_description"
        infer_instance._policy.modality_configs = {
            "video": MagicMock(modality_keys=["front_view"]),
            "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        }
        out = infer_instance._build_policy_observation(obs)
        assert out["video"]["front_view"].shape == (1, 1, 224, 224, 3)
        assert out["video"]["front_view"].dtype == np.uint8

    def test_state_shape_btd(self, infer_instance):
        """state[key] 必须是 (B=1, T=1, D) float32."""
        obs = self._make_mock_obs()
        infer_instance._policy = MagicMock()
        infer_instance._policy.language_key = "annotation.human.task_description"
        infer_instance._policy.modality_configs = {
            "video": MagicMock(modality_keys=["front_view"]),
            "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        }
        out = infer_instance._build_policy_observation(obs)
        for k, v in out["state"].items():
            assert v.shape == (1, 1, v.shape[-1]), f"state[{k}] shape={v.shape}"
            assert v.dtype == np.float32

    def test_language_uses_self_instruction(self, infer_instance):
        """默认使用 self.instruction。"""
        infer_instance.instruction = "step back carefully"
        obs = self._make_mock_obs()
        infer_instance._policy = MagicMock()
        infer_instance._policy.language_key = "annotation.human.task_description"
        infer_instance._policy.modality_configs = {
            "video": MagicMock(modality_keys=["front_view"]),
            "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        }
        out = infer_instance._build_policy_observation(obs)
        assert out["language"]["annotation.human.task_description"] == [["step back carefully"]]

    def test_obs_instruction_overrides(self, infer_instance):
        """obs 里有 instruction 时优先 (支持数据回放每 step 不同指令)."""
        infer_instance.instruction = "default"
        obs = self._make_mock_obs()
        obs["annotation.language.action_text"] = "from_obs"
        infer_instance._policy = MagicMock()
        infer_instance._policy.language_key = "annotation.human.task_description"
        infer_instance._policy.modality_configs = {
            "video": MagicMock(modality_keys=["front_view"]),
            "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        }
        out = infer_instance._build_policy_observation(obs)
        assert out["language"]["annotation.human.task_description"] == [["from_obs"]]

    def test_video_fallback_when_missing(self, infer_instance):
        """obs 无 video → 0 帧填充."""
        obs = self._make_mock_obs(frame=None)
        infer_instance._policy = MagicMock()
        infer_instance._policy.language_key = "annotation.human.task_description"
        infer_instance._policy.modality_configs = {
            "video": MagicMock(modality_keys=["front_view"]),
            "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        }
        out = infer_instance._build_policy_observation(obs)
        assert out["video"]["front_view"].sum() == 0

    @staticmethod
    def _make_mock_obs(frame: np.ndarray | None = None) -> dict:
        obs = {
            "state.joint_pos":    np.zeros(29, dtype=np.float32),
            "state.joint_vel":    np.zeros(29, dtype=np.float32),
            "state.base_pos":     np.zeros(3,  dtype=np.float32),
            "state.base_quat":    np.zeros(4,  dtype=np.float32),
            "state.base_lin_vel": np.zeros(3,  dtype=np.float32),
            "state.base_ang_vel": np.zeros(3,  dtype=np.float32),
        }
        if frame is not None:
            obs["video.front_view"] = frame
        return obs


class TestActionQueueManagement:
    """Action chunking 队列."""

    def test_reset_clears_queue(self, infer_instance):
        infer_instance._action_queue = [np.zeros(29)]
        infer_instance._action_queue_start_pos = np.zeros(29)
        infer_instance.reset_action_queue()
        assert infer_instance._action_queue == []
        assert infer_instance._action_queue_start_pos is None


class TestExecutionHorizonBounds:
    """execution_horizon 必须在 [1, action_horizon] 范围内."""

    def test_clamp_high(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model), action_horizon=16, execution_horizon=100,
        )
        assert inst.execution_horizon == 16

    def test_clamp_low(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model), action_horizon=16, execution_horizon=0,
        )
        assert inst.execution_horizon == 1

    def test_normal_value_preserved(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model), action_horizon=16, execution_horizon=4,
        )
        assert inst.execution_horizon == 4


class TestRobotDefaults:
    """不同 robot 参数应加载不同的关节配置."""

    def test_g1_29_joints(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), robot="g1")
        assert inst.num_joints == 29
        assert inst.default_angles.shape == (29,)

    def test_go2_12_joints(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model), robot="go2")
        assert inst.num_joints == 12
        assert inst.default_angles.shape == (12,)

    def test_task_id_auto_selected(self, mock_torch, tmp_path):
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst_g1 = GR00TLocalInference(model_path=str(fake_model), robot="g1")
        inst_go2 = GR00TLocalInference(model_path=str(fake_model), robot="go2")
        assert "G1" in inst_g1.task_id
        assert "Go2" in inst_go2.task_id


class TestRelativeActionDecode:
    """_pop_cached_action — RELATIVE 模式 delta 累加."""

    def test_relative_adds_delta_to_current(self, mock_torch, tmp_path):
        """RELATIVE: 返回 current_pos + delta."""
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model))
        # 模拟 RELATIVE 模式
        ActionRep = MagicMock()
        ActionRep.RELATIVE = "RELATIVE"
        ActionRep.ABSOLUTE = "ABSOLUTE"
        inst._action_rep = ActionRep.RELATIVE

        current_pos = np.array([1.0] * 29, dtype=np.float32)
        delta = np.array([0.01] * 29, dtype=np.float32)
        inst._action_queue = [delta]

        obs = {"state.joint_pos": current_pos}
        result = inst._pop_cached_action(obs, ActionRep)

        np.testing.assert_array_almost_equal(result, current_pos + delta)

    def test_absolute_passes_through(self, mock_torch, tmp_path):
        """ABSOLUTE: delta 直接作为目标."""
        from infer import GR00TLocalInference
        fake_model = tmp_path / "fake_model"
        fake_model.mkdir()
        inst = GR00TLocalInference(model_path=str(fake_model))
        ActionRep = MagicMock()
        ActionRep.RELATIVE = "RELATIVE"
        ActionRep.ABSOLUTE = "ABSOLUTE"
        inst._action_rep = ActionRep.ABSOLUTE

        delta = np.array([0.5] * 29, dtype=np.float32)
        inst._action_queue = [delta]
        result = inst._pop_cached_action({"state.joint_pos": np.zeros(29)}, ActionRep)
        np.testing.assert_array_almost_equal(result, delta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])