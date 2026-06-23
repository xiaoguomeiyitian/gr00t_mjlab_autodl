#!/usr/bin/env python3.12
"""测试 convert_to_lerobot.py 中的纯函数:
   - _to_python_list
   - _get_state_vector
   - _get_action_vector
   - 4 种 action_mode 维度验证
   - STATE_KEYS_ORDER 完整性
"""
import sys, pytest, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convert_to_lerobot import (
    _to_python_list, _get_state_vector, _get_action_vector,
    STATE_KEYS_ORDER, DATA_PATH_TEMPLATE, VIDEO_PATH_TEMPLATE,
    LEROBOT_CODEBASE_VERSION,
)


class TestStateVectorAssembly:
    """_get_state_vector — 按 STATE_KEYS_ORDER 拼接 state 向量."""

    def test_full_obs_g1_state_dim(self):
        """完整 obs → G1 71 维."""
        obs = {
            "state.joint_pos":    np.zeros(29, dtype=np.float32),
            "state.joint_vel":    np.zeros(29, dtype=np.float32),
            "state.base_pos":     np.zeros(3,  dtype=np.float32),
            "state.base_quat":    np.zeros(4,  dtype=np.float32),
            "state.base_lin_vel": np.zeros(3,  dtype=np.float32),
            "state.base_ang_vel": np.zeros(3,  dtype=np.float32),
        }
        v = _get_state_vector(obs, num_joints=29)
        assert v.shape == (71,)
        assert v.dtype == np.float32

    def test_full_obs_go2_state_dim(self):
        """完整 obs → Go2 37 维."""
        obs = {
            "state.joint_pos":    np.zeros(12, dtype=np.float32),
            "state.joint_vel":    np.zeros(12, dtype=np.float32),
            "state.base_pos":     np.zeros(3,  dtype=np.float32),
            "state.base_quat":    np.zeros(4,  dtype=np.float32),
            "state.base_lin_vel": np.zeros(3,  dtype=np.float32),
            "state.base_ang_vel": np.zeros(3,  dtype=np.float32),
        }
        v = _get_state_vector(obs, num_joints=12)
        assert v.shape == (37,)

    def test_missing_fields_zero_filled(self):
        """缺失的 state 字段用 0 填充, 不报错."""
        obs = {"state.joint_pos": np.ones(29, dtype=np.float32)}
        v = _get_state_vector(obs, num_joints=29)
        assert v.shape == (71,)
        assert v[0] == 1.0  # joint_pos 第 0 个
        assert v[29] == 0.0  # joint_vel 缺失 → 0

    def test_wrong_dim_zero_filled(self):
        """维度不对的字段 → 警告 + 0 填充."""
        obs = {"state.joint_pos": np.ones(5, dtype=np.float32)}  # 错误维度
        v = _get_state_vector(obs, num_joints=29)
        assert v.shape == (71,)
        assert v[0] == 0.0  # 错误的被填充为 0

    def test_state_keys_order_is_six(self):
        assert len(STATE_KEYS_ORDER) == 6
        assert "joint_pos" in STATE_KEYS_ORDER
        assert "joint_vel" in STATE_KEYS_ORDER
        assert "base_quat" in STATE_KEYS_ORDER


class TestActionVectorAbsolute:
    """absolute 模式 — joint_position_target."""

    def test_correct_dim_g1(self):
        act = {"action.joint_position_target": np.ones(29, dtype=np.float32)}
        v = _get_action_vector(act, "absolute", num_joints=29)
        assert v.shape == (29,)
        assert v[0] == 1.0

    def test_correct_dim_go2(self):
        act = {"action.joint_position_target": np.ones(12, dtype=np.float32)}
        v = _get_action_vector(act, "absolute", num_joints=12)
        assert v.shape == (12,)

    def test_fallback_to_target_joint_pos(self):
        """兼容旧字段名 target_joint_pos."""
        act = {"target_joint_pos": np.ones(29, dtype=np.float32) * 0.5}
        v = _get_action_vector(act, "absolute", num_joints=29)
        assert v[0] == 0.5

    def test_missing_zero_filled(self):
        v = _get_action_vector({}, "absolute", num_joints=29)
        assert v.shape == (29,)
        assert v.sum() == 0.0


class TestActionVectorDelta:
    """delta 模式 — joint_position_delta."""

    def test_correct_dim_g1(self):
        act = {"action.joint_position_delta": np.ones(29, dtype=np.float32) * 0.01}
        v = _get_action_vector(act, "delta", num_joints=29)
        assert v.shape == (29,)
        assert v[0] == 0.01

    def test_missing_zero_filled(self):
        """delta 字段缺失 → 全 0 (即"不动作"), 不报错."""
        v = _get_action_vector({}, "delta", num_joints=12)
        assert v.shape == (12,)
        assert v.sum() == 0.0

    def test_wrong_dim_zero_filled(self):
        act = {"action.joint_position_delta": np.ones(5, dtype=np.float32)}
        v = _get_action_vector(act, "delta", num_joints=29)
        assert v.shape == (29,)
        assert v.sum() == 0.0


class TestActionVectorRelativeEef:
    """relative_eef 模式 — ee_pose_delta (7 维 pos+quat)."""

    def test_correct_dim(self):
        act = {"action.ee_pose_delta": np.array([0.1, 0, 0, 1, 0, 0, 0], dtype=np.float32)}
        v = _get_action_vector(act, "relative_eef", num_joints=29)
        assert v.shape == (7,)
        assert v[0] == 0.1
        assert v[3] == 1.0  # quat w

    def test_missing_zero_filled(self):
        v = _get_action_vector({}, "relative_eef", num_joints=29)
        assert v.shape == (7,)
        assert v.sum() == 0.0


class TestActionVectorErrorHandling:
    """未知 action_mode → ValueError."""

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="未知 action_mode"):
            _get_action_vector({}, "weird_mode", num_joints=29)


class TestActionModeKeyConsistency:
    """P2 修复: action_mode 语义必须与 modality.json action key 一致.

    这是 convert_to_lerobot.py 实际写入 modality.json 后做的 self-check,
    这里直接验证 expected_action_keys 字典.
    """
    @pytest.fixture
    def expected_keys(self):
        """从 convert_to_lerobot 源码提取 expected_action_keys 字典."""
        import ast
        from pathlib import Path as _P
        text = (_P(__file__).resolve().parent.parent / "src" / "convert_to_lerobot.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "expected_action_keys":
                        return ast.literal_eval(node.value)
        pytest.fail("expected_action_keys not found in source")

    def test_absolute_has_target(self, expected_keys):
        assert "joint_position_target" in expected_keys["absolute"]

    def test_delta_has_delta(self, expected_keys):
        assert "joint_position_delta" in expected_keys["delta"]

    def test_relative_eef_has_ee_pose(self, expected_keys):
        assert "ee_pose_delta" in expected_keys["relative_eef"]


class TestLeRobotConstants:
    """LeRobot v2 元数据常量."""

    def test_lerobot_version_v2(self):
        assert LEROBOT_CODEBASE_VERSION == "v2.1"

    def test_data_path_template(self):
        assert "{episode_chunk:03d}" in DATA_PATH_TEMPLATE
        assert "{episode_index:06d}" in DATA_PATH_TEMPLATE
        assert DATA_PATH_TEMPLATE.endswith(".parquet")

    def test_video_path_template(self):
        assert "{video_key}" in VIDEO_PATH_TEMPLATE
        assert VIDEO_PATH_TEMPLATE.endswith(".mp4")


class TestToPythonList:
    """_to_python_list — numpy/tensor → python list."""

    def test_none_passthrough(self):
        assert _to_python_list(None) is None

    def test_numpy_1d(self):
        assert _to_python_list(np.array([1, 2, 3])) == [1, 2, 3]

    def test_list_passthrough(self):
        assert _to_python_list([1, 2, 3]) == [1, 2, 3]

    def test_nested_list_of_arrays(self):
        result = _to_python_list([np.array([1, 2]), np.array([3])])
        assert result == [[1, 2], [3]]

    def test_scalar(self):
        assert _to_python_list(42) == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])