#!/usr/bin/env python3.12
"""测试 infer.py 核心推理逻辑 — get_action / _merge_action_keys / _pop_cached_action.

覆盖:
  - get_action: 单 action key / 多 action key 合并
  - _merge_action_keys: 按 state_key 映射到关节
  - _pop_cached_action: ABSOLUTE / RELATIVE 模式
  - action chunking: execution_horizon > 1 时的队列行为
  - _apply_action_to_joints: RELATIVE / ABSOLUTE 映射
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_torch():
    """Mock torch 模块, device='auto' 走 cpu."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": fake_torch}):
        yield fake_torch


@pytest.fixture
def infer_g1(mock_torch, tmp_path):
    """构造 GR00TLocalInference 实例 (G1, 29 joints)."""
    from infer import GR00TLocalInference
    fake_model = tmp_path / "g1_model"
    fake_model.mkdir()
    return GR00TLocalInference(
        model_path=str(fake_model),
        robot="g1",
        quantize="none",
        device="cpu",
        instruction="walk forward",
        action_horizon=16,
        execution_horizon=1,
    )


@pytest.fixture
def infer_go2(mock_torch, tmp_path):
    """构造 GR00TLocalInference 实例 (Go2, 12 joints)."""
    from infer import GR00TLocalInference
    fake_model = tmp_path / "go2_model"
    fake_model.mkdir()
    return GR00TLocalInference(
        model_path=str(fake_model),
        robot="go2",
        quantize="none",
        device="cpu",
        instruction="trot",
        action_horizon=16,
        execution_horizon=1,
    )


# ── Helper: 构造 mock policy ──────────────────────────────────────────────


def _make_mock_policy(action_keys, action_rep, state_key=None):
    """构造一个 mock policy 对象."""
    mock_policy = MagicMock()
    mock_policy.modality_configs = {
        "action": MagicMock(
            modality_keys=action_keys,
            action_configs=[MagicMock(rep=action_rep, state_key=state_key or action_keys[0])],
        ),
        "state": MagicMock(modality_keys=["joint_pos", "joint_vel"]),
        "video": MagicMock(modality_keys=["front_view"]),
    }
    mock_policy.language_key = "annotation.human.task_description"
    mock_policy.check_observation = MagicMock()
    return mock_policy


def _make_g1_obs():
    """构造 G1 观测."""
    return {
        "state.joint_pos": np.zeros(29, dtype=np.float32),
        "state.joint_vel": np.zeros(29, dtype=np.float32),
    }


# ── Test get_action: single action key ────────────────────────────────────


class TestGetActionSingleKey:
    """get_action — 单 action key 场景."""

    def test_single_key_output_shape(self, infer_g1):
        """单 action key → 输出 (num_joints,) float32."""
        mock_policy = _make_mock_policy(["joint_position_delta"], "RELATIVE")
        # _get_action 返回 (action_dict, info)
        mock_policy._get_action.return_value = (
            {"joint_position_delta": np.zeros((1, 16, 29), dtype=np.float32)},
            {},
        )
        infer_g1._policy = mock_policy
        infer_g1._action_rep = "RELATIVE"
        infer_g1._action_keys = ["joint_position_delta"]

        action = infer_g1.get_action(_make_g1_obs())
        assert action.shape == (29,)
        assert action.dtype == np.float32

    def test_single_key_caches_to_queue(self, infer_g1):
        """get_action 后: execution_horizon=1 → 缓存 1 步并立即消费, 队列空."""
        mock_policy = _make_mock_policy(["joint_position_target"], "ABSOLUTE")
        mock_policy._get_action.return_value = (
            {"joint_position_target": np.ones((1, 16, 29), dtype=np.float32) * 0.5},
            {},
        )
        infer_g1._policy = mock_policy
        infer_g1._action_rep = "ABSOLUTE"
        infer_g1._action_keys = ["joint_position_target"]

        action = infer_g1.get_action(_make_g1_obs())

        # execution_horizon=1 → 缓存 1 步并立即消费, 队列空
        assert len(infer_g1._action_queue) == 0
        # 但返回了 action
        assert action.shape == (29,)
        np.testing.assert_array_almost_equal(action, 0.5)

    def test_execution_horizon_gt_1(self, infer_g1):
        """execution_horizon > 1 → 缓存多步, 立即消费 1 步, 队列剩 n-1 步."""
        infer_g1.execution_horizon = 4
        mock_policy = _make_mock_policy(["joint_position_target"], "ABSOLUTE")
        mock_policy._get_action.return_value = (
            {"joint_position_target": np.ones((1, 16, 29), dtype=np.float32)},
            {},
        )
        infer_g1._policy = mock_policy
        infer_g1._action_rep = "ABSOLUTE"
        infer_g1._action_keys = ["joint_position_target"]

        infer_g1.get_action(_make_g1_obs())

        # execution_horizon=4 → 缓存 4 步, 立即消费 1 步, 队列剩 3 步
        assert len(infer_g1._action_queue) == 3


# ── Test _merge_action_keys ───────────────────────────────────────────────


class TestMergeActionKeys:
    """_merge_action_keys — 多 action key 合并."""

    def test_single_key_direct(self, infer_g1):
        """单 key → 直接返回 (T, D)."""
        action_data = np.ones((1, 16, 29), dtype=np.float32) * 0.3
        infer_g1._action_keys = ["joint_position_delta"]
        result = infer_g1._merge_action_keys({"joint_position_delta": action_data})
        assert result.shape == (16, 29)
        np.testing.assert_array_almost_equal(result, 0.3)

    def test_multiple_keys_with_state_key(self, infer_g1):
        """多 key 有 state_key → 按 _part_ranges 映射到关节."""
        # Mock action_configs
        left_leg_ac = MagicMock()
        left_leg_ac.state_key = "left_leg"
        left_leg_ac.rep = "RELATIVE"
        right_leg_ac = MagicMock()
        right_leg_ac.state_key = "right_leg"
        right_leg_ac.rep = "ABSOLUTE"

        infer_g1._policy = MagicMock()
        infer_g1._policy.modality_configs = {
            "action": MagicMock(
                action_configs=[left_leg_ac, right_leg_ac],
            ),
        }
        infer_g1._action_keys = ["left_leg", "right_leg"]
        infer_g1._action_rep = "ABSOLUTE"

        action_dict = {
            "left_leg": np.ones((1, 16, 6), dtype=np.float32) * 0.1,
            "right_leg": np.ones((1, 16, 6), dtype=np.float32) * 0.2,
        }

        result = infer_g1._merge_action_keys(action_dict)
        assert result.shape == (16, 29)  # G1 num_joints


# ── Test _pop_cached_action ───────────────────────────────────────────────


class TestPopCachedAction:
    """_pop_cached_action — 队列消费."""

    def test_absolute_returns_raw(self, infer_g1):
        """ABSOLUTE 模式: 直接返回缓存值."""
        infer_g1._action_rep = "ABSOLUTE"
        cached = np.ones(29, dtype=np.float32) * 0.5
        infer_g1._action_queue = [cached.copy()]

        obs = _make_g1_obs()

        # Mock ActionRepresentation
        mock_rep = MagicMock()
        mock_rep.ABSOLUTE = "ABSOLUTE"
        mock_rep.RELATIVE = "RELATIVE"

        result = infer_g1._pop_cached_action(obs, mock_rep)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_relative_adds_current_pos(self, infer_g1):
        """RELATIVE 模式: current_pos + delta."""
        infer_g1._action_rep = "RELATIVE"
        delta = np.ones(29, dtype=np.float32) * 0.1
        infer_g1._action_queue = [delta.copy()]

        current_pos = np.ones(29, dtype=np.float32) * 0.3
        obs = {"state.joint_pos": current_pos}

        mock_rep = MagicMock()
        mock_rep.ABSOLUTE = "ABSOLUTE"
        mock_rep.RELATIVE = "RELATIVE"

        result = infer_g1._pop_cached_action(obs, mock_rep)
        np.testing.assert_array_almost_equal(result, 0.4)  # 0.3 + 0.1

    def test_queue_consumed_after_pop(self, infer_g1):
        """pop 后队列长度减 1."""
        infer_g1._action_rep = "ABSOLUTE"
        infer_g1._action_queue = [
            np.zeros(29, dtype=np.float32),
            np.ones(29, dtype=np.float32),
        ]

        obs = _make_g1_obs()
        mock_rep = MagicMock()
        mock_rep.ABSOLUTE = "ABSOLUTE"
        mock_rep.RELATIVE = "RELATIVE"

        infer_g1._pop_cached_action(obs, mock_rep)
        assert len(infer_g1._action_queue) == 1

    def test_empty_queue_raises(self, infer_g1):
        """空队列 → RuntimeError."""
        infer_g1._action_queue = []
        obs = _make_g1_obs()
        mock_rep = MagicMock()

        with pytest.raises(RuntimeError, match="action queue empty"):
            infer_g1._pop_cached_action(obs, mock_rep)


# ── Test _apply_action_to_joints ──────────────────────────────────────────


class TestApplyActionToJoints:
    """_apply_action_to_joints — state_key 到关节索引的映射."""

    def test_left_leg_mapping(self, infer_g1):
        """left_leg state_key → 映射到关节 0-5."""
        output = np.zeros((16, 29), dtype=np.float32)
        action_data = np.ones((16, 6), dtype=np.float32) * 0.5

        mock_ac = MagicMock()
        mock_ac.state_key = "left_leg"
        mock_ac.rep = "ABSOLUTE"

        infer_g1._apply_action_to_joints(output, action_data, mock_ac)

        # 关节 0-5 应被设为 0.5
        np.testing.assert_array_almost_equal(output[:, 0:6], 0.5)
        # 关节 6-28 应仍为 0
        np.testing.assert_array_almost_equal(output[:, 6:], 0.0)

    def test_right_leg_mapping(self, infer_g1):
        """right_leg state_key → 映射到关节 6-11."""
        output = np.zeros((16, 29), dtype=np.float32)
        action_data = np.ones((16, 6), dtype=np.float32) * 0.3

        mock_ac = MagicMock()
        mock_ac.state_key = "right_leg"
        mock_ac.rep = "ABSOLUTE"

        infer_g1._apply_action_to_joints(output, action_data, mock_ac)

        np.testing.assert_array_almost_equal(output[:, 6:12], 0.3)
        # 其他关节不受影响
        np.testing.assert_array_almost_equal(output[:, 0:6], 0.0)

    def test_waist_mapping(self, infer_g1):
        """waist state_key → 映射到关节 12-14."""
        output = np.zeros((16, 29), dtype=np.float32)
        action_data = np.ones((16, 3), dtype=np.float32) * 0.2

        mock_ac = MagicMock()
        mock_ac.state_key = "waist"
        mock_ac.rep = "ABSOLUTE"

        infer_g1._apply_action_to_joints(output, action_data, mock_ac)

        np.testing.assert_array_almost_equal(output[:, 12:15], 0.2)

    def test_unknown_state_key_noop(self, infer_g1):
        """未知 state_key → 不修改 output."""
        output = np.zeros((16, 29), dtype=np.float32)
        action_data = np.ones((16, 6), dtype=np.float32) * 0.5

        mock_ac = MagicMock()
        mock_ac.state_key = "unknown_part"
        mock_ac.rep = "ABSOLUTE"

        infer_g1._apply_action_to_joints(output, action_data, mock_ac)

        # output 应全为 0
        np.testing.assert_array_almost_equal(output, 0.0)

    def test_relative_mode_uses_action_directly(self, infer_g1):
        """RELATIVE 模式: action_data 直接作为目标 (简化实现)."""
        output = np.zeros((16, 29), dtype=np.float32)
        action_data = np.ones((16, 6), dtype=np.float32) * 0.1

        mock_ac = MagicMock()
        mock_ac.state_key = "left_leg"
        mock_ac.rep = "RELATIVE"

        infer_g1._apply_action_to_joints(output, action_data, mock_ac)

        # 在简化实现中, RELATIVE 直接使用 action_data 作为目标
        np.testing.assert_array_almost_equal(output[:, 0:6], 0.1)


# ── Test Go2 inference ────────────────────────────────────────────────────


class TestGo2Inference:
    """Go2 机器人推理测试."""

    def test_go2_num_joints(self, infer_go2):
        """Go2 应有 12 joints."""
        assert infer_go2.num_joints == 12

    def test_go2_action_shape(self, infer_go2):
        """Go2 get_action → (12,) 输出."""
        mock_policy = MagicMock()
        mock_policy.modality_configs = {
            "action": MagicMock(
                modality_keys=["joint_position_target"],
                action_configs=[MagicMock(rep="ABSOLUTE", state_key="joint_position_target")],
            ),
            "state": MagicMock(modality_keys=["joint_pos"]),
            "video": MagicMock(modality_keys=["front_view"]),
        }
        mock_policy.language_key = "annotation.human.task_description"
        mock_policy._get_action.return_value = (
            {"joint_position_target": np.zeros((1, 16, 12), dtype=np.float32)},
            {},
        )
        mock_policy.check_observation = MagicMock()

        infer_go2._policy = mock_policy
        infer_go2._action_rep = "ABSOLUTE"
        infer_go2._action_keys = ["joint_position_target"]

        obs = {"state.joint_pos": np.zeros(12, dtype=np.float32)}
        action = infer_go2.get_action(obs)
        assert action.shape == (12,)


# ── Test edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """边界条件和错误处理."""

    def test_execution_horizon_clipped_to_action_horizon(self, mock_torch, tmp_path):
        """execution_horizon > action_horizon → 自动裁剪."""
        from infer import GR00TLocalInference
        fake_model = tmp_path / "model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model),
            action_horizon=8,
            execution_horizon=20,  # 超过 action_horizon
        )
        assert inst.execution_horizon == 8

    def test_execution_horizon_minimum_1(self, mock_torch, tmp_path):
        """execution_horizon 最小为 1."""
        from infer import GR00TLocalInference
        fake_model = tmp_path / "model"
        fake_model.mkdir()
        inst = GR00TLocalInference(
            model_path=str(fake_model),
            action_horizon=16,
            execution_horizon=0,  # 应被修正为 1
        )
        assert inst.execution_horizon == 1

    def test_reset_action_queue_clears_all(self, infer_g1):
        """reset_action_queue 清空队列和 start_pos."""
        infer_g1._action_queue = [np.zeros(29), np.ones(29)]
        infer_g1._action_queue_start_pos = np.zeros(29)

        infer_g1.reset_action_queue()

        assert infer_g1._action_queue == []
        assert infer_g1._action_queue_start_pos is None

    def test_go2_uses_correct_config(self, infer_go2):
        """Go2 实例从 go2_config 读取关节信息."""
        assert infer_go2.num_joints == 12
        assert len(infer_go2.default_angles) == 12

    def test_g1_29dof_default_angles(self, infer_g1):
        """G1 默认 29 joints."""
        assert infer_g1.num_joints == 29
        assert len(infer_g1.default_angles) == 29


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
