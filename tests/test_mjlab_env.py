#!/usr/bin/env python3.12
"""测试 mjlab_env.py — 从 unitree_rl_mjlab env 拿 per-key obs + render."""
import sys, pytest, numpy as np
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mjlab_env import get_per_key_obs, render_frame


class TestGetPerKeyObs:
    """get_per_key_obs — 从 mjlab Entity.data 拿状态."""

    def test_returns_dict_with_expected_keys(self):
        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_rd = MagicMock()
        fake_rd.root_link_pos_w = np.array([[0, 0, 0.5]])
        fake_rd.root_link_quat_w = np.array([[1, 0, 0, 0]])
        fake_rd.root_link_lin_vel_w = np.array([[0, 0, 0]])
        fake_rd.root_link_ang_vel_w = np.array([[0, 0, 0]])
        fake_rd.joint_pos = np.array([[0.0] * 29])
        fake_rd.joint_vel = np.array([[0.0] * 29])
        fake_robot.data = fake_rd
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        for key in ["base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"]:
            assert key in obs, f"missing key: {key}"

    def test_failure_returns_empty_dict(self):
        """env API 不可用时, 返回空 dict 不报错."""
        fake_env = MagicMock()
        fake_env.unwrapped.scene.__getitem__.side_effect = KeyError("no robot")
        obs = get_per_key_obs(fake_env)
        assert obs == {}

    def test_handles_numpy_arrays(self):
        """确保拿到的值是 numpy array."""
        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_rd = MagicMock()
        fake_rd.root_link_pos_w = np.array([[1.0, 2.0, 3.0]])
        fake_rd.root_link_quat_w = np.array([[1, 0, 0, 0]], dtype=float)
        fake_rd.root_link_lin_vel_w = np.zeros((1, 3))
        fake_rd.root_link_ang_vel_w = np.zeros((1, 3))
        fake_rd.joint_pos = np.zeros((1, 12))
        fake_rd.joint_vel = np.zeros((1, 12))
        fake_robot.data = fake_rd
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        assert isinstance(obs["joint_pos"], np.ndarray)


class TestRenderFrame:
    """render_frame — 拿 env 渲染的 RGB 帧."""

    def test_none_env_returns_none(self):
        assert render_frame(None) is None

    def test_gym_style_render(self):
        """env.render() 返回 array 时直接用."""
        fake_env = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_env.render.return_value = frame
        # 用 cv2 或 PIL 调尺寸
        out = render_frame(fake_env, height=224, width=224)
        # cv2 resize 会被尝试, 如果都没有就保留原尺寸
        assert out is not None
        # 可能保持 100x100 (没装 cv2/PIL), 也可能被 resize 到 224x224
        assert out.shape[2] == 3  # always RGB

    def test_float_frame_normalized_to_uint8(self):
        """0~1 float 渲染 → 255 uint8."""
        fake_env = MagicMock()
        frame = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        fake_env.render.return_value = frame
        out = render_frame(fake_env, height=224, width=224)
        assert out.dtype == np.uint8

    def test_failure_returns_none(self):
        """env 渲染失败时不抛异常."""
        fake_env = MagicMock()
        fake_env.render.side_effect = RuntimeError("gl context lost")
        # 其他 fallback 也会失败
        fake_env.unwrapped.sim._data = MagicMock(side_effect=Exception("nope"))
        out = render_frame(fake_env)
        assert out is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])