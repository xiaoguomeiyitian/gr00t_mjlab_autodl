#!/usr/bin/env python3.12
"""测试 mjlab_env.py — get_per_key_obs 和 render_frame 边界条件.

覆盖:
  - get_per_key_obs: 空 scene、部分数据缺失、无 data 属性
  - render_frame: uint8/float 帧、尺寸不匹配、None env
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestGetPerKeyObsExtended:
    """get_per_key_obs — 更多边界条件."""

    def test_empty_scene(self):
        """scene 为空 dict → 返回空 dict."""
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_env.unwrapped.scene = {}

        obs = get_per_key_obs(fake_env)
        assert obs == {}

    def test_robot_without_data_attr(self):
        """robot 没有 data 属性 → 返回空 dict."""
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_robot = MagicMock(spec=[])  # 无属性
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        assert obs == {}

    def test_partial_data(self):
        """部分数据缺失 → 只返回可用的 key."""
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_rd = MagicMock()
        fake_rd.root_link_pos_w = np.array([[0, 0, 0.5]])
        # 其他属性缺失 — 通过 spec 限制
        del fake_rd.root_link_quat_w
        del fake_rd.root_link_lin_vel_w
        del fake_rd.root_link_ang_vel_w
        del fake_rd.joint_pos
        del fake_rd.joint_vel
        fake_robot.data = fake_rd
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        assert "base_pos" in obs
        # 缺失的 key 不应出现
        assert "base_quat" not in obs

    def test_batch_dim_preserved(self):
        """batch 维 (1, ...) 应保留."""
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_robot = MagicMock()
        fake_rd = MagicMock()
        fake_rd.root_link_pos_w = np.array([[0, 0, 0.5]])  # (1, 3)
        fake_rd.root_link_quat_w = np.array([[1, 0, 0, 0]])
        fake_rd.root_link_lin_vel_w = np.zeros((1, 3))
        fake_rd.root_link_ang_vel_w = np.zeros((1, 3))
        fake_rd.joint_pos = np.zeros((1, 29))
        fake_rd.joint_vel = np.zeros((1, 29))
        fake_robot.data = fake_rd
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        # 应保留 batch 维
        assert obs["base_pos"].ndim >= 1

    def test_scene_key_error(self):
        """scene 中无 robot key → 返回空 dict."""
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_env.unwrapped.scene = {"not_robot": MagicMock()}

        obs = get_per_key_obs(fake_env)
        assert obs == {}

    def test_data_access_exception(self):
        """data 属性访问异常 → 返回空 dict."""
        from unittest.mock import PropertyMock
        from mjlab_env import get_per_key_obs

        fake_env = MagicMock()
        fake_robot = MagicMock()
        # Make robot.data raise an exception on access
        type(fake_robot).data = PropertyMock(side_effect=RuntimeError("no data"))
        fake_env.unwrapped.scene = {"robot": fake_robot}

        obs = get_per_key_obs(fake_env)
        assert obs == {}


class TestRenderFrameExtended:
    """render_frame — 更多场景."""

    def test_uint8_frame_passthrough(self):
        """uint8 帧直接返回."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_env.render.return_value = frame

        out = render_frame(fake_env, height=100, width=100)
        assert out is not None
        assert out.dtype == np.uint8

    def test_float_frame_0_to_1(self):
        """float [0, 1] → uint8 [0, 255]."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        frame = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        fake_env.render.return_value = frame

        out = render_frame(fake_env, height=50, width=50)
        assert out.dtype == np.uint8
        # 0.5 * 255 ≈ 127 or 128
        assert 120 <= out[0, 0, 0] <= 135

    def test_float_frame_0_to_255(self):
        """float [0, 255] → uint8 直接 clip."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        frame = np.ones((50, 50, 3), dtype=np.float64) * 200.0
        fake_env.render.return_value = frame

        out = render_frame(fake_env, height=50, width=50)
        assert out.dtype == np.uint8

    def test_wrong_size_frame(self):
        """尺寸不匹配 → 尝试 resize."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        fake_env.render.return_value = frame

        out = render_frame(fake_env, height=224, width=224)
        # 如果 cv2/PIL 不可用, 可能保持原尺寸
        assert out is not None
        assert out.shape[2] == 3

    def test_none_frame_from_env(self):
        """env.render() 返回 None → fallback 或 None."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        fake_env.render.return_value = None
        # fallback 也会失败
        fake_env.unwrapped.sim.mj_data = MagicMock(side_effect=Exception("no sim"))

        out = render_frame(fake_env)
        # 可能返回 None 或 fallback 结果
        # 关键是不要崩溃

    def test_env_without_render(self):
        """env 没有 render 方法 → 不崩溃."""
        from mjlab_env import render_frame

        fake_env = MagicMock(spec=[])  # 无 render 方法

        out = render_frame(fake_env)
        assert out is None

    def test_2d_frame_handled(self):
        """2D 帧 (灰度) → 不崩溃."""
        from mjlab_env import render_frame

        fake_env = MagicMock()
        frame = np.zeros((50, 50), dtype=np.uint8)
        fake_env.render.return_value = frame

        out = render_frame(fake_env, height=50, width=50)
        # 可能返回 None 或处理后的结果
        # 关键是不要崩溃


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
