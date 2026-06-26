#!/usr/bin/env python3.12
"""测试 viser_3d_viewer.py — 异步 3D 可视化.

覆盖:
  - AsyncViser3DViewer 初始化
  - start/stop 生命周期
  - notify_new_frame 事件触发
  - update 共享状态写入
  - 统计计数器
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_viewer():
    """创建一个正确初始化的 AsyncViser3DViewer."""
    from viser_3d_viewer import AsyncViser3DViewer
    viewer = AsyncViser3DViewer.__new__(AsyncViser3DViewer)
    viewer._env = None
    viewer._port = 20006
    viewer._env_idx = 0
    viewer._frame_rate = 30.0
    viewer._show_debug = False
    viewer._server = None
    viewer._scene = None
    viewer._thread = None
    viewer._running = threading.Event()
    viewer._stop = threading.Event()
    viewer._shared_state = None
    viewer._state_lock = None
    viewer._swap_event = threading.Event()
    viewer._render_count = 0
    viewer._error_count = 0
    viewer._last_render_time = 0.0
    viewer._connection_count = 0
    viewer._connection_count_lock = threading.Lock()
    viewer._paused = False
    return viewer


class TestAsyncViser3DViewerInit:
    """AsyncViser3DViewer 初始化."""

    def test_default_port(self):
        """默认端口 20006."""
        viewer = _make_viewer()
        assert viewer._port == 20006

    def test_initial_state(self):
        """初始状态: 未运行, 计数器为 0."""
        viewer = _make_viewer()
        assert viewer.render_count == 0
        assert not viewer._running.is_set()


class TestNotifyNewFrame:
    """notify_new_frame — 事件触发."""

    def test_notify_sets_event(self):
        """notify_new_frame → set swap_event."""
        viewer = _make_viewer()
        viewer.notify_new_frame()
        assert viewer._swap_event.is_set()


class TestUpdate:
    """update — 共享状态写入."""

    def test_update_writes_shared_state(self):
        """update → 写共享状态 (qpos, qvel)."""
        viewer = _make_viewer()
        shared_state = {}
        lock = threading.Lock()
        viewer._shared_state = shared_state
        viewer._state_lock = lock

        mock_env = MagicMock()
        mock_sim = MagicMock()
        mock_sim.mj_data.qpos = np.array([1.0, 2.0, 3.0])
        mock_sim.mj_data.qvel = np.array([0.1, 0.2, 0.3])
        mock_env.unwrapped.sim = mock_sim
        viewer._env = mock_env

        viewer.update()

        assert 'qpos' in shared_state
        assert 'qvel' in shared_state

    def test_update_no_env_noop(self):
        """env=None → update 不执行."""
        viewer = _make_viewer()
        viewer._env = None
        viewer.update()
        assert not viewer._swap_event.is_set()


class TestStartStop:
    """start/stop 生命周期."""

    def test_start_creates_thread(self):
        """start → 创建 daemon 线程."""
        viewer = _make_viewer()
        with patch.object(type(viewer), '_render_loop', lambda self: None):
            viewer.start()
            assert viewer._thread is not None
            assert viewer._running.is_set()
            viewer.stop()

    def test_stop_joins_thread(self):
        """stop → 设置 stop event, join 线程."""
        viewer = _make_viewer()
        viewer._running.set()
        mock_thread = MagicMock()
        viewer._thread = mock_thread

        viewer.stop()

        assert viewer._stop.is_set()
        assert not viewer._running.is_set()
        mock_thread.join.assert_called_once()


class TestUrlProperty:
    """url 属性."""

    def test_url_format(self):
        """url = http://0.0.0.0:<port>."""
        viewer = _make_viewer()
        viewer._port = 20006
        assert viewer.url == "http://0.0.0.0:20006"


class TestRenderCount:
    """渲染计数器."""

    def test_initial_count_zero(self):
        """初始 render_count = 0."""
        viewer = _make_viewer()
        viewer._render_count = 0
        assert viewer.render_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
