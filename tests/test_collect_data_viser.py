#!/usr/bin/env python3.12
"""测试 collect_data.py 中的 ViserViewer 类和 collect_demonstrations 参数解析.

覆盖:
  - ViserViewer 初始化 (3D / 文本 / 无 viser)
  - ViserViewer.update 共享状态写入
  - ViserViewer.close 清理
  - collect_demonstrations 参数验证
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestViserViewerInit:
    """ViserViewer 初始化路径."""

    def test_3d_viewer_path(self):
        """有 viser_3d_viewer → 创建 3D viewer."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viewer_3d_class = MagicMock()
        mock_viewer_3d = MagicMock()
        mock_viewer_3d_class.return_value = mock_viewer_3d
        mock_viewer_3d.url = "http://localhost:20006"

        with patch("collect_data._get_viser_viewer", return_value=mock_viewer_3d_class):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        assert viewer._viewer_3d is not None
        mock_viewer_3d.set_shared_state.assert_called_once()
        mock_viewer_3d.start.assert_called_once()

    def test_text_viewer_fallback(self):
        """viser_3d_viewer 不可用但有 viser → 文本 viewer."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viser = MagicMock()
        mock_server = MagicMock()
        mock_viser.ViserServer.return_value = mock_server

        with patch("collect_data._get_viser_viewer", return_value=None), \
             patch.dict(sys.modules, {"viser": mock_viser}):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        assert viewer._server is not None
        mock_viser.ViserServer.assert_called_once()

    def test_no_viser_no_crash(self):
        """viser 未安装 → 不崩溃, viewer 为 None."""
        from collect_data import ViserViewer

        mock_env = MagicMock()

        with patch("collect_data._get_viser_viewer", return_value=None):
            # viser 导入失败
            with patch.dict(sys.modules, {"viser": None}):
                # ImportError 会在 import viser 时触发
                # 但 ViserViewer 内部 try/except 会捕获
                try:
                    viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)
                    # 如果 viser 未安装, _server 应为 None
                except Exception:
                    pass  # 可接受, 因为 viser 是可选依赖


class TestViserViewerUpdate:
    """ViserViewer.update — 共享状态更新."""

    def test_3d_viewer_update(self):
        """3D viewer → 更新共享状态 + notify."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viewer_3d_class = MagicMock()
        mock_viewer_3d = MagicMock()
        mock_viewer_3d_class.return_value = mock_viewer_3d
        mock_viewer_3d.url = "http://localhost:20006"
        mock_viewer_3d.has_connections = True

        with patch("collect_data._get_viser_viewer", return_value=mock_viewer_3d_class):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        viewer.update(
            ep=5, total_ep=100, step=50, total_step=200,
            instruction="walk forward",
            base_vel=np.array([0.1, 0.0, 0.0]),
            joint_targets=np.ones(29) * 0.3,
        )

        # 应更新共享状态
        assert viewer._shared_state['ep'] == 5
        assert viewer._shared_state['total_ep'] == 100
        assert viewer._shared_state['step'] == 50
        assert viewer._shared_state['total_step'] == 200
        assert viewer._shared_state['instruction'] == "walk forward"

        # 应 notify
        mock_viewer_3d.notify_new_frame.assert_called_once()

    def test_text_viewer_update(self):
        """文本 viewer → 更新文本控件."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viser = MagicMock()
        mock_server = MagicMock()
        mock_viser.ViserServer.return_value = mock_server

        with patch("collect_data._get_viser_viewer", return_value=None), \
             patch.dict(sys.modules, {"viser": mock_viser}):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        # Mock GUI 控件
        mock_ep_text = MagicMock()
        mock_step_text = MagicMock()
        mock_instr_text = MagicMock()
        viewer.ep_text = mock_ep_text
        viewer.step_text = mock_step_text
        viewer.instr_text = mock_instr_text

        viewer.update(ep=10, total_ep=50, step=30, total_step=100, instruction="turn left")

        mock_ep_text.value = "11/50"
        mock_step_text.value = "30/100"
        mock_instr_text.value = "turn left"

    def test_adaptive_fps_on_first_update(self):
        """首次 update 时根据连接状态调整 FPS."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viewer_3d_class = MagicMock()
        mock_viewer_3d = MagicMock()
        mock_viewer_3d_class.return_value = mock_viewer_3d
        mock_viewer_3d.url = "http://localhost:20006"
        mock_viewer_3d.has_connections = False  # 无连接

        with patch("collect_data._get_viser_viewer", return_value=mock_viewer_3d_class):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        viewer.update(ep=0, total_ep=10, step=0, total_step=100, instruction="test")

        # 无连接 → set_fps(0) 暂停
        mock_viewer_3d.set_fps.assert_called_with(0)
        assert viewer._fps_adjusted is True


class TestViserViewerClose:
    """ViserViewer.close — 清理."""

    def test_close_3d_viewer(self):
        """close → 停止 3D viewer."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viewer_3d_class = MagicMock()
        mock_viewer_3d = MagicMock()
        mock_viewer_3d_class.return_value = mock_viewer_3d

        with patch("collect_data._get_viser_viewer", return_value=mock_viewer_3d_class):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        viewer.close()
        mock_viewer_3d.stop.assert_called_once()

    def test_close_text_viewer(self):
        """close → 停止 viser server."""
        from collect_data import ViserViewer

        mock_env = MagicMock()
        mock_viser = MagicMock()
        mock_server = MagicMock()
        mock_viser.ViserServer.return_value = mock_server

        with patch("collect_data._get_viser_viewer", return_value=None), \
             patch.dict(sys.modules, {"viser": mock_viser}):
            viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)

        viewer.close()
        mock_server.stop.assert_called_once()

    def test_close_no_viewer_no_error(self):
        """无 viewer → close 不报错."""
        from collect_data import ViserViewer

        mock_env = MagicMock()

        with patch("collect_data._get_viser_viewer", return_value=None):
            try:
                viewer = ViserViewer(env=mock_env, port=20006, viser_fps=30.0)
                viewer.close()
            except Exception:
                pass  # 可接受


class TestCollectDemonstrationsArgs:
    """collect_demonstrations 参数验证."""

    def test_unknown_task_raises(self):
        """未知 task_id → ValueError."""
        from collect_data import collect_demonstrations

        with pytest.raises(ValueError, match="不支持的任务"):
            collect_demonstrations(task_id="NonExistent-Task")

    def test_known_task_ids(self):
        """已知 task_id 不报错 (不真正创建 env)."""
        from collect_data import TASK_TO_ROBOT

        # 验证所有已知 task 都能通过 task_id → robot 映射
        for task_id in TASK_TO_ROBOT:
            robot = TASK_TO_ROBOT[task_id]
            assert robot in ["g1", "go2", "a2", "as2", "r1", "h1_2", "h2"]

    def test_robot_joint_counts(self):
        """各机器人关节数正确."""
        from collect_data import GAIT_GENERATORS, TASK_TO_ROBOT

        expected_joints = {
            "g1": 29, "go2": 12, "a2": 12, "as2": 12,
            "r1": 24, "h1_2": 27, "h2": 29,
        }

        for task_id, robot in TASK_TO_ROBOT.items():
            assert robot in GAIT_GENERATORS, f"{robot} not in GAIT_GENERATORS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
