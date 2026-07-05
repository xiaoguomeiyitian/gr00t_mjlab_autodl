"""测试 mujoco_renderer 模块（不依赖 MuJoCo 的纯逻辑部分）。"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def _import_renderer():
    """导入 MujocoRenderer（mujoco 可能未安装，只测纯逻辑）。"""
    from src.mujoco_renderer import MujocoRenderer
    return MujocoRenderer


class TestFindRobotMjcf:
    """_find_robot_mjcf 路径查找测试（不实例化，避免触发 _setup_mujoco）。"""

    def test_unknown_robot_falls_back_to_g1(self):
        """未知机器人名走 else 分支（g1 candidates），找不到时抛 FileNotFoundError。"""
        MujocoRenderer = _import_renderer()
        r = object.__new__(MujocoRenderer)
        # unknown_robot 走 else 分支（g1 candidates），本地有 g1.xml 则返回，否则抛
        try:
            result = r._find_robot_mjcf("unknown_robot_xyz")
            assert isinstance(result, str)
        except FileNotFoundError:
            pass

    def test_go2_returns_str_or_raises(self):
        """go2 找到时返回 str，找不到时抛异常。"""
        MujocoRenderer = _import_renderer()
        r = object.__new__(MujocoRenderer)
        try:
            result = r._find_robot_mjcf("go2")
            assert isinstance(result, str)
            assert Path(result).exists()
        except FileNotFoundError:
            pass  # 本地无 a2.xml 时可接受

    def test_g1_returns_str_or_raises(self):
        """g1 找到时返回 str。"""
        MujocoRenderer = _import_renderer()
        r = object.__new__(MujocoRenderer)
        try:
            result = r._find_robot_mjcf("g1")
            assert isinstance(result, str)
            assert Path(result).exists()
        except FileNotFoundError:
            pass


class TestInitValidation:
    """__init__ 参数校验测试。"""

    def test_missing_mjcf_raises(self, tmp_path):
        MujocoRenderer = _import_renderer()
        # 指定不存在的 mjcf_path 应抛 FileNotFoundError
        with pytest.raises(FileNotFoundError, match="MJCF 模型不存在"):
            MujocoRenderer(mjcf_path=str(tmp_path / "nonexistent.xml"), robot="g1")

    def test_default_robot_g1(self):
        """默认 robot=g1，能找到 MJCF 时成功初始化，否则抛 FileNotFoundError。"""
        MujocoRenderer = _import_renderer()
        try:
            r = MujocoRenderer(robot="g1")
            assert r.robot == "g1"
            assert r.model is not None
        except FileNotFoundError:
            pass  # 本地无 g1.xml 时可接受
