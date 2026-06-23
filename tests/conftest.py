#!/usr/bin/env python3.12
"""Shared pytest fixtures for gr00t_mjlab_autodl tests.

关键点:
  - 自动把 src/ 加入 sys.path (其他 test_* 不必重复)
  - 把 Isaac-GR00T 加入 sys.path (用于 new_embodiment_config 测试)
  - 提供 mock_torch fixture 让 infer 测试不需要真实 GPU
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ISAAC_GR00T = PROJECT_ROOT.parent / "Isaac-GR00T"

# 路径注入 (一次性, 多次 import 不会重复加)
for p in [str(SRC_DIR), str(ISAAC_GR00T)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def mock_torch():
    """Mock torch 模块 — 让 device='auto' 走 cpu 分支."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.randn = lambda *args, **kwargs: __import__("numpy").zeros(args)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        yield fake_torch


@pytest.fixture
def fake_model_path(tmp_path):
    """创建一个空的 model 目录 (供 GR00TLocalInference 实例化)."""
    p = tmp_path / "fake_model"
    p.mkdir()
    return p


@pytest.fixture(autouse=True)
def reset_action_queue():
    """避免 infer 测试间共享 _action_queue."""
    yield
    # pytest 不需要显式清理, 每个 fixture 是 fresh instance