"""共享 pytest 配置和 fixtures。"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─── 路径 fixtures ───

@pytest.fixture
def project_root() -> Path:
    """项目根目录路径。"""
    return Path(__file__).parent.parent


@pytest.fixture
def src_dir(project_root: Path) -> Path:
    """src/ 目录路径。"""
    return project_root / "src"


@pytest.fixture
def temp_dir() -> Path:
    """临时目录（测试结束后自动清理）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ─── 数据 fixtures ───

@pytest.fixture
def g1_raw_episode(temp_dir: Path) -> dict:
    """生成一个模拟的 G1 episode 数据（npz 内容）。"""
    num_steps = 10
    state_dim = 71
    action_dim = 29
    return {
        "states": np.random.randn(num_steps, state_dim).astype(np.float32),
        "actions": np.random.randn(num_steps, action_dim).astype(np.float32),
        "rewards": np.random.randn(num_steps).astype(np.float32),
        "task_name": "Mjlab-Velocity-Flat-Unitree-G1",
        "robot": "g1",
        "action_mode": "delta",
    }


@pytest.fixture
def go2_raw_episode(temp_dir: Path) -> dict:
    """生成一个模拟的 Go2 episode 数据（npz 内容）。"""
    num_steps = 8
    state_dim = 37
    action_dim = 12
    return {
        "states": np.random.randn(num_steps, state_dim).astype(np.float32),
        "actions": np.random.randn(num_steps, action_dim).astype(np.float32),
        "rewards": np.random.randn(num_steps).astype(np.float32),
        "task_name": "Mjlab-Velocity-Flat-Unitree-Go2",
        "robot": "go2",
        "action_mode": "delta",
    }


@pytest.fixture
def sample_safetensors(temp_dir: Path) -> Path:
    """创建模拟 safetensors 模型文件。"""
    from safetensors.numpy import save_file

    model_dir = temp_dir / "test_model"
    model_dir.mkdir()

    weights = {
        "diffusion.net.0.weight": np.random.randn(128, 256).astype(np.float32),
        "diffusion.net.2.weight": np.random.randn(64, 128).astype(np.float32),
        "projector.weight": np.random.randn(32, 64).astype(np.float32),
        "layernorm.weight": np.random.randn(128).astype(np.float32),
        "layernorm.bias": np.random.randn(128).astype(np.float32),
    }
    save_file(weights, str(model_dir / "model.safetensors"))

    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump({"model_type": "test"}, f)

    return model_dir
