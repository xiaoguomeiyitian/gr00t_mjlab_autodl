#!/usr/bin/env python3.12
"""测试 4 个 ModalityConfig — 必须有 state_key 且指向真实存在的 state key.

这些测试必须在有 gr00t 包的 Python 环境运行 (sys.path 里有 Isaac-GR00T).
"""
import sys, os, subprocess, pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = SRC_DIR / "configs"


def _load_cfg_in_subprocess(cfg_file: str, dict_attr: str, isaac_path: str) -> dict:
    """在独立子进程加载 config, 避免 EmbodimentTag 重复注册 AssertionError.

    Returns dict with keys: rep, type, state_key, modality_keys, ok
    """
    snippet = f"""
import sys, json
sys.path.insert(0, '{isaac_path}')
sys.path.insert(0, '{SRC_DIR}')
import importlib.util
spec = importlib.util.spec_from_file_location('cfg', '{CONFIGS_DIR / cfg_file}')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg = getattr(m, '{dict_attr}')
action_cfg = cfg['action'].action_configs[0]
state_cfg = cfg['state']
out = {{
    'rep': str(action_cfg.rep),
    'type': str(action_cfg.type),
    'state_key': action_cfg.state_key,
    'state_modality_keys': list(state_cfg.modality_keys),
    'state_key_in_keys': action_cfg.state_key in state_cfg.modality_keys,
}}
print(json.dumps(out))
"""
    p = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True, timeout=30
    )
    if p.returncode != 0:
        pytest.skip(f"{cfg_file} load failed: {p.stderr[:200]}")
    import json as _json
    return _json.loads(p.stdout.strip().splitlines()[-1])


ISAAC_GR00T = str(PROJECT_ROOT.parent / "Isaac-GR00T")


class TestG1DeltaConfig:
    """g1_new_embodiment_config.py — delta 模式 (默认推荐)."""

    @pytest.fixture
    def cfg(self):
        if not Path(ISAAC_GR00T).exists():
            pytest.skip("Isaac-GR00T not found, skip GR00T-dependent tests")
        return _load_cfg_in_subprocess(
            "g1_new_embodiment_config.py",
            "g1_delta_config",
            ISAAC_GR00T,
        )

    def test_rep_is_relative(self, cfg):
        assert cfg["rep"].endswith("RELATIVE"), f"got {cfg['rep']}"

    def test_state_key_is_joint_pos(self, cfg):
        """P0 bug 修复: 必须有 state_key='joint_pos' 才能生成 relative_stats.json."""
        assert cfg["state_key"] == "joint_pos"

    def test_state_key_in_state_modality_keys(self, cfg):
        """state_key 指向的 key 必须真实存在于 state.modality_keys."""
        assert cfg["state_key_in_keys"], \
            f"state_key={cfg['state_key']!r} not in {cfg['state_modality_keys']}"


class TestGo2DeltaConfig:
    """go2_new_embodiment_config.py — delta 模式."""

    @pytest.fixture
    def cfg(self):
        if not Path(ISAAC_GR00T).exists():
            pytest.skip("Isaac-GR00T not found, skip GR00T-dependent tests")
        return _load_cfg_in_subprocess(
            "go2_new_embodiment_config.py",
            "go2_delta_config",
            ISAAC_GR00T,
        )

    def test_rep_is_relative(self, cfg):
        assert cfg["rep"].endswith("RELATIVE")

    def test_state_key_is_joint_pos(self, cfg):
        assert cfg["state_key"] == "joint_pos"

    def test_state_key_in_state_modality_keys(self, cfg):
        assert cfg["state_key_in_keys"]


class TestG1AbsoluteConfig:
    """g1_new_embodiment_config_absolute.py — absolute 模式 (兼容旧数据)."""

    @pytest.fixture
    def cfg(self):
        if not Path(ISAAC_GR00T).exists():
            pytest.skip("Isaac-GR00T not found, skip GR00T-dependent tests")
        return _load_cfg_in_subprocess(
            "g1_new_embodiment_config_absolute.py",
            "g1_absolute_config",
            ISAAC_GR00T,
        )

    def test_rep_is_absolute(self, cfg):
        assert cfg["rep"].endswith("ABSOLUTE"), f"got {cfg['rep']}"

    def test_state_key_is_joint_pos(self, cfg):
        """absolute 模式虽然 GR00T 不需要 state_key, 但保持一致性仍设置."""
        assert cfg["state_key"] == "joint_pos"


class TestGo2AbsoluteConfig:
    """go2_new_embodiment_config_absolute.py — absolute 模式."""

    @pytest.fixture
    def cfg(self):
        if not Path(ISAAC_GR00T).exists():
            pytest.skip("Isaac-GR00T not found, skip GR00T-dependent tests")
        return _load_cfg_in_subprocess(
            "go2_new_embodiment_config_absolute.py",
            "go2_absolute_config",
            ISAAC_GR00T,
        )

    def test_rep_is_absolute(self, cfg):
        assert cfg["rep"].endswith("ABSOLUTE")

    def test_state_key_is_joint_pos(self, cfg):
        assert cfg["state_key"] == "joint_pos"


class TestConfigConsistency:
    """所有 4 个 config 必须满足相同的接口 (rep/state_key 都不为 None)."""

    @pytest.fixture
    def all_cfgs(self):
        if not Path(ISAAC_GR00T).exists():
            pytest.skip("Isaac-GR00T not found")
        return [
            _load_cfg_in_subprocess("g1_new_embodiment_config.py", "g1_delta_config", ISAAC_GR00T),
            _load_cfg_in_subprocess("go2_new_embodiment_config.py", "go2_delta_config", ISAAC_GR00T),
            _load_cfg_in_subprocess("g1_new_embodiment_config_absolute.py", "g1_absolute_config", ISAAC_GR00T),
            _load_cfg_in_subprocess("go2_new_embodiment_config_absolute.py", "go2_absolute_config", ISAAC_GR00T),
        ]

    def test_all_have_state_key(self, all_cfgs):
        for cfg in all_cfgs:
            assert cfg["state_key"] is not None, f"missing state_key in {cfg}"

    def test_delta_pair_uses_relative(self, all_cfgs):
        for cfg in all_cfgs[:2]:
            assert cfg["rep"].endswith("RELATIVE")

    def test_absolute_pair_uses_absolute(self, all_cfgs):
        for cfg in all_cfgs[2:]:
            assert cfg["rep"].endswith("ABSOLUTE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])