"""测试 6 个 modality_config 文件的 get_modality_config() 结构。"""

import pytest

# gr00t 未安装时跳过整个文件
pytest.importorskip("gr00t")


@pytest.mark.parametrize("robot,module_name", [
    ("g1", "g1_modality_config"),
    ("go2", "go2_modality_config"),
    ("h1", "h1_modality_config"),
    ("h1_2", "h1_2_modality_config"),
    ("h1_with_hand", "h1_with_hand_modality_config"),
    ("h2", "h2_modality_config"),
])
def test_get_modality_config_structure(robot, module_name):
    """每个 modality_config 的 get_modality_config() 返回结构正确的 dict。"""
    import importlib
    mod = importlib.import_module(f"src.configs.{module_name}")
    config = mod.get_modality_config()

    # 必须含 4 个 modality
    assert "video" in config
    assert "state" in config
    assert "action" in config
    assert "language" in config

    # video.modality_keys 是 list
    video_keys = config["video"].modality_keys
    assert isinstance(video_keys, list)
    assert len(video_keys) >= 1

    # state.modality_keys 含 joint_pos
    state_keys = config["state"].modality_keys
    assert "joint_pos" in state_keys

    # action.delta_indices 是 list 且非空
    action_delta = config["action"].delta_indices
    assert isinstance(action_delta, list)
    assert len(action_delta) > 0


@pytest.mark.parametrize("robot,module_name,expected_video_keys", [
    ("g1", "g1_modality_config", ["front", "wrist"]),
    ("go2", "go2_modality_config", ["front", "back"]),
    ("h1", "h1_modality_config", ["front", "wrist"]),
])
def test_video_keys_per_robot(robot, module_name, expected_video_keys):
    """不同机器人的 video.modality_keys 应正确。"""
    import importlib
    mod = importlib.import_module(f"src.configs.{module_name}")
    config = mod.get_modality_config()
    assert config["video"].modality_keys == expected_video_keys


def test_action_horizon_16():
    """action delta_indices 应为 16 步 horizon。"""
    from src.configs.g1_modality_config import get_modality_config
    config = get_modality_config()
    assert config["action"].delta_indices == list(range(0, 16))
