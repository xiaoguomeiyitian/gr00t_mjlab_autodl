"""
端到端数据链路测试：验证 convert_to_lerobot 产出的数据 + ObservationBuilder
构建的观测能通过模拟的 Gr00tPolicy.check_observation 校验。

不连接真实 Policy Server，用本地 LeRobotEpisodeLoader 加载数据，
用 ObservationBuilder 构建观测，模拟 check_observation 的全部断言。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.convert_to_lerobot import convert_to_lerobot, _create_placeholder_video
from src.lerobot_loader import LeRobotEpisodeLoader
from src.observation_builder import ObservationBuilder, state_slices_from_config


def _mock_check_observation(obs: dict, modality_config: dict) -> None:
    """模拟 Isaac-GR00T Gr00tPolicy.check_observation 的关键校验。

    参考 gr00t/policy/gr00t_policy.py:240-380。
    """
    # 顶层三个 modality，且都是 dict
    for modality in ["video", "state", "language"]:
        assert modality in obs, f"缺少 {modality}"
        assert isinstance(obs[modality], dict), f"{modality} 必须是 dict"

    bs = -1

    # video: (B, T, H, W, 3) uint8
    for video_key in modality_config["video"]["modality_keys"]:
        assert video_key in obs["video"], f"video 缺少 {video_key}"
        v = obs["video"][video_key]
        assert isinstance(v, np.ndarray), f"video {video_key} 必须是 ndarray"
        assert v.dtype == np.uint8, f"video {video_key} dtype 必须是 uint8"
        assert v.ndim == 5, f"video {video_key} 必须是 5 维 (B,T,H,W,3)"
        assert v.shape[-1] == 3, f"video {video_key} 通道必须是 3"
        assert v.shape[1] == len(modality_config["video"]["delta_indices"]), \
            f"video {video_key} T 维不匹配"
        if bs == -1:
            bs = v.shape[0]
        else:
            assert v.shape[0] == bs, "batch size 不一致"

    # state: (B, T, D) float32
    for state_key in modality_config["state"]["modality_keys"]:
        assert state_key in obs["state"], f"state 缺少 {state_key}"
        s = obs["state"][state_key]
        assert isinstance(s, np.ndarray), f"state {state_key} 必须是 ndarray"
        assert s.dtype == np.float32, f"state {state_key} dtype 必须是 float32"
        assert s.ndim == 3, f"state {state_key} 必须是 3 维 (B,T,D)"
        assert s.shape[1] == len(modality_config["state"]["delta_indices"]), \
            f"state {state_key} T 维不匹配"
        if bs == -1:
            bs = s.shape[0]
        else:
            assert s.shape[0] == bs, "batch size 不一致"

    # language: list[list[list[str]]]
    for language_key in modality_config["language"]["modality_keys"]:
        assert language_key in obs["language"], f"language 缺少 {language_key}"
        val = obs["language"][language_key]
        assert isinstance(val, list), f"language {language_key} 必须是 list"
        assert len(val) == bs, f"language {language_key} B 维不匹配"
        for batch_item in val:
            assert isinstance(batch_item, list), "language batch item 必须是 list"
            assert len(batch_item) == len(modality_config["language"]["delta_indices"]), \
                "language T 维不匹配"
            assert len(batch_item) == 1, "language 每 timestep 1 个 str"
            assert isinstance(batch_item[0], str), "language 元素必须是 str"


class TestEndToEndDataPipeline:
    """端到端数据链路测试。"""

    @pytest.fixture
    def lerobot_dataset(self, temp_dir):
        """创建一个完整的 LeRobot v2 数据集（g1，2 episodes）。"""
        raw = temp_dir / "g1_raw"
        raw.mkdir()
        for ep_idx in range(2):
            np.savez_compressed(
                str(raw / f"episode_{ep_idx:04d}.npz"),
                states=np.random.randn(10, 71).astype(np.float32),
                actions=np.random.randn(10, 29).astype(np.float32),
                rewards=np.zeros(10, dtype=np.float32),
            )
            _create_placeholder_video(str(raw / f"episode_{ep_idx:04d}.mp4"), 10, 30, ["front"])

        out = temp_dir / "g1_lerobot"
        convert_to_lerobot(
            input_dir=str(raw), output_dir=str(out),
            robot="g1", task_description="walk forward",
        )
        return out

    def test_modality_json_keys_match_config(self, lerobot_dataset):
        """modality.json 的 state/action/video key 与 modality_config 一致。"""
        with open(lerobot_dataset / "meta" / "modality.json") as f:
            modality = json.load(f)
        # state keys 与 g1_modality_config.py 一致
        expected_state_keys = {"joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel"}
        assert set(modality["state"].keys()) == expected_state_keys
        # action key
        assert "joint_position_delta" in modality["action"]
        # video keys
        assert "front" in modality["video"]
        assert "wrist" in modality["video"]

    def test_observation_passes_check_observation(self, lerobot_dataset):
        """用 LeRobotEpisodeLoader + ObservationBuilder 构建观测，通过模拟 check_observation。"""
        dataset = LeRobotEpisodeLoader(dataset_path=str(lerobot_dataset))
        assert len(dataset) >= 1

        # 模拟 modality_config（与 g1_modality_config.py 一致）
        modality_config = {
            "video": {"delta_indices": [0], "modality_keys": ["front", "wrist"]},
            "state": {"delta_indices": [0], "modality_keys": [
                "joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel"]},
            "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
        }

        state_slices = state_slices_from_config("g1")
        obs_builder = ObservationBuilder(
            camera_keys=["front", "wrist"],
            state_dim=71,
            state_slices=state_slices,
            language_key="annotation.human.task_description",
            num_obs_steps=1,
        )

        # 对每个 episode 的前几帧构建观测并校验
        for traj_id in range(min(2, len(dataset))):
            episode = dataset[traj_id]
            task_desc = episode.get_task_description(0)
            for t in range(min(3, len(episode))):
                frame = episode.get_frame(t)
                obs = obs_builder.build(
                    images=frame["images"],
                    state=frame["state"],
                    language=task_desc,
                )
                # 应该通过模拟的 check_observation
                _mock_check_observation(obs, modality_config)

    def test_state_split_values_correct(self, lerobot_dataset):
        """state 拆分后的值与原始拼接向量对应。"""
        dataset = LeRobotEpisodeLoader(dataset_path=str(lerobot_dataset))
        episode = dataset[0]
        frame = episode.get_frame(0)
        original_state = frame["state"]  # (71,)

        state_slices = state_slices_from_config("g1")
        obs_builder = ObservationBuilder(
            camera_keys=["front"], state_dim=71, state_slices=state_slices,
        )
        obs = obs_builder.build(images=frame["images"], state=original_state)

        # 验证每个切片的值与原始向量对应区间一致
        for key, (start, end) in state_slices.items():
            np.testing.assert_array_equal(
                obs["state"][key][0, 0],
                original_state[start:end].astype(np.float32),
            )

    def test_tasks_jsonl_loadable(self, lerobot_dataset):
        """tasks.jsonl 能被 LeRobotEpisodeLoader 正确加载。"""
        dataset = LeRobotEpisodeLoader(dataset_path=str(lerobot_dataset))
        assert len(dataset.tasks) >= 1
        assert dataset.tasks[0]["task"] == "walk forward"
        # episode 能取到任务描述
        episode = dataset[0]
        assert episode.get_task_description(0) == "walk forward"
