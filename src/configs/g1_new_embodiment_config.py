"""G1 modality config (delta / relative action) — registers for `NEW_EMBODIMENT`.

This file is the canonical `register_modality_config(...)` registration that is
loaded by Isaac-GR00T's `launch_finetune.py --modality-config-path` argument and
by `gr00t/data/stats.py --modality-config-path`. The file is imported for its
side effects (the `register_modality_config` call at module bottom).

Mapping between this Python config and the on-disk `meta/modality.json`:

| ModalityConfig key       | meta/modality.json key   | Parquet column                  |
|--------------------------|--------------------------|---------------------------------|
| state.joint_pos          | state.joint_pos          | observation.state[0:29]         |
| state.joint_vel          | state.joint_vel          | observation.state[29:58]        |
| state.base_pos           | state.base_pos           | observation.state[58:61]        |
| state.base_quat          | state.base_quat          | observation.state[61:65]        |
| state.base_lin_vel       | state.base_lin_vel       | observation.state[65:68]        |
| state.base_ang_vel       | state.base_ang_vel       | observation.state[68:71]        |
| action.joint_position_delta | action.joint_position_delta | action[0:29]                |
| video.front_view         | video.front_view         | observation.images.front_view   |
| language.annotation.human.task_description | annotation.human.task_description | task_index (resolved via tasks.jsonl) |

The `register_modality_config` call must be at the bottom of the file (or
otherwise at module top-level), because Isaac-GR00T imports the file via
`importlib.import_module(<stem>)` and relies on the side effect.

Action representation: `RELATIVE` (NON_EEF) — the model is trained to predict
`target - current_joint_pos` deltas. At inference time, the consumer must
reconstruct the absolute target with `target = current_joint_pos + delta[0]`.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# ── 维度常量 (与 configs/g1_config.py 一致) ──────────────────────────
NUM_JOINTS = 29
ACTION_HORIZON = 16  # 预测未来 16 步 (与 examples/SO100/so100_config.py 一致)


g1_delta_config = {
    # ── 视频: 当前帧 (T=1) ──────────────────────────────────────
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view"],
    ),
    # ── 状态: 当前时刻本体感知 (T=1) ────────────────────────────
    # 6 个 key, 总维度 29*2 + 3 + 4 + 3 + 3 = 71
    # 顺序与 convert_to_lerobot.py 拼接顺序严格一致:
    #   joint_pos(29) | joint_vel(29) | base_pos(3) | base_quat(4)
    #                | base_lin_vel(3) | base_ang_vel(3)
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_pos",      # 0:29
            "joint_vel",      # 29:58
            "base_pos",       # 58:61
            "base_quat",      # 61:65
            "base_lin_vel",   # 65:68
            "base_ang_vel",   # 68:71
        ],
    ),
    # ── 动作: 16 步预测 ─────────────────────────────────────────
    # RELATIVE + NON_EEF → processor 在内部把 action 减去 current_state
    # 得到相对增量, 模型学的是增量, 反归一化时再加回 current_state
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "joint_position_delta",  # (29,) 相对 current_joint_pos 的增量
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="joint_pos",
            ),
        ],
    ),
    # ── 语言: 当前任务的文字描述 ───────────────────────────────
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# 副作用注册: 导入此文件即注册 NEW_EMBODIMENT
register_modality_config(g1_delta_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
