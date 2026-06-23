"""Go2 modality config (delta / relative action) — registers for `NEW_EMBODIMENT`.

Go2 是 12 关节四足机器人, 与 G1 共用 state 字段结构 (joint_pos + base_*),
但只有 12 个关节。Action 同样为 16 步预测的关节增量。

参考:
  - Isaac-GR00T/examples/SO100/so100_config.py (RELATIVE action 模式)
  - Isaac-GR00T/gr00t/data/types.py:79-141 (ModalityConfig / ActionConfig)
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


NUM_JOINTS = 12
ACTION_HORIZON = 16


go2_delta_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_pos",      # 0:12
            "joint_vel",      # 12:24
            "base_pos",       # 24:27
            "base_quat",      # 27:31
            "base_lin_vel",   # 31:34
            "base_ang_vel",   # 34:37
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "joint_position_delta",
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
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(go2_delta_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
