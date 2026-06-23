"""Go2 modality config (absolute action) — registers for `NEW_EMBODIMENT`.

Go2 absolute action variant. Use with `--action-mode absolute` when collecting.
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


go2_absolute_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_pos",
            "joint_vel",
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "joint_position_delta",  # absolute 模式下实际存 joint_position_target
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="joint_pos",   # ← 修复: 显式声明, 保持与 delta 变体对称
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(go2_absolute_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
