"""
Go2 四足机器人 ModalityConfig — 用于 Isaac-GR00T 微调训练。

在 AutoDL 云端运行：
    python gr00t/experiment/launch_finetune.py \
        --modality-config-path /root/gr00t_mjlab_autodl/go2_modality_config.py \
        ...
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

go2_config = {
    # 视频：当前帧
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "back"],
    ),
    # 本体感知：当前关节状态
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
    # 动作：16 步预测 horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["joint_position_delta"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # 语言指令
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(go2_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
