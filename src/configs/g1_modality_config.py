"""
G1 人形机器人 ModalityConfig — 用于 Isaac-GR00T 微调训练。

在 AutoDL 云端运行：
    python gr00t/experiment/launch_finetune.py \
        --modality-config-path /autodl-fs/data/gr00t_mjlab_autodl/g1_modality_config.py \
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

g1_config = {
    # 视频：当前帧（G1 有两个相机视角 front + wrist）
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    # 本体感知：当前关节状态（必须与 modality.json 中 state key 对应）
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
                rep=ActionRepresentation.RELATIVE,  # delta from current joint_pos
                type=ActionType.NON_EEF,  # 关节空间，非末端执行器
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

register_modality_config(g1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
