"""
H2 人形机器人 ModalityConfig — 用于 Isaac-GR00T 微调训练。

在 AutoDL 云端运行：
    python gr00t/experiment/launch_finetune.py \
        --modality-config-path /root/gr00t_mjlab_autodl/h2_modality_config.py \
        ...

注意：Isaac-GR00T 的 register_modality_config 不允许同一 tag 重复注册。
本文件提供 get_modality_config() 返回 config dict，由调用方按需注册，
避免多机器人共用 NEW_EMBODIMENT 互相覆盖。
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

def get_modality_config() -> dict:
    """返回 h2 的 ModalityConfig dict（不注册）。"""
    return {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
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
    # action key 与 state key "joint_pos" 一致，state_key 显式指定。
    # 数据集中存储绝对关节角，absolute→relative 转换由 processor 处理。
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["joint_pos"],
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


# 注意：不在模块导入时自动注册，由训练脚本按需调用：
#   from src.configs.h2_modality_config import get_modality_config
#   register_modality_config(get_modality_config(), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
