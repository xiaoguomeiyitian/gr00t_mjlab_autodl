"""
Go2 四足机器人 ModalityConfig — 用于 Isaac-GR00T 微调训练。

在 AutoDL 云端运行：
    python gr00t/experiment/launch_finetune.py \
        --modality-config-path /root/gr00t_mjlab_autodl/go2_modality_config.py \
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
    """返回 go2 的 ModalityConfig dict（不注册）。"""
    return {
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


# 兼容旧接口：模块导入时自动注册为 NEW_EMBODIMENT
# 注意：单进程只能注册一个 NEW_EMBODIMENT，多机器人场景请用 get_modality_config() 手动注册
go2_config = get_modality_config()
register_modality_config(go2_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
