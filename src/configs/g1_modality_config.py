"""
G1 人形机器人 ModalityConfig — 用于 Isaac-GR00T 微调训练。

在 AutoDL 云端运行：
    python gr00t/experiment/launch_finetune.py \
        --modality-config-path /root/gr00t_mjlab_autodl/g1_modality_config.py \
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
    """返回 G1 的 ModalityConfig dict（不注册）。"""
    return {
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


# 兼容旧接口：模块导入时自动注册为 NEW_EMBODIMENT
# 注意：单进程只能注册一个 NEW_EMBODIMENT，多机器人场景请用 get_modality_config() 手动注册
g1_config = get_modality_config()
register_modality_config(g1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
