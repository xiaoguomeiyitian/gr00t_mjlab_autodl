"""Go2 四足机器人配置 — 12 关节。

关节名称和默认姿态来源于 unitree_rl_mjlab 官方。
"""

from __future__ import annotations

from typing import Any


# ── 关节信息 (与 unitree_rl_mjlab MJCF 一致) ──────────────────────────────

GO2_NUM_JOINTS = 12

GO2_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

GO2_STATE_DIM = GO2_NUM_JOINTS * 2 + 3 + 4 + 3 + 3  # = 37
GO2_ACTION_DIM = GO2_NUM_JOINTS  # = 12

# ── 动作空间选项 ──────────────────────────────────────────────────────────
GO2_ACTION_MODES = ("absolute", "delta")
GO2_DEFAULT_ACTION_MODE = "delta"

# 默认站立姿态 (INIT_STATE from unitree_rl_mjlab)
GO2_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": -0.1,
    "FL_thigh_joint": 0.9,
    "FL_calf_joint": -1.8,
    "FR_hip_joint": 0.1,
    "FR_thigh_joint": 0.9,
    "FR_calf_joint": -1.8,
    "RL_hip_joint": -0.1,
    "RL_thigh_joint": 0.9,
    "RL_calf_joint": -1.8,
    "RR_hip_joint": 0.1,
    "RR_thigh_joint": 0.9,
    "RR_calf_joint": -1.8,
}

# ── unitree_rl_mjlab task ID ────────────────────────────────────────────────
GO2_TASK_ID_FLAT = "Unitree-Go2-Flat"
GO2_TASK_ID_ROUGH = "Unitree-Go2-Rough"

# 站立高度 (INIT_STATE.pos.z)
GO2_BASE_HEIGHT = 0.32

# 仿真参数
GO2_DT = 0.02  # 50 Hz

# Embodiment tag (GR00T 专用)
GO2_EMBODIMENT_TAG = "NEW_EMBODIMENT"
GO2_EMBODIMENT_ID = 29

# ── 渲染 / 视频参数 ──────────────────────────────────────────────────────
GO2_VIDEO_HEIGHT = 224
GO2_VIDEO_WIDTH = 224
GO2_VIDEO_FPS = 50
GO2_VIDEO_KEY = "video.front_view"
GO2_DEFAULT_CAMERA_NAME = "front_view"


# LeRobot v2 modality.json (GR00T 兼容)
# 状态拼接顺序: joint_pos(12) | joint_vel(12) | base_pos(3) | base_quat(4)
#              | base_lin_vel(3) | base_ang_vel(3) = 总 37 维


def get_go2_modality_config(action_mode: str = "delta", include_video: bool = True) -> dict[str, Any]:
    """返回 Go2 的 LeRobot v2 modality.json 内容 (GR00T 兼容).

    Args:
        action_mode: "absolute" | "delta"
        include_video: 是否包含 video 块 (采集时如未采视频须设 False)
    """
    # 复用 G1 的 layout 辅助函数 (go2 与 g1 的 state/action 字段结构一致)
    from configs.g1_config import _build_state_layout, _build_action_layout

    state_layout, state_total = _build_state_layout(GO2_NUM_JOINTS)
    action_layout, action_total = _build_action_layout(action_mode, GO2_NUM_JOINTS)

    state_block: dict[str, Any] = {}
    cursor = 0
    for key, dim in state_layout:
        state_block[key] = {"start": cursor, "end": cursor + dim}
        cursor += dim
    assert cursor == state_total, f"state dim mismatch: {cursor} != {state_total}"

    action_block: dict[str, Any] = {}
    cursor = 0
    for key, dim in action_layout:
        action_block[key] = {"start": cursor, "end": cursor + dim}
        cursor += dim
    assert cursor == action_total, f"action dim mismatch: {cursor} != {action_total}"

    cfg: dict[str, Any] = {
        "state": state_block,
        "action": action_block,
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }

    if include_video:
        cfg["video"] = {
            "front_view": {"original_key": "observation.images.front_view"},
        }

    return cfg


def get_go2_state_dim() -> int:
    """返回 Go2 state 拼接后的总维度."""
    from configs.g1_config import _build_state_layout
    _, total = _build_state_layout(GO2_NUM_JOINTS)
    return total


def get_go2_action_dim(action_mode: str = "delta") -> int:
    """返回 Go2 action 拼接后的总维度."""
    from configs.g1_config import _build_action_layout
    _, total = _build_action_layout(action_mode, GO2_NUM_JOINTS)
    return total
