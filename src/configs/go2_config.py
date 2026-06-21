"""Go2 四足机器人配置 — 12 关节。

关节名称和默认姿态来源于 unitree_rl_mjlab 官方:
  src/assets/robots/unitree_go2/go2_constants.py (INIT_STATE)
  src/tasks/velocity/config/go2/env_cfgs.py

GR00T 模态配置用于数据收集和 fine-tune 管线。
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

# ── 默认站立姿态 (INIT_STATE from unitree_rl_mjlab) ────────────────────────
# source: src/assets/robots/unitree_go2/go2_constants.py
GO2_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.9,
    "FL_calf_joint": -1.8,
    "FR_hip_joint": 0.1,
    "FR_thigh_joint": 0.9,
    "FR_calf_joint": -1.8,
    "RL_hip_joint": -0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.8,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.0,
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


# ── LeRobot v2 modality.json ────────────────────────────────────────────────

def get_go2_modality_config(action_mode: str = "delta") -> dict[str, Any]:
    """返回 Go2 的 LeRobot v2 modality.json 内容。

    Args:
        action_mode: "absolute" | "delta"
    """
    if action_mode == "absolute":
        action_block = {
            "action.joint_position_target": {
                "dtype": "float32",
                "shape": [GO2_NUM_JOINTS],
                "description": "Go2 12 关节目标位置 (rad, 绝对量)",
            },
        }
    elif action_mode == "delta":
        action_block = {
            "action.joint_position_delta": {
                "dtype": "float32",
                "shape": [GO2_NUM_JOINTS],
                "description": "Go2 12 关节目标位置增量 (rad, 相对当前)",
            },
            "action.joint_position_last": {
                "dtype": "float32",
                "shape": [GO2_NUM_JOINTS],
                "description": "Go2 12 关节上一时刻位置 (rad, 用于 delta 累加)",
            },
        }
    else:
        raise ValueError(f"未知 action_mode: {action_mode}")

    return {
        "state": {
            "state.joint_pos": {
                "dtype": "float32",
                "shape": [GO2_NUM_JOINTS],
                "description": "Go2 12 关节位置 (rad)",
            },
            "state.joint_vel": {
                "dtype": "float32",
                "shape": [GO2_NUM_JOINTS],
                "description": "Go2 12 关节速度 (rad/s)",
            },
            "state.base_pos": {
                "dtype": "float32",
                "shape": [3],
                "description": "基座位置 (m, world frame)",
            },
            "state.base_quat": {
                "dtype": "float32",
                "shape": [4],
                "description": "基座四元数 (wxyz, world frame)",
            },
            "state.base_lin_vel": {
                "dtype": "float32",
                "shape": [3],
                "description": "基座线速度 (m/s, world frame)",
            },
            "state.base_ang_vel": {
                "dtype": "float32",
                "shape": [3],
                "description": "基座角速度 (rad/s, world frame)",
            },
        },
        "action": action_block,
        "video": {
            GO2_VIDEO_KEY: {
                "dtype": "video",
                "shape": [GO2_VIDEO_HEIGHT, GO2_VIDEO_WIDTH, 3],
                "fps": GO2_VIDEO_FPS,
                "description": f"前方 RGB 相机 ({GO2_VIDEO_HEIGHT}x{GO2_VIDEO_WIDTH} @ {GO2_VIDEO_FPS}fps)",
            },
        },
    }
