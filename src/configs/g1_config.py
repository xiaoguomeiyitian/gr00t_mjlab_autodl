"""G1 人形机器人配置 — 29 关节。

关节名称和默认姿态来源于 unitree_rl_mjlab 官方:
  src/assets/robots/unitree_g1/g1_constants.py (HOME_KEYFRAME)
  src/tasks/velocity/config/g1/env_cfgs.py

GR00T 模态配置用于数据收集和 fine-tune 管线。
"""

from __future__ import annotations

from typing import Any


# ── 关节信息 (与 unitree_rl_mjlab MJCF 一致) ──────────────────────────────

G1_NUM_JOINTS = 29

G1_JOINT_NAMES = [
    # 左腿 (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿 (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰部 (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂 (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # 右臂 (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# GR00T 模态维度
# 真实维度: joint_pos(29) + joint_vel(29) + base_pos(3) + base_quat(4) + lin_vel(3) + ang_vel(3) = 71
G1_STATE_DIM = G1_NUM_JOINTS * 2 + 3 + 4 + 3 + 3  # = 71
G1_ACTION_DIM = G1_NUM_JOINTS  # = 29

# ── 动作空间选项 ──────────────────────────────────────────────────────────
# - "absolute": 关节目标绝对位置 (rad), 直接可喂 mjlab JointPositionActionCfg
# - "delta":    关节目标相对当前位置的增量 (rad), GR00T N1.7 推荐
# - "relative_eef": 末端执行器位姿增量 (GR00T 默认), 适用于操作任务
G1_ACTION_MODES = ("absolute", "delta", "relative_eef")
G1_DEFAULT_ACTION_MODE = "delta"

# ── 默认站立姿态 (HOME_KEYFRAME from unitree_rl_mjlab) ─────────────────────
# source: src/assets/robots/unitree_g1/g1_constants.py
G1_DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# ── unitree_rl_mjlab task ID ────────────────────────────────────────────────
G1_TASK_ID_FLAT = "Unitree-G1-Flat"
G1_TASK_ID_ROUGH = "Unitree-G1-Rough"

# 站立高度 (HOME_KEYFRAME.pos.z)
G1_BASE_HEIGHT = 0.8

# 仿真参数
G1_DT = 0.02  # 50 Hz (mjlab 默认)

# Embodiment tag (GR00T 专用)
G1_EMBODIMENT_TAG = "NEW_EMBODIMENT"
G1_EMBODIMENT_ID = 29

# ── 渲染 / 视频参数 ──────────────────────────────────────────────────────
G1_VIDEO_HEIGHT = 224
G1_VIDEO_WIDTH = 224
G1_VIDEO_FPS = 50  # mjlab 默认 50Hz, 与 G1_DT 匹配
G1_VIDEO_KEY = "video.front_view"
# mjlab 默认 camera name (位于 pelvis / torso 上方)
G1_DEFAULT_CAMERA_NAME = "front_view"


# ── LeRobot v2 modality.json ────────────────────────────────────────────────

def get_g1_modality_config(action_mode: str = "delta") -> dict[str, Any]:
    """返回 G1 的 LeRobot v2 modality.json 内容。

    Args:
        action_mode: "absolute" | "delta" | "relative_eef"
            - absolute:     action.joint_position_target  (29,)
            - delta:        action.joint_position_delta   (29,)  ← GR00T N1.7 推荐
            - relative_eef: action.ee_pose_delta         (7,)   ← 末端位姿增量
    """
    if action_mode == "absolute":
        action_block = {
            "action.joint_position_target": {
                "dtype": "float32",
                "shape": [G1_NUM_JOINTS],
                "description": "G1 29 关节目标位置 (rad, 绝对量)",
            },
        }
    elif action_mode == "delta":
        action_block = {
            "action.joint_position_delta": {
                "dtype": "float32",
                "shape": [G1_NUM_JOINTS],
                "description": "G1 29 关节目标位置增量 (rad, 相对当前)",
            },
            # 同时记录上一时刻绝对位置 (用于推理时累加回绝对量)
            "action.joint_position_last": {
                "dtype": "float32",
                "shape": [G1_NUM_JOINTS],
                "description": "G1 29 关节上一时刻位置 (rad, 用于 delta 累加)",
            },
        }
    elif action_mode == "relative_eef":
        action_block = {
            "action.ee_pose_delta": {
                "dtype": "float32",
                "shape": [7],
                "description": "末端位姿增量 (pos:3 + quat:4)",
            },
        }
    else:
        raise ValueError(f"未知 action_mode: {action_mode}")

    return {
        "state": {
            "state.joint_pos": {
                "dtype": "float32",
                "shape": [G1_NUM_JOINTS],
                "description": "G1 29 关节位置 (rad)",
            },
            "state.joint_vel": {
                "dtype": "float32",
                "shape": [G1_NUM_JOINTS],
                "description": "G1 29 关节速度 (rad/s)",
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
            G1_VIDEO_KEY: {
                "dtype": "video",
                "shape": [G1_VIDEO_HEIGHT, G1_VIDEO_WIDTH, 3],
                "fps": G1_VIDEO_FPS,
                "description": f"前方 RGB 相机 ({G1_VIDEO_HEIGHT}x{G1_VIDEO_WIDTH} @ {G1_VIDEO_FPS}fps)",
            },
        },
    }
