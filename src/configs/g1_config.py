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
G1_STATE_DIM = 37   # 12 joint_pos + 12 joint_vel + 3 base_pos + 4 base_quat + 3 lin_vel + 3 ang_vel
G1_ACTION_DIM = 29  # 全部 29 关节

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


# ── LeRobot v2 modality.json ────────────────────────────────────────────────

def get_g1_modality_config() -> dict[str, Any]:
    """返回 G1 的 LeRobot v2 modality.json 内容。"""
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
                "description": "基座位置 (m)",
            },
            "state.base_quat": {
                "dtype": "float32",
                "shape": [4],
                "description": "基座四元数 (wxyz)",
            },
            "state.base_lin_vel": {
                "dtype": "float32",
                "shape": [3],
                "description": "基座线速度 (m/s)",
            },
            "state.base_ang_vel": {
                "dtype": "float32",
                "shape": [3],
                "description": "基座角速度 (rad/s)",
            },
        },
        "action": {
            "action.joint_position_target": {
                "dtype": "float32",
                "shape": [G1_ACTION_DIM],
                "description": "G1 29 关节目标位置 (rad)",
            },
        },
        "video": {
            "video.front_view": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "fps": 50,
                "description": "前方 RGB 相机 (224x224)",
            },
        },
    }
