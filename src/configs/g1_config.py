"""G1 人形机器人配置 — 29 关节。

关节名称和默认姿态来源于 unitree_rl_mjlab 官方。
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

# ── 23Dof 变种配置 (Unitree-G1-23Dof-Flat/Rough) ───────────────────────────
# 与 unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml 一致
# 关节差异: 23Dof 砍掉了 waist_roll/waist_pitch/4 个 wrist_pitch/wrist_yaw (共 6 个)
# 顺序与 MJCF <joint> 出现顺序一致
G1_23DOF_NUM_JOINTS = 23

G1_23DOF_JOINT_NAMES = [
    # 左腿 (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿 (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰部 (1, 只剩 waist_yaw)
    "waist_yaw_joint",
    # 左臂 (5, 无 wrist_pitch/wrist_yaw)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint",
    # 右臂 (5)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint",
]
assert len(G1_23DOF_JOINT_NAMES) == 23

G1_23DOF_STATE_DIM = G1_23DOF_NUM_JOINTS * 2 + 3 + 4 + 3 + 3  # = 23*2+13 = 59
G1_23DOF_DT = 0.02

# 23Dof HOME_KEYFRAME (from g1_23dof_constants.py:179-191)
# 原配置用 regex 匹配, 实际值:
#   hip_pitch → -0.1, knee → 0.3, ankle_pitch → -0.2
#   shoulder_pitch → 0.35, elbow → 0.87
#   left_shoulder_roll → 0.18, right_shoulder_roll → -0.18
#   其余 (hip_roll/hip_yaw/ankle_roll/waist_yaw/shoulder_yaw/wrist_roll) → 0.0
G1_23DOF_DEFAULT_JOINT_ANGLES = {
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
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
}

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
G1_VIDEO_FPS = 50
G1_VIDEO_KEY = "video.front_view"

G1_DEFAULT_CAMERA_NAME = "front_view"


# LeRobot v2 modality.json (GR00T 兼容)
# 状态拼接顺序: joint_pos(29) | joint_vel(29) | base_pos(3) | base_quat(4)
#              | base_lin_vel(3) | base_ang_vel(3) = 总 71 维


def _build_state_layout(num_joints: int) -> tuple[list[tuple[str, int]], int]:
    """返回 (key 顺序 + 各 dim) 和总维度, 用于 state / start 索引计算."""
    layout = [
        ("joint_pos", num_joints),
        ("joint_vel", num_joints),
        ("base_pos", 3),
        ("base_quat", 4),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
    ]
    total = sum(d for _, d in layout)
    return layout, total


def _build_action_layout(action_mode: str, num_joints: int) -> tuple[list[tuple[str, int]], int]:
    """返回 (key 顺序 + 各 dim) 和总维度, 用于 action 拼接和 start 索引计算."""
    if action_mode == "absolute":
        layout = [("joint_position_target", num_joints)]
    elif action_mode == "delta":
        # 采集时同时存 target 和 delta, 但 GR00T 训练只选一种作为 action 主键
        layout = [("joint_position_delta", num_joints)]
    elif action_mode == "relative_eef":
        # ee_pose_delta: pos(3) + quat(4) = 7
        layout = [("ee_pose_delta", 7)]
    else:
        raise ValueError(f"未知 action_mode: {action_mode}")
    return layout, sum(d for _, d in layout)


def get_g1_modality_config(action_mode: str = "delta", include_video: bool = True) -> dict[str, Any]:
    """返回 G1 的 LeRobot v2 modality.json 内容 (GR00T 兼容).

    Args:
        action_mode: "absolute" | "delta" | "relative_eef"
        include_video: 是否包含 video 块 (采集时如未采视频须设 False)
    """
    state_layout, state_total = _build_state_layout(G1_NUM_JOINTS)
    action_layout, action_total = _build_action_layout(action_mode, G1_NUM_JOINTS)

    # state: { key: { start, end } } - 切片索引 (end 排他)
    state_block: dict[str, Any] = {}
    cursor = 0
    for key, dim in state_layout:
        state_block[key] = {"start": cursor, "end": cursor + dim}
        cursor += dim
    assert cursor == state_total, f"state dim mismatch: {cursor} != {state_total}"

    # action: 同理
    action_block: dict[str, Any] = {}
    cursor = 0
    for key, dim in action_layout:
        action_block[key] = {"start": cursor, "end": cursor + dim}
        cursor += dim
    assert cursor == action_total, f"action dim mismatch: {cursor} != {action_total}"

    # 顶层 dict
    cfg: dict[str, Any] = {
        "state": state_block,
        "action": action_block,
        # annotation 指向 parquet 中的 task_index 列 (int, 索引 tasks.jsonl)
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }

    # video 块 (可选, 实际视频 parquet 列 = observation.images.<video_key>)
    if include_video:
        cfg["video"] = {
            "front_view": {"original_key": "observation.images.front_view"},
        }

    return cfg


def get_g1_state_dim() -> int:
    """返回 G1 state 拼接后的总维度 (供信息查询)."""
    _, total = _build_state_layout(G1_NUM_JOINTS)
    return total


def get_g1_action_dim(action_mode: str = "delta") -> int:
    """返回 G1 action 拼接后的总维度."""
    _, total = _build_action_layout(action_mode, G1_NUM_JOINTS)
    return total
