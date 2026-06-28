"""
G1 人形机器人配置（29 关节）

关节顺序：
- 左腿 6：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- 右腿 6：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- 腰部 3：waist_yaw, waist_roll, waist_pitch
- 左臂 7：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
- 右臂 7：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
"""

# 关节名称
LEFT_LEG = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
]
RIGHT_LEG = [
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
]
WAIST = ["waist_yaw", "waist_roll", "waist_pitch"]
LEFT_ARM = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
]
RIGHT_ARM = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

ALL_JOINTS = LEFT_LEG + RIGHT_LEG + WAIST + LEFT_ARM + RIGHT_ARM
NUM_JOINTS = len(ALL_JOINTS)  # 29

# 维度
STATE_DIM = 71  # joint_pos(29) + joint_vel(29) + base_pos(3) + base_quat(4) + base_lin_vel(3) + base_ang_vel(3)
ACTION_DIM = 29

# 关节索引切片
SLICES = {
    "joint_pos": (0, 29),
    "joint_vel": (29, 58),
    "base_pos": (58, 61),
    "base_quat": (61, 65),
    "base_lin_vel": (65, 68),
    "base_ang_vel": (68, 71),
}

# 关节限位（弧度）
JOINT_LIMITS = {
    "left_hip_pitch": (-2.5, 0.5),
    "left_hip_roll": (-0.5, 2.5),
    "left_hip_yaw": (-0.8, 0.8),
    "left_knee": (-0.1, 2.8),
    "left_ankle_pitch": (-0.8, 0.8),
    "left_ankle_roll": (-0.5, 0.5),
    "right_hip_pitch": (-2.5, 0.5),
    "right_hip_roll": (-2.5, 0.5),
    "right_hip_yaw": (-0.8, 0.8),
    "right_knee": (-0.1, 2.8),
    "right_ankle_pitch": (-0.8, 0.8),
    "right_ankle_roll": (-0.5, 0.5),
    "waist_yaw": (-2.6, 2.6),
    "waist_roll": (-0.3, 0.3),
    "waist_pitch": (-0.3, 0.3),
    "left_shoulder_pitch": (-3.1, 2.5),
    "left_shoulder_roll": (-0.5, 3.1),
    "left_shoulder_yaw": (-2.6, 2.6),
    "left_elbow": (-0.1, 2.8),
    "left_wrist_roll": (-1.5, 1.5),
    "left_wrist_pitch": (-1.5, 1.5),
    "left_wrist_yaw": (-1.5, 1.5),
    "right_shoulder_pitch": (-3.1, 2.5),
    "right_shoulder_roll": (-3.1, 0.5),
    "right_shoulder_yaw": (-2.6, 2.6),
    "right_elbow": (-0.1, 2.8),
    "right_wrist_roll": (-1.5, 1.5),
    "right_wrist_pitch": (-1.5, 1.5),
    "right_wrist_yaw": (-1.5, 1.5),
}

# 站立姿态 HOME
HOME_KEYFRAME = {
    "left_hip_pitch": -0.1,
    "left_knee": 0.3,
    "left_ankle_pitch": -0.2,
    "right_hip_pitch": -0.1,
    "right_knee": 0.3,
    "right_ankle_pitch": -0.2,
    "left_shoulder_pitch": 0.35,
    "left_elbow": 0.87,
    "right_shoulder_pitch": 0.35,
    "right_elbow": 0.87,
}
