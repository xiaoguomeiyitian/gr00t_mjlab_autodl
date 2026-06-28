"""
Go2 四足机器人配置（12 关节）

关节顺序：FL/FR/RL/RR × hip/thigh/calf
"""

# 关节名称
LEGS = ["FL", "FR", "RL", "RR"]
JOINT_TYPES = ["hip", "thigh", "calf"]
ALL_JOINTS = [f"{leg}_{joint}" for leg in LEGS for joint in JOINT_TYPES]
NUM_JOINTS = len(ALL_JOINTS)  # 12

# 维度
STATE_DIM = 37  # joint_pos(12) + joint_vel(12) + base_pos(3) + base_quat(4) + base_lin_vel(3) + base_ang_vel(3)
ACTION_DIM = 12

# 关节索引切片
SLICES = {
    "joint_pos": (0, 12),
    "joint_vel": (12, 24),
    "base_pos": (24, 27),
    "base_quat": (27, 31),
    "base_lin_vel": (31, 34),
    "base_ang_vel": (34, 37),
}

# 关节限位（弧度）
JOINT_LIMITS = {
    "FL_hip": (-1.0, 1.0),
    "FL_thigh": (-0.5, 2.5),
    "FL_calf": (-2.8, -0.1),
    "FR_hip": (-1.0, 1.0),
    "FR_thigh": (-0.5, 2.5),
    "FR_calf": (-2.8, -0.1),
    "RL_hip": (-1.0, 1.0),
    "RL_thigh": (-0.5, 2.5),
    "RL_calf": (-2.8, -0.1),
    "RR_hip": (-1.0, 1.0),
    "RR_thigh": (-0.5, 2.5),
    "RR_calf": (-2.8, -0.1),
}

# 初始站立姿态
INIT_STATE = {
    "FL_hip": 0.1,
    "FR_hip": -0.1,
    "RL_hip": 0.1,
    "RR_hip": -0.1,
    "FL_thigh": 0.9,
    "FR_thigh": 0.9,
    "RL_thigh": 0.9,
    "RR_thigh": 0.9,
    "FL_calf": -1.8,
    "FR_calf": -1.8,
    "RL_calf": -1.8,
    "RR_calf": -1.8,
}
