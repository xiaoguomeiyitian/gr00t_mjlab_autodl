"""测试机器人配置模块。"""

import pytest
import numpy as np


class TestG1Config:
    """G1 人形机器人配置测试。"""

    def test_num_joints(self):
        from src.configs.g1_config import NUM_JOINTS
        assert NUM_JOINTS == 29

    def test_state_dim(self):
        from src.configs.g1_config import STATE_DIM
        assert STATE_DIM == 71

    def test_action_dim(self):
        from src.configs.g1_config import ACTION_DIM
        assert ACTION_DIM == 29

    def test_all_joints_length(self):
        from src.configs.g1_config import ALL_JOINTS
        assert len(ALL_JOINTS) == 29

    def test_all_joints_unique(self):
        from src.configs.g1_config import ALL_JOINTS
        assert len(set(ALL_JOINTS)) == len(ALL_JOINTS)

    def test_joint_groups(self):
        from src.configs.g1_config import LEFT_LEG, RIGHT_LEG, WAIST, LEFT_ARM, RIGHT_ARM
        assert len(LEFT_LEG) == 6
        assert len(RIGHT_LEG) == 6
        assert len(WAIST) == 3
        assert len(LEFT_ARM) == 7
        assert len(RIGHT_ARM) == 7

    def test_all_joints_composition(self):
        from src.configs.g1_config import ALL_JOINTS, LEFT_LEG, RIGHT_LEG, WAIST, LEFT_ARM, RIGHT_ARM
        expected = LEFT_LEG + RIGHT_LEG + WAIST + LEFT_ARM + RIGHT_ARM
        assert ALL_JOINTS == expected

    def test_slices(self):
        from src.configs.g1_config import SLICES
        assert SLICES["joint_pos"] == (0, 29)
        assert SLICES["joint_vel"] == (29, 58)
        assert SLICES["base_pos"] == (58, 61)
        assert SLICES["base_quat"] == (61, 65)
        assert SLICES["base_lin_vel"] == (65, 68)
        assert SLICES["base_ang_vel"] == (68, 71)

    def test_slices_contiguous(self):
        from src.configs.g1_config import SLICES
        sorted_slices = sorted(SLICES.values(), key=lambda x: x[0])
        for i in range(len(sorted_slices) - 1):
            assert sorted_slices[i][1] == sorted_slices[i + 1][0]

    def test_joint_limits_exist_for_all_joints(self):
        from src.configs.g1_config import ALL_JOINTS, JOINT_LIMITS
        for joint in ALL_JOINTS:
            assert joint in JOINT_LIMITS, f"Missing limit for {joint}"

    def test_joint_limits_ordered(self):
        from src.configs.g1_config import JOINT_LIMITS
        for joint, (lo, hi) in JOINT_LIMITS.items():
            assert lo < hi, f"{joint}: lower {lo} >= upper {hi}"

    def test_home_keyframe_valid(self):
        from src.configs.g1_config import HOME_KEYFRAME, JOINT_LIMITS
        for joint, value in HOME_KEYFRAME.items():
            assert joint in JOINT_LIMITS
            lo, hi = JOINT_LIMITS[joint]
            assert lo <= value <= hi, f"{joint}: {value} not in [{lo}, {hi}]"


class TestGo2Config:
    """Go2 四足机器人配置测试。"""

    def test_num_joints(self):
        from src.configs.go2_config import NUM_JOINTS
        assert NUM_JOINTS == 12

    def test_state_dim(self):
        from src.configs.go2_config import STATE_DIM
        assert STATE_DIM == 37

    def test_action_dim(self):
        from src.configs.go2_config import ACTION_DIM
        assert ACTION_DIM == 12

    def test_all_joints_length(self):
        from src.configs.go2_config import ALL_JOINTS
        assert len(ALL_JOINTS) == 12

    def test_joint_naming_pattern(self):
        from src.configs.go2_config import ALL_JOINTS, LEGS, JOINT_TYPES
        expected = [f"{leg}_{joint}" for leg in LEGS for joint in JOINT_TYPES]
        assert ALL_JOINTS == expected

    def test_slices(self):
        from src.configs.go2_config import SLICES
        assert SLICES["joint_pos"] == (0, 12)
        assert SLICES["joint_vel"] == (12, 24)
        assert SLICES["base_pos"] == (24, 27)
        assert SLICES["base_quat"] == (27, 31)
        assert SLICES["base_lin_vel"] == (31, 34)
        assert SLICES["base_ang_vel"] == (34, 37)

    def test_init_state_all_joints(self):
        from src.configs.go2_config import INIT_STATE, ALL_JOINTS
        for joint in ALL_JOINTS:
            assert joint in INIT_STATE, f"Missing init state for {joint}"
