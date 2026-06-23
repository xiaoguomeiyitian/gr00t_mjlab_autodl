#!/usr/bin/env python3.12
"""测试 g1_config.py / go2_config.py — 不依赖 GR00T 的纯常量与关节布局."""
import sys, pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestG1Config:
    """G1 关节名称、维度、默认姿态."""

    def test_num_joints_is_29(self):
        from configs.g1_config import G1_NUM_JOINTS
        assert G1_NUM_JOINTS == 29

    def test_joint_names_count_matches_num(self):
        from configs.g1_config import G1_JOINT_NAMES, G1_NUM_JOINTS
        assert len(G1_JOINT_NAMES) == G1_NUM_JOINTS

    def test_joint_names_unique(self):
        from configs.g1_config import G1_JOINT_NAMES
        assert len(set(G1_JOINT_NAMES)) == len(G1_JOINT_NAMES)

    def test_state_dim_equals_71(self):
        """joint_pos(29) + joint_vel(29) + base_pos(3) + base_quat(4) + lin_vel(3) + ang_vel(3) = 71"""
        from configs.g1_config import G1_STATE_DIM
        assert G1_STATE_DIM == 71

    def test_action_dim_equals_num_joints(self):
        from configs.g1_config import G1_ACTION_DIM, G1_NUM_JOINTS
        assert G1_ACTION_DIM == G1_NUM_JOINTS

    def test_default_angles_for_all_joints(self):
        """G1_DEFAULT_JOINT_ANGLES 必须覆盖所有 G1_JOINT_NAMES."""
        from configs.g1_config import G1_JOINT_NAMES, G1_DEFAULT_JOINT_ANGLES
        missing = [n for n in G1_JOINT_NAMES if n not in G1_DEFAULT_JOINT_ANGLES]
        assert missing == [], f"missing default angles for: {missing}"

    def test_default_angles_count(self):
        from configs.g1_config import G1_JOINT_NAMES, G1_DEFAULT_JOINT_ANGLES
        assert len(G1_DEFAULT_JOINT_ANGLES) == len(G1_JOINT_NAMES)

    def test_23dof_joints_subset(self):
        """23Dof 是 29Dof 的子集 — 关节名必须出现在 G1_JOINT_NAMES 中."""
        from configs.g1_config import G1_JOINT_NAMES, G1_23DOF_JOINT_NAMES, G1_23DOF_NUM_JOINTS
        assert G1_23DOF_NUM_JOINTS == 23
        assert len(G1_23DOF_JOINT_NAMES) == 23
        for name in G1_23DOF_JOINT_NAMES:
            assert name in G1_JOINT_NAMES, f"23Dof joint {name} not in 29Dof list"


class TestGo2Config:
    """Go2 关节名称、维度、默认姿态."""

    def test_num_joints_is_12(self):
        from configs.go2_config import GO2_NUM_JOINTS
        assert GO2_NUM_JOINTS == 12

    def test_state_dim_equals_37(self):
        """joint_pos(12) + joint_vel(12) + base_pos(3) + base_quat(4) + lin_vel(3) + ang_vel(3) = 37"""
        from configs.go2_config import GO2_STATE_DIM
        assert GO2_STATE_DIM == 37

    def test_action_modes_contain_delta_and_absolute(self):
        from configs.go2_config import GO2_ACTION_MODES
        assert "delta" in GO2_ACTION_MODES
        assert "absolute" in GO2_ACTION_MODES

    def test_default_action_mode_is_delta(self):
        """修复后默认应为 delta (与 GR00T 官方一致)."""
        from configs.go2_config import GO2_DEFAULT_ACTION_MODE
        assert GO2_DEFAULT_ACTION_MODE == "delta"

    def test_default_angles_match_mjlab_official(self):
        """默认姿态对齐 unitree_rl_mjlab 官方 (左 hip=-0.1, 右 hip=+0.1)."""
        from configs.go2_config import GO2_DEFAULT_JOINT_ANGLES
        assert GO2_DEFAULT_JOINT_ANGLES["FL_hip_joint"] == -0.1
        assert GO2_DEFAULT_JOINT_ANGLES["FR_hip_joint"] == 0.1
        assert GO2_DEFAULT_JOINT_ANGLES["RL_hip_joint"] == -0.1
        assert GO2_DEFAULT_JOINT_ANGLES["RR_hip_joint"] == 0.1

    def test_thigh_and_calf_consistent(self):
        """4 条腿 thigh 应相同, calf 应相同."""
        from configs.go2_config import GO2_DEFAULT_JOINT_ANGLES
        thigh_values = {GO2_DEFAULT_JOINT_ANGLES[f"{leg}_thigh_joint"]
                        for leg in ["FL", "FR", "RL", "RR"]}
        calf_values = {GO2_DEFAULT_JOINT_ANGLES[f"{leg}_calf_joint"]
                       for leg in ["FL", "FR", "RL", "RR"]}
        assert thigh_values == {0.9}, f"thigh values differ: {thigh_values}"
        assert calf_values == {-1.8}, f"calf values differ: {calf_values}"


class TestRobotDimensionsConsistency:
    """G1 和 Go2 的状态/动作维度一致性."""

    def test_g1_state_dim_matches_layout(self):
        from configs.g1_config import G1_NUM_JOINTS, G1_STATE_DIM
        # joint_pos + joint_vel = 2 * num_joints, base = 3+4+3+3 = 13
        assert G1_STATE_DIM == G1_NUM_JOINTS * 2 + 13

    def test_go2_state_dim_matches_layout(self):
        from configs.go2_config import GO2_NUM_JOINTS, GO2_STATE_DIM
        assert GO2_STATE_DIM == GO2_NUM_JOINTS * 2 + 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])