#!/usr/bin/env python3.12
"""测试 collect_data.py 中:
   - 步态生成器 (G1 29Dof, G1 23Dof, Go2, generic)
   - TASK_TO_ROBOT 映射
   - _to_numpy 转换工具
"""
import sys, pytest, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestG1GaitGenerator:
    """gait_generator_g1 — 必须产生正确维度的关节目标."""

    def test_default_29dof(self):
        from collect_data import gait_generator_g1
        out = gait_generator_g1(0)
        assert out.shape == (29,)
        assert out.dtype == np.float32

    def test_explicit_29dof(self):
        from collect_data import gait_generator_g1
        out = gait_generator_g1(0, num_joints=29)
        assert out.shape == (29,)

    def test_23dof_variant(self):
        """23Dof 变种 (Unitree-G1-23Dof-Flat/Rough)."""
        from collect_data import gait_generator_g1
        out = gait_generator_g1(0, num_joints=23)
        assert out.shape == (23,)
        assert out.dtype == np.float32

    def test_step_advances_signal(self):
        """不同 step 应该产生不同动作 (不是常量).

        注意: 不能用 step=50, 因为 speed=0.5 时 freq=1, 50*0.02*1=1.0,
        sin(2*pi) = 0, 回到原姿态. 用 step=5 时 t=0.1, phase=sin(2*pi*1*0.1)≈0.59
        """
        from collect_data import gait_generator_g1
        a = gait_generator_g1(0, speed=0.5)        # phase=0
        b = gait_generator_g1(5, speed=0.5)        # phase≈0.59
        c = gait_generator_g1(10, speed=0.5)       # phase≈0.95
        assert not np.allclose(a, b)
        assert not np.allclose(a, c)
        # hip_pitch (idx 0) 应在步态中变化
        assert abs(a[0] - b[0]) > 0.01

    def test_speed_affects_frequency(self):
        """更高 speed → 更快的相位变化."""
        from collect_data import gait_generator_g1
        a_slow = gait_generator_g1(10, speed=0.3)
        a_fast = gait_generator_g1(10, speed=1.5)
        # 至少 hip_pitch 应该有差异
        assert not np.allclose(a_slow, a_fast)

    def test_within_joint_limits(self):
        """关节目标应在合理范围内 (±π)."""
        from collect_data import gait_generator_g1
        for step in range(0, 200, 10):
            out = gait_generator_g1(step, speed=1.0)
            assert np.all(np.abs(out) < np.pi + 0.1), \
                f"step={step}: out of joint range {out}"


class TestGo2GaitGenerator:
    """gait_generator_go2 — 12 维 trot 步态."""

    def test_default_dim_12(self):
        from collect_data import gait_generator_go2
        out = gait_generator_go2(0)
        assert out.shape == (12,)
        assert out.dtype == np.float32

    def test_trot_phase_diagonal_legs(self):
        """对角腿同步: FL 与 RR 同相, FR 与 RL 反相."""
        from collect_data import gait_generator_go2
        out = gait_generator_go2(5, speed=1.0)  # t=0.1, phase≈0.95, 非零
        # FL thigh (idx 1) vs RR thigh (idx 10) 应同相 (默认值相同 + 同相扰动)
        # FR thigh (idx 4) vs RL thigh (idx 7) 应同相
        # FL vs FR 应反相 (相对 default 的偏移)
        fl_thigh_offset = out[1] - out[10]
        fr_thigh_offset = out[4] - out[7]
        assert abs(fl_thigh_offset) < 1e-4, f"FL-RR 偏移不同: {fl_thigh_offset}"
        assert abs(fr_thigh_offset) < 1e-4, f"FR-RL 偏移不同: {fr_thigh_offset}"

    def test_default_stance_in_mjlab_range(self):
        """Go2 站立姿态在 mjlab 物理合理范围 (hip ±0.2, thigh 0.7-1.0, calf -2.0 ~ -1.5).

        与 mjlab 官方 go2_constants.py 一致
        与 go2_config.GO2_DEFAULT_JOINT_ANGLES 不一致 (FL_hip 应 -0.1 而非 0.0, RL/RR_thigh 应 0.9 而非 1.0)
        修复后 gait_generator_go2 直接从 config 读默认姿态.
        """
        from collect_data import gait_generator_go2
        from configs.go2_config import GO2_DEFAULT_JOINT_ANGLES, GO2_JOINT_NAMES
        out = gait_generator_go2(0)  # t=0, phase=0, 无扰动
        # 应当完全等于 config 中的默认姿态 (因为 t=0 时扰动项都是 0)
        expected = np.array(
            [GO2_DEFAULT_JOINT_ANGLES[n] for n in GO2_JOINT_NAMES],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(out, expected, decimal=6)
        # sanity: hip 在 ±0.2 内, thigh ∈ [0.7, 1.0], calf ∈ [-2.0, -1.5]
        assert abs(out[0]) < 0.2
        assert 0.7 <= out[1] <= 1.0
        assert -2.0 <= out[2] <= -1.5


class TestGenericGaitGenerator:
    """gait_generator_generic — 用于 A2/R1/H1_2/H2/As2."""

    def test_default_zero_angles(self):
        from collect_data import gait_generator_generic
        out = gait_generator_generic(0, num_joints=17)
        assert out.shape == (17,)

    def test_with_custom_default(self):
        from collect_data import gait_generator_generic
        defaults = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        out = gait_generator_generic(0, num_joints=3, default_angles=defaults)
        assert out.shape == (3,)
        # 初始应当接近 defaults + 扰动
        assert np.allclose(out, defaults, atol=0.15)


class TestGaitGeneratorRegistry:
    """GAIT_GENERATORS 字典完整性."""

    def test_g1_and_go2_registered(self):
        from collect_data import GAIT_GENERATORS
        assert "g1" in GAIT_GENERATORS
        assert "go2" in GAIT_GENERATORS

    def test_all_registered_callable(self):
        from collect_data import GAIT_GENERATORS
        for name, fn in GAIT_GENERATORS.items():
            assert callable(fn), f"{name} is not callable"


class TestTaskToRobotMapping:
    """TASK_TO_ROBOT — 任务 ID 解析为机器人类型."""

    def test_g1_tasks(self):
        from collect_data import TASK_TO_ROBOT
        assert TASK_TO_ROBOT["Unitree-G1-Flat"] == "g1"
        assert TASK_TO_ROBOT["Unitree-G1-Rough"] == "g1"

    def test_go2_tasks(self):
        from collect_data import TASK_TO_ROBOT
        assert TASK_TO_ROBOT["Unitree-Go2-Flat"] == "go2"
        assert TASK_TO_ROBOT["Unitree-Go2-Rough"] == "go2"

    def test_23dof_tasks_map_to_g1(self):
        from collect_data import TASK_TO_ROBOT
        # G1 23Dof 变种仍归类为 g1 (generate 29Dof by default 不对, 需 num_joints=23)
        assert TASK_TO_ROBOT["Unitree-G1-23Dof-Flat"] == "g1"
        assert TASK_TO_ROBOT["Unitree-G1-23Dof-Rough"] == "g1"

    def test_all_values_are_robot_types(self):
        """值必须在 GAIT_GENERATORS 中注册."""
        from collect_data import TASK_TO_ROBOT, GAIT_GENERATORS
        for task, robot in TASK_TO_ROBOT.items():
            assert robot in GAIT_GENERATORS, \
                f"Task {task} maps to {robot}, not in GAIT_GENERATORS"


class TestToNumpy:
    """_to_numpy — tensor/ndarray 转换 + batch 维 squeeze."""

    def test_none_passthrough(self):
        from collect_data import _to_numpy
        assert _to_numpy(None) is None

    def test_numpy_1d(self):
        from collect_data import _to_numpy
        result = _to_numpy(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_numpy_2d_squeeze_batch(self):
        """shape (1, D) → (D,)."""
        from collect_data import _to_numpy
        result = _to_numpy(np.array([[1.0, 2.0, 3.0]]))
        assert result.shape == (3,)

    def test_numpy_2d_keep_batch(self):
        """shape (B, D) with B>1 → 保留 batch 维."""
        from collect_data import _to_numpy
        result = _to_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])