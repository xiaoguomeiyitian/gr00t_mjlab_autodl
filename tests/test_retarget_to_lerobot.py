"""
test_retarget_to_lerobot.py — 测试 robot_retargeter → LeRobot v2 转换。

覆盖：
- RetargetMotionLoader: CSV/NPZ 加载
- compute_state_vector: 状态向量计算
- compute_delta_actions: delta 动作计算
- compute_angular_velocity: 角速度计算
- sliding_window_split: 滑动窗口切分
- get_motion_label: 动作标签推断
- convert_to_lerobot: 端到端转换
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retarget_motion_loader import RetargetMotionLoader
from src.retarget_to_lerobot import (
    compute_state_vector,
    compute_delta_actions,
    compute_angular_velocity,
    sliding_window_split,
    convert_to_lerobot,
)
from src.configs.motion_labels import get_motion_label, LABEL_MAP


# ─── Fixtures ───

@pytest.fixture
def sample_csv_file():
    """创建临时 CSV 文件（robot_retargeter qpos 格式）。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # 格式：[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_0..joint_28]
        T = 100
        for t in range(T):
            pos = [0.0, 0.0, 0.8]
            quat = [0.0, 0.0, 0.0, 1.0]  # xyzw
            joints = [np.sin(0.1 * t + i * 0.1) * 0.3 for i in range(29)]
            line = ", ".join(map(str, pos + quat + joints))
            f.write(line + "\n")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_npz_file():
    """创建临时 NPZ 文件（export_npz.py 格式）。"""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        T = 100
        data = {
            "joint_pos": np.random.randn(T, 29).astype(np.float32) * 0.3,
            "joint_vel": np.random.randn(T, 29).astype(np.float32) * 0.1,
            "body_pos_w": np.random.randn(T, 14, 3).astype(np.float32),
            "body_quat_w": np.random.randn(T, 14, 4).astype(np.float32),
            "body_lin_vel_w": np.random.randn(T, 14, 3).astype(np.float32),
            "body_ang_vel_w": np.random.randn(T, 14, 3).astype(np.float32),
            "fps": np.array([30.0]),
        }
        np.savez(f, **data)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_output_dir():
    """创建临时输出目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ─── Tests: RetargetMotionLoader ───

class TestRetargetMotionLoader:
    def test_load_csv(self, sample_csv_file):
        """测试 CSV 加载。"""
        loader = RetargetMotionLoader(sample_csv_file, fps=30.0)
        base_pos, base_quat, joint_pos, fps = loader.load()

        assert base_pos.shape == (100, 3)
        assert base_quat.shape == (100, 4)
        assert joint_pos.shape == (100, 29)
        assert fps == 30.0

    def test_load_csv_quaternion_wxyz(self, sample_csv_file):
        """测试四元数转换为 wxyz。"""
        loader = RetargetMotionLoader(sample_csv_file, fps=30.0)
        _, base_quat, _, _ = loader.load()

        # 输入是 [0,0,0,1] (xyzw)，输出应该是 [1,0,0,0] (wxyz)
        np.testing.assert_allclose(base_quat[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_load_npz(self, sample_npz_file):
        """测试 NPZ 加载。"""
        loader = RetargetMotionLoader(sample_npz_file)
        base_pos, base_quat, joint_pos, fps = loader.load()

        assert base_pos.shape == (100, 3)
        assert base_quat.shape == (100, 4)
        assert joint_pos.shape == (100, 29)
        assert fps == 30.0

    def test_file_not_found(self):
        """测试文件不存在时抛出异常。"""
        with pytest.raises(FileNotFoundError):
            RetargetMotionLoader("nonexistent.csv")

    def test_unsupported_format(self):
        """测试不支持的文件格式。"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                RetargetMotionLoader(f.name).load()


# ─── Tests: compute_state_vector ───

class TestComputeStateVector:
    def test_output_shape(self):
        """测试输出形状。"""
        T = 100
        base_pos = np.random.randn(T, 3).astype(np.float32)
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0  # w=1
        joint_pos = np.random.randn(T, 29).astype(np.float32)

        states = compute_state_vector(base_pos, base_quat, joint_pos, dt=1 / 30.0)

        assert states.shape == (100, 71)  # 29 + 29 + 3 + 4 + 3 + 3

    def test_state_content(self):
        """测试状态向量内容正确。"""
        T = 10
        base_pos = np.zeros((T, 3), dtype=np.float32)
        base_pos[:, 2] = 0.8  # z=0.8
        base_quat = np.zeros((T, 4), dtype=np.float32)
        base_quat[:, 0] = 1.0
        joint_pos = np.ones((T, 29), dtype=np.float32) * 0.5

        states = compute_state_vector(base_pos, base_quat, joint_pos, dt=1 / 30.0)

        # joint_pos 应该是 0.5
        np.testing.assert_allclose(states[0, :29], 0.5, atol=1e-5)
        # base_pos z 应该是 0.8
        assert states[0, 58 + 2] == pytest.approx(0.8, abs=1e-5)
        # base_quat w 应该是 1.0
        assert states[0, 61 + 0] == pytest.approx(1.0, abs=1e-5)


# ─── Tests: compute_delta_actions ───

class TestComputeDeltaActions:
    def test_output_shape(self):
        """测试输出形状。"""
        T = 100
        joint_pos = np.random.randn(T, 29).astype(np.float32)
        actions = compute_delta_actions(joint_pos)
        assert actions.shape == (100, 29)

    def test_delta_calculation(self):
        """测试 delta 计算正确。"""
        T = 5
        joint_pos = np.array([
            [0.0, 0.0],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.6, 0.8],
            [1.0, 1.2],
        ], dtype=np.float32)

        actions = compute_delta_actions(joint_pos)

        # action[0] = joint_pos[1] - joint_pos[0] = [0.1, 0.2]
        np.testing.assert_allclose(actions[0], [0.1, 0.2], atol=1e-6)
        # action[1] = joint_pos[2] - joint_pos[1] = [0.2, 0.2]
        np.testing.assert_allclose(actions[1], [0.2, 0.2], atol=1e-6)
        # 最后一帧复制前一帧
        np.testing.assert_allclose(actions[-1], actions[-2], atol=1e-6)


# ─── Tests: compute_angular_velocity ───

class TestComputeAngularVelocity:
    def test_zero_rotation(self):
        """测试无旋转时角速度为 0。"""
        T = 10
        quat = np.zeros((T, 4), dtype=np.float32)
        quat[:, 0] = 1.0  # w=1, 无旋转

        ang_vel = compute_angular_velocity(quat, dt=1 / 30.0)

        np.testing.assert_allclose(ang_vel, 0.0, atol=1e-5)

    def test_output_shape(self):
        """测试输出形状。"""
        T = 10
        quat = np.zeros((T, 4), dtype=np.float32)
        quat[:, 0] = 1.0

        ang_vel = compute_angular_velocity(quat, dt=1 / 30.0)

        assert ang_vel.shape == (T, 3)


# ─── Tests: sliding_window_split ───

class TestSlidingWindowSplit:
    def test_basic_split(self):
        """测试基本切分。"""
        episodes = sliding_window_split(1000, window_size=300, overlap=0.5)
        assert len(episodes) > 0
        for start, end in episodes:
            assert end - start == 300
            assert start >= 0
            assert end <= 1000

    def test_overlap(self):
        """测试重叠。"""
        episodes = sliding_window_split(1000, window_size=300, overlap=0.5)
        if len(episodes) >= 2:
            # 第二个 episode 的 start 应该在第一个的中间
            assert episodes[1][0] == episodes[0][0] + 150  # 300 * 0.5

    def test_short_sequence(self):
        """测试短序列（自动调整为整个序列作为一个 episode）。"""
        episodes = sliding_window_split(100, window_size=300, overlap=0.5)
        # 短序列自动调整为整个序列作为一个 episode
        assert len(episodes) == 1
        assert episodes[0] == (0, 100)

    def test_exact_fit(self):
        """测试恰好匹配。"""
        episodes = sliding_window_split(600, window_size=300, overlap=0.0)
        assert len(episodes) == 2
        assert episodes[0] == (0, 300)
        assert episodes[1] == (300, 600)


# ─── Tests: get_motion_label ───

class TestGetMotionLabel:
    def test_walk(self):
        assert get_motion_label("walk1_subject2.csv") == "walk forward"

    def test_run(self):
        assert get_motion_label("run1_subject5.csv") == "run forward"

    def test_dance(self):
        assert get_motion_label("dance1_subject3.csv") == "perform dancing motion"

    def test_fight(self):
        assert get_motion_label("fight1_subject2.csv") == "perform fighting motion"

    def test_jump(self):
        assert get_motion_label("jumps1_subject1.csv") == "jump up repeatedly"

    def test_fall(self):
        assert get_motion_label("fallAndGetUp1_subject4.csv") == "fall and get up"

    def test_unknown(self):
        assert get_motion_label("unknown_motion.csv") == "perform the locomotion task"

    def test_custom_default(self):
        assert get_motion_label("xyz.csv", default="custom task") == "custom task"


# ─── Tests: convert_to_lerobot (端到端) ───

class TestConvertToLeRobot:
    def test_csv_to_lerobot(self, sample_csv_file, temp_output_dir):
        """测试 CSV → LeRobot v2 完整转换。"""
        output_dir = os.path.join(temp_output_dir, "g1_test")

        convert_to_lerobot(
            motion_file=sample_csv_file,
            robot="g1",
            output_dir=output_dir,
            episode_length=30,
            overlap=0.5,
            fps=30.0,
            task_description="test walking",
            render_videos=False,  # 跳过视频渲染加速测试
        )

        # 检查输出目录结构
        assert os.path.exists(os.path.join(output_dir, "meta"))
        assert os.path.exists(os.path.join(output_dir, "data", "chunk-000"))
        assert os.path.exists(os.path.join(output_dir, "videos", "chunk-000"))

        # 检查 meta 文件
        assert os.path.exists(os.path.join(output_dir, "meta", "info.json"))
        assert os.path.exists(os.path.join(output_dir, "meta", "episodes.jsonl"))
        assert os.path.exists(os.path.join(output_dir, "meta", "tasks.jsonl"))
        assert os.path.exists(os.path.join(output_dir, "meta", "modality.json"))

        # 检查 info.json
        with open(os.path.join(output_dir, "meta", "info.json")) as f:
            info = json.load(f)
        assert info["robot_type"] == "g1"
        assert info["total_episodes"] > 0
        assert info["fps"] == 30.0

        # 检查 modality.json
        with open(os.path.join(output_dir, "meta", "modality.json")) as f:
            modality = json.load(f)
        assert "state" in modality
        assert "action" in modality
        assert modality["state"]["joint_pos"] == {"start": 0, "end": 29}
        assert modality["action"]["joint_position_delta"] == {"start": 0, "end": 29}

        # 检查 parquet
        parquet_path = os.path.join(output_dir, "data", "chunk-000", "episode_000000.parquet")
        assert os.path.exists(parquet_path)
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        assert "observation.state" in df.columns
        assert "action" in df.columns

    def test_npz_to_lerobot(self, sample_npz_file, temp_output_dir):
        """测试 NPZ → LeRobot v2 完整转换。"""
        output_dir = os.path.join(temp_output_dir, "g1_npz_test")

        convert_to_lerobot(
            motion_file=sample_npz_file,
            robot="g1",
            output_dir=output_dir,
            episode_length=30,
            overlap=0.5,
            render_videos=False,
        )

        assert os.path.exists(os.path.join(output_dir, "meta", "info.json"))
        with open(os.path.join(output_dir, "meta", "info.json")) as f:
            info = json.load(f)
        assert info["total_episodes"] > 0

    def test_auto_task_label(self, sample_csv_file, temp_output_dir):
        """测试自动推断任务标签。"""
        output_dir = os.path.join(temp_output_dir, "g1_auto_task")

        # 使用包含 "walk" 的文件名
        walk_csv = os.path.join(temp_output_dir, "walk1_subject1.csv")
        os.symlink(sample_csv_file, walk_csv)

        convert_to_lerobot(
            motion_file=walk_csv,
            robot="g1",
            output_dir=output_dir,
            episode_length=30,
            overlap=0.5,
            render_videos=False,
        )

        with open(os.path.join(output_dir, "meta", "tasks.jsonl")) as f:
            task_line = f.readline()
            task = json.loads(task_line)
        assert task["task_description"] == "walk forward"

        os.unlink(walk_csv)
