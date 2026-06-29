"""
retarget_motion_loader.py — 加载 robot_retargeter 产出的运动数据。

支持两种输入格式：
  1. qpos CSV: [pos_xyz(3), quat_xyzw(4), joints(N)] — robot_retargeter 直接输出
  2. NPZ: joint_pos, joint_vel, body_pos_w, body_quat_w, ... — export_npz.py 输出

统一输出：(base_pos, base_quat, joint_pos, fps)

用法:
    loader = RetargetMotionLoader("output_data/robot_motion/xxx_g1.csv")
    base_pos, base_quat, joint_pos, fps = loader.load()
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class RetargetMotionLoader:
    """加载 robot_retargeter 的运动数据。"""

    def __init__(
        self,
        motion_file: str,
        fps: Optional[float] = None,
    ):
        """
        Args:
            motion_file: CSV 或 NPZ 文件路径
            fps: 帧率（CSV 模式下默认 30，NPZ 模式下从文件读取）
        """
        self.motion_file = Path(motion_file)
        self.fps = fps

        if not self.motion_file.exists():
            raise FileNotFoundError(f"动作文件不存在: {self.motion_file}")

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        加载运动数据。

        Returns:
            base_pos: (T, 3) 基座位置
            base_quat: (T, 4) 基座四元数 (wxyz)
            joint_pos: (T, N) 关节位置（弧度）
            fps: 帧率
        """
        suffix = self.motion_file.suffix.lower()
        if suffix == ".csv":
            return self._load_csv()
        elif suffix == ".npz":
            return self._load_npz()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}（支持 .csv, .npz）")

    def _load_csv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """加载 qpos CSV 格式。

        CSV 格式：[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_0, ..., joint_N]
        """
        motion = np.loadtxt(str(self.motion_file), delimiter=",")
        if motion.ndim == 1:
            motion = motion[None, :]

        T = motion.shape[0]
        fps = self.fps or 30.0

        # 解析：前 3 列 = base_pos，接下来 4 列 = base_quat (xyzw)，剩余 = joints
        base_pos = motion[:, 0:3].astype(np.float32)
        base_quat_xyzw = motion[:, 3:7].astype(np.float32)
        joint_pos = motion[:, 7:].astype(np.float32)

        # 转换四元数 xyzw → wxyz（GR00T/MuJoCo 标准）
        base_quat = base_quat_xyzw[:, [3, 0, 1, 2]]

        # 归一化四元数
        norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        base_quat = base_quat / norms

        print(f"  📂 CSV 加载: {self.motion_file.name}")
        print(f"     帧数: {T}, FPS: {fps}")
        print(f"     base_pos: {base_pos.shape}, base_quat: {base_quat.shape}, joint_pos: {joint_pos.shape}")

        return base_pos, base_quat, joint_pos, fps

    def _load_npz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """加载 NPZ 格式（export_npz.py 输出）。

        NPZ 包含：joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps
        """
        data = np.load(str(self.motion_file))

        fps_data = data.get("fps", np.array([30.0]))
        fps = self.fps or float(fps_data.item() if hasattr(fps_data, "item") else fps_data)
        joint_pos = data["joint_pos"].astype(np.float32)  # (T, N)

        T = joint_pos.shape[0]

        # 尝试获取 base 信息
        if "body_pos_w" in data and "body_quat_w" in data:
            # body_pos_w: (T, B, 3), body_quat_w: (T, B, 4) wxyz
            # 第一个 body 通常是 pelvis/root
            base_pos = data["body_pos_w"][:, 0, :].astype(np.float32)  # (T, 3)
            base_quat = data["body_quat_w"][:, 0, :].astype(np.float32)  # (T, 4) wxyz
        else:
            # 没有 body 信息，使用默认值
            base_pos = np.zeros((T, 3), dtype=np.float32)
            base_pos[:, 2] = 0.8  # 假设基座高度 0.8m
            base_quat = np.zeros((T, 4), dtype=np.float32)
            base_quat[:, 0] = 1.0  # w=1, xyz=0

        # 归一化四元数
        norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        base_quat = base_quat / norms

        print(f"  📂 NPZ 加载: {self.motion_file.name}")
        print(f"     帧数: {T}, FPS: {fps}")
        print(f"     base_pos: {base_pos.shape}, base_quat: {base_quat.shape}, joint_pos: {joint_pos.shape}")

        return base_pos, base_quat, joint_pos, fps


def load_motion(motion_file: str, fps: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    便捷函数：加载运动数据。

    Args:
        motion_file: CSV 或 NPZ 文件路径
        fps: 帧率

    Returns:
        base_pos: (T, 3)
        base_quat: (T, 4) wxyz
        joint_pos: (T, N)
        fps: 帧率
    """
    loader = RetargetMotionLoader(motion_file, fps=fps)
    return loader.load()
