"""retarget_motion_loader.py — 加载 robot_retargeter 产出的运动数据。"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class RetargetMotionLoader:
    """加载 robot_retargeter 的运动数据。

    Args:
        motion_file: 动作文件路径 (.csv / .npz)
        fps: 覆盖文件中的帧率
        quat_order: CSV 中四元数顺序，"xyzw" 或 "wxyz"（MuJoCo 原生 wxyz）
    """

    def __init__(
        self,
        motion_file: str,
        fps: Optional[float] = None,
        quat_order: str = "xyzw",
    ):
        self.motion_file = Path(motion_file)
        self.fps = fps
        if quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"quat_order 必须为 'xyzw' 或 'wxyz'，得到 {quat_order}")
        self.quat_order = quat_order

        if not self.motion_file.exists():
            raise FileNotFoundError(f"动作文件不存在: {self.motion_file}")

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        suffix = self.motion_file.suffix.lower()
        if suffix == ".csv":
            return self._load_csv()
        elif suffix == ".npz":
            return self._load_npz()
        else:
            raise ValueError(f"不支持的文件格式: {suffix}（支持 .csv, .npz）")

    def _load_csv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        motion = np.loadtxt(str(self.motion_file), delimiter=",", skiprows=1)
        if motion.ndim == 1:
            motion = motion[None, :]

        T = motion.shape[0]
        fps = self.fps or 30.0

        base_pos = motion[:, 0:3].astype(np.float32)
        base_quat_raw = motion[:, 3:7].astype(np.float32)
        joint_pos = motion[:, 7:].astype(np.float32)

        # 统一转为 wxyz（GR00T/MuJoCo 标准）
        if self.quat_order == "wxyz":
            base_quat = base_quat_raw.copy()
        else:  # xyzw → wxyz
            base_quat = base_quat_raw[:, [3, 0, 1, 2]]

        norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
        # 全零四元数（不应出现）归一化为单位四元数 [1,0,0,0]
        zero_mask = norms[:, 0] < 1e-6
        norms = np.maximum(norms, 1e-8)
        base_quat = base_quat / norms
        if zero_mask.any():
            base_quat[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        print(f"  📂 CSV 加载: {self.motion_file.name}")
        print(f"     帧数: {T}, FPS: {fps}")
        print(f"     base_pos: {base_pos.shape}, base_quat: {base_quat.shape}, joint_pos: {joint_pos.shape}")

        return base_pos, base_quat, joint_pos, fps

    def _load_npz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        data = np.load(str(self.motion_file))

        fps_data = data.get("fps", np.array([30.0]))
        # 统一 fps 解析：兼容 0-d / 1-d / 标量
        fps_arr = np.asarray(fps_data).ravel()
        fps = self.fps or float(fps_arr[0]) if fps_arr.size > 0 else 30.0
        joint_pos = data["joint_pos"].astype(np.float32)

        T = joint_pos.shape[0]

        if "body_pos_w" in data and "body_quat_w" in data:
            body_pos_w = data["body_pos_w"]
            assert body_pos_w.ndim == 3, f"body_pos_w 应为 (T, num_bodies, 3)，实际 ndim={body_pos_w.ndim}"
            base_pos = body_pos_w[:, 0, :].astype(np.float32)
            base_quat = data["body_quat_w"][:, 0, :].astype(np.float32)
        else:
            base_pos = np.zeros((T, 3), dtype=np.float32)
            base_pos[:, 2] = 0.8
            base_quat = np.zeros((T, 4), dtype=np.float32)
            base_quat[:, 0] = 1.0

        norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        base_quat = base_quat / norms

        print(f"  📂 NPZ 加载: {self.motion_file.name}")
        print(f"     帧数: {T}, FPS: {fps}")
        print(f"     base_pos: {base_pos.shape}, base_quat: {base_quat.shape}, joint_pos: {joint_pos.shape}")

        return base_pos, base_quat, joint_pos, fps


def load_motion(motion_file: str, fps: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loader = RetargetMotionLoader(motion_file, fps=fps)
    return loader.load()
