"""
mujoco_renderer.py — MuJoCo 离线渲染器，用于生成训练所需的相机图像。

加载机器人 MJCF 模型，设置相机视角，逐帧渲染关节轨迹 → 输出 mp4 视频。

用法:
    renderer = MujocoRenderer(mjcf_path="path/to/g1.xml", robot="g1")
    renderer.render_motion(
        joint_pos=joint_pos,      # (T, 29)
        base_pos=base_pos,        # (T, 3)
        base_quat=base_quat,      # (T, 4) wxyz
        output_path="output.mp4",
        camera_name="front",
        fps=30,
    )
"""

import os
from pathlib import Path
from typing import Optional

# 在无头服务器上强制使用 osmesa 后端（必须在 import mujoco 之前）
if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np


class MujocoRenderer:

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        robot: str = "g1",
        image_size: tuple = (224, 224),
    ):
        """
        Args:
            mjcf_path: MJCF 模型文件路径（None 则自动查找）
            robot: 机器人类型（用于自动查找模型）
            image_size: 渲染图像尺寸 (H, W)
        """
        self.robot = robot
        self.image_size = image_size

        if mjcf_path is None:
            mjcf_path = self._find_robot_mjcf(robot)

        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF 模型不存在: {self.mjcf_path}")

        self._setup_mujoco()

    def _find_robot_mjcf(self, robot: str) -> str:
        """自动查找 robot_retargeter 项目中的 MJCF 模型。"""
        # __file__ = .../gr00t_mjlab_autodl/src/mujoco_renderer.py
        # parent.parent = .../gr00t_mjlab_autodl/
        search_roots = [
            Path(__file__).resolve().parent.parent,  # gr00t_mjlab_autodl/
            Path(__file__).resolve().parent.parent.parent,  # unitree/
            Path.home() / "work",
        ]

        candidates = []
        if robot == "go2":
            candidates = [
                "unitree/robot_retargeter/asset/robot/a2_description/a2.xml",
                "robot_retargeter/asset/robot/a2_description/a2.xml",
            ]
        elif robot == "h1":
            candidates = [
                "unitree/robot_retargeter/asset/robot/h1_description/mjcf/h1.xml",
                "robot_retargeter/asset/robot/h1_description/mjcf/h1.xml",
            ]
        elif robot == "h1_2":
            candidates = [
                "unitree/robot_retargeter/asset/robot/h1_2_description/h1_2.xml",
                "robot_retargeter/asset/robot/h1_2_description/h1_2.xml",
            ]
        elif robot == "h2":
            candidates = [
                "unitree/robot_retargeter/asset/robot/h2_description/H2.xml",
                "robot_retargeter/asset/robot/h2_description/H2.xml",
            ]
        else:  # g1 (默认)
            # 优先使用 mjlab 的 G1 模型（包含 tracking 相机）
            candidates = [
                ".venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml",
                "unitree/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml",
                "unitree/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
                "robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
            ]

        for root in search_roots:
            for rel in candidates:
                full_path = root / rel
                if full_path.exists():
                    return str(full_path)

        raise FileNotFoundError(
            f"无法找到 {robot} 的 MJCF 模型。请指定 mjcf_path 参数。"
        )

    def _setup_mujoco(self):
        """初始化 MuJoCo 模型和数据。"""
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        # 设置渲染器
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.image_size[0],
            width=self.image_size[1],
        )

        # 获取相机名称
        self.camera_names = []
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self.camera_names.append(name)

        print(f"  ✅ MuJoCo 初始化: {self.mjcf_path.name}")
        print(f"     图像尺寸: {self.image_size}")
        print(f"     可用相机: {self.camera_names}")

    def render_motion(
        self,
        joint_pos: np.ndarray,
        output_path: str,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        camera_name: Optional[str] = None,
        fps: float = 30.0,
    ):
        """
        渲染关节轨迹为视频。

        Args:
            joint_pos: (T, N) 关节位置
            output_path: 输出 mp4 路径
            base_pos: (T, 3) 基座位置（可选）
            base_quat: (T, 4) 基座四元数 wxyz（可选）
            camera_name: 相机名称（None 则使用第一个相机）
            fps: 帧率
        """
        import cv2
        import mujoco

        T = joint_pos.shape[0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 设置相机
        if camera_name is None:
            if self.camera_names:
                camera_name = self.camera_names[0]
            else:
                camera_name = None  # 使用默认相机

        # 创建视频写入器
        h, w = self.image_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {output_path}")

        # 获取 free joint 和关节的 qpos 地址
        # MuJoCo 的 qpos 布局：[free_joint_pos(3), free_joint_quat(4), joint_0, joint_1, ...]
        # free joint 通常是第一个 joint
        num_joints = joint_pos.shape[1]

        print(f"  🎬 渲染视频: {T} 帧 → {output_path.name}")

        for t in range(T):
            # 设置基座状态
            if base_pos is not None:
                self.data.qpos[0:3] = base_pos[t]
            if base_quat is not None:
                self.data.qpos[3:7] = base_quat[t]  # wxyz

            # 设置关节位置（从第 7 个 qpos 开始）
            self.data.qpos[7:7 + num_joints] = joint_pos[t]

            # 零速度
            self.data.qvel[:] = 0.0

            # 前向运动学
            mujoco.mj_forward(self.model, self.data)

            # 渲染
            self.renderer.update_scene(self.data, camera=camera_name)
            img = self.renderer.render()

            # RGB → BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

            if (t + 1) % 100 == 0 or t == T - 1:
                print(f"     渲染进度: {t + 1}/{T}")

        writer.release()
        print(f"  ✅ 视频保存: {output_path}")

    def render_motion_multicam(
        self,
        joint_pos: np.ndarray,
        output_dir: str,
        camera_names: list,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ):
        """
        多相机渲染，每个相机一个视频。

        Args:
            joint_pos: (T, N) 关节位置
            output_dir: 输出目录
            camera_names: 相机名称列表
            base_pos: (T, 3) 基座位置
            base_quat: (T, 4) 基座四元数 wxyz
            fps: 帧率
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cam_name in camera_names:
            output_path = output_dir / f"{cam_name}.mp4"
            self.render_motion(
                joint_pos=joint_pos,
                output_path=str(output_path),
                base_pos=base_pos,
                base_quat=base_quat,
                camera_name=cam_name,
                fps=fps,
            )


def render_motion_to_video(
    mjcf_path: str,
    joint_pos: np.ndarray,
    output_path: str,
    base_pos: Optional[np.ndarray] = None,
    base_quat: Optional[np.ndarray] = None,
    camera_name: Optional[str] = None,
    fps: float = 30.0,
    image_size: tuple = (224, 224),
):
    """
    便捷函数：渲染关节轨迹为视频。

    Args:
        mjcf_path: MJCF 模型路径
        joint_pos: (T, N) 关节位置
        output_path: 输出 mp4 路径
        base_pos: (T, 3) 基座位置
        base_quat: (T, 4) 基座四元数 wxyz
        camera_name: 相机名称
        fps: 帧率
        image_size: 图像尺寸 (H, W)
    """
    renderer = MujocoRenderer(mjcf_path=mjcf_path, image_size=image_size)
    renderer.render_motion(
        joint_pos=joint_pos,
        output_path=output_path,
        base_pos=base_pos,
        base_quat=base_quat,
        camera_name=camera_name,
        fps=fps,
    )
