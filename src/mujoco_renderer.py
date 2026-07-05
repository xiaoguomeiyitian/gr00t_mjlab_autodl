"""mujoco_renderer.py — MuJoCo 离线渲染器。"""

import os
from pathlib import Path
from typing import Optional

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
        self.robot = robot
        self.image_size = image_size

        if mjcf_path is None:
            mjcf_path = self._find_robot_mjcf(robot)

        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF 模型不存在: {self.mjcf_path}")

        self._setup_mujoco()

    def _find_robot_mjcf(self, robot: str) -> str:
        search_roots = [
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve().parent.parent.parent,
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
        else:
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

        raise FileNotFoundError(f"无法找到 {robot} 的 MJCF 模型。请指定 mjcf_path 参数。")

    def _setup_mujoco(self):
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(
            self.model,
            height=self.image_size[0],
            width=self.image_size[1],
        )

        self.camera_names = []
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                self.camera_names.append(name)

        # 动态计算关节 qpos/qvel 地址，避免硬编码 qpos[7:]
        self.joint_qpos_indices = []
        self.has_free_base = False
        self.base_qpos_start = 0
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type == 0:  # mjJNT_FREE
                self.has_free_base = True
                self.base_qpos_start = self.model.jnt_qposadr[i]
            elif jnt_type in (2, 3):  # slide / hinge
                self.joint_qpos_indices.append(self.model.jnt_qposadr[i])

        print(f"  ✅ MuJoCo 初始化: {self.mjcf_path.name}")
        print(f"     图像尺寸: {self.image_size}")
        print(f"     可用相机: {self.camera_names}")
        print(f"     free base: {self.has_free_base}, 关节数: {len(self.joint_qpos_indices)}")

    def render_motion(
        self,
        joint_pos: np.ndarray,
        output_path: str,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        camera_name: Optional[str] = None,
        fps: float = 30.0,
    ):
        import cv2
        import mujoco

        T = joint_pos.shape[0]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if camera_name is None:
            camera_name = self.camera_names[0] if self.camera_names else None

        h, w = self.image_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {output_path}")

        num_joints = joint_pos.shape[1]
        if num_joints > len(self.joint_qpos_indices):
            raise ValueError(
                f"joint_pos 列数 {num_joints} 超过模型可写关节数 {len(self.joint_qpos_indices)}"
            )

        print(f"  🎬 渲染视频: {T} 帧 → {output_path.name}")

        for t in range(T):
            if self.has_free_base:
                if base_pos is not None:
                    self.data.qpos[self.base_qpos_start:self.base_qpos_start + 3] = base_pos[t]
                if base_quat is not None:
                    self.data.qpos[self.base_qpos_start + 3:self.base_qpos_start + 7] = base_quat[t]

            for i in range(num_joints):
                self.data.qpos[self.joint_qpos_indices[i]] = joint_pos[t, i]

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            self.renderer.update_scene(self.data, camera=camera_name)
            img = self.renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

            if (t + 1) % 100 == 0 or t == T - 1:
                print(f"     渲染进度: {t + 1}/{T}", flush=True)

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
        import cv2
        import mujoco

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        T = joint_pos.shape[0]
        num_joints = joint_pos.shape[1]
        if num_joints > len(self.joint_qpos_indices):
            raise ValueError(
                f"joint_pos 列数 {num_joints} 超过模型可写关节数 {len(self.joint_qpos_indices)}"
            )

        h, w = self.image_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {}
        for cam_name in camera_names:
            out_path = output_dir / f"{cam_name}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建视频写入器: {out_path}")
            writers[cam_name] = writer

        print(f"  🎬 多相机渲染: {T} 帧 × {len(camera_names)} 相机")

        for t in range(T):
            if self.has_free_base:
                if base_pos is not None:
                    self.data.qpos[self.base_qpos_start:self.base_qpos_start + 3] = base_pos[t]
                if base_quat is not None:
                    self.data.qpos[self.base_qpos_start + 3:self.base_qpos_start + 7] = base_quat[t]

            for i in range(num_joints):
                self.data.qpos[self.joint_qpos_indices[i]] = joint_pos[t, i]

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            for cam_name, writer in writers.items():
                self.renderer.update_scene(self.data, camera=cam_name)
                img = self.renderer.render()
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if (t + 1) % 100 == 0 or t == T - 1:
                print(f"     渲染进度: {t + 1}/{T}", flush=True)

        for writer in writers.values():
            writer.release()
        print(f"  ✅ 多相机视频保存: {output_dir}")
