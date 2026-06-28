"""
viser_viewer.py — Viser 浏览器 3D 可视化客户端。

在浏览器中查看机器人模型和动作回放，无需桌面环境。

依赖: pip install viser mujoco
运行: python -m src.viz.viser_viewer --robot g1 --port 20006
  --robot 可选: g1, h1, h1_with_hand, h1_2, h2, go2

浏览器打开: http://localhost:20006
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np


def find_robot_mjcf(robot: str) -> Optional[str]:
    """自动查找 robot_retargeter 项目中的 MJCF 模型。"""
    search_roots = [
        Path(__file__).resolve().parent.parent.parent,  # gr00t_mjlab_autodl/
        Path(__file__).resolve().parent.parent.parent.parent,  # gr00t/
        Path.home() / "work",  # ~/work/
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
    elif robot == "h1_with_hand":
        candidates = [
            "unitree/robot_retargeter/asset/robot/h1_description/mjcf/h1_with_hand.xml",
            "robot_retargeter/asset/robot/h1_description/mjcf/h1_with_hand.xml",
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
        candidates = [
            "unitree/robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
            "robot_retargeter/asset/robot/g1_description/mjcf/g1.xml",
        ]
    for root in search_roots:
        for rel in candidates:
            p = root / rel
            if p.exists():
                return str(p)
    return None


class ViserViewer:
    """Viser 浏览器 3D 可视化。"""

    def __init__(self, port: int = 20006, mjcf_path: Optional[str] = None, robot: str = "g1"):
        """
        Args:
            port: Viser 服务端口
            mjcf_path: MuJoCo MJCF 模型文件路径
            robot: 机器人类型（g1/go2）
        """
        self.port = port
        self.mjcf_path = mjcf_path
        self.robot = robot
        self.server = None
        self.model = None
        self.data = None
        self._scene_handles = {}
        self._joint_handles = {}
        self._initialized = False

    def init(self):
        """初始化 Viser 服务器和 MuJoCo 模型。"""
        try:
            import viser
        except ImportError:
            print("❌ 未安装 viser，请运行: pip install viser")
            print("   或使用 MuJoCo 可视化: ./start.sh mujoco")
            raise SystemExit(1)

        self.server = viser.ViserServer(port=self.port)
        self.server.scene.set_up_direction("+z")

        # 添加地面网格（细线条）
        self.server.scene.add_grid(
            "ground",
            width=10,
            height=10,
            cell_size=0.5,
            cell_thickness=0.5,
            cell_color=(80, 80, 80),
            section_thickness=0.8,
            section_color=(50, 50, 50),
            position=(0, 0, -0.01),
            wxyz=(1, 0, 0, 0),
        )

        # 加载 MJCF（如果有）
        if self.mjcf_path and Path(self.mjcf_path).exists():
            self._load_mjcf(self.mjcf_path)
        else:
            # 自动查找 robot_retargeter 中的模型
            auto_mjcf = self._find_robot_mjcf()
            if auto_mjcf:
                self._load_mjcf(auto_mjcf)
            else:
                # 兜底：生成占位机器人
                self._create_placeholder_robot()

        self._initialized = True
        print(f"✅ Viser 服务器启动: http://localhost:{self.port}")

    def _find_robot_mjcf(self) -> Optional[str]:
        """自动查找 robot_retargeter 项目中的 MJCF 模型。"""
        return find_robot_mjcf(self.robot)

    def _create_placeholder_robot(self):
        """根据 robot 类型创建占位机器人模型并上传到 Viser。"""
        try:
            import mujoco
        except ImportError:
            print("  ⚠️  未安装 mujoco，使用简易几何体")
            self._create_simple_placeholder()
            return

        if self.robot == "go2":
            xml = self._go2_xml()
        elif self.robot == "h2":
            xml = self._h2_xml()
        else:
            # g1, h1, h1_with_hand, h1_2 都使用人形占位
            xml = self._g1_xml()

        try:
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            self._upload_meshes()
            print(f"  ✅ 占位 {self.robot} 模型加载完成")
            print(f"     关节数: {self.model.njnt}")
        except Exception as e:
            print(f"  ⚠️  占位模型加载失败: {e}")
            self._create_simple_placeholder()

    def _create_simple_placeholder(self):
        """创建最简占位几何体（无 mujoco 时使用）。"""
        # 躯干
        self.server.scene.add_box(
            "body/torso", dimensions=(0.2, 0.15, 0.35),
            position=(0, 0, 0.85), color=(0.2, 0.6, 1.0),
        )
        # 头部
        self.server.scene.add_icosphere(
            "body/head", radius=0.08,
            position=(0, 0, 1.1), color=(0.9, 0.9, 0.9),
        )
        # 左臂
        self.server.scene.add_cylinder(
            "body/left_arm", radius=0.03, height=0.3,
            position=(0.15, 0, 0.9), color=(0.2, 0.5, 0.9),
        )
        # 右臂
        self.server.scene.add_cylinder(
            "body/right_arm", radius=0.03, height=0.3,
            position=(-0.15, 0, 0.9), color=(0.2, 0.5, 0.9),
        )
        # 左腿
        self.server.scene.add_cylinder(
            "body/left_leg", radius=0.04, height=0.4,
            position=(0.08, 0, 0.45), color=(0.8, 0.2, 0.2),
        )
        # 右腿
        self.server.scene.add_cylinder(
            "body/right_leg", radius=0.04, height=0.4,
            position=(-0.08, 0, 0.45), color=(0.2, 0.8, 0.2),
        )
        print(f"  ✅ 简易占位几何体已添加")

    @staticmethod
    def _g1_xml() -> str:
        """G1 人形机器人简化 MJCF。"""
        return """
<mujoco model="g1_placeholder">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>

    <!-- 躯干 -->
    <body name="torso" pos="0 0 0.75">
      <geom type="box" size="0.1 0.08 0.18" rgba="0.2 0.6 1 1"/>
      <joint type="free" name="root"/>

      <!-- 头部 -->
      <body name="head" pos="0 0 0.25">
        <geom type="sphere" size="0.07" rgba="0.9 0.9 0.9 1"/>
        <joint type="hinge" axis="0 1 0" name="neck_pitch" range="-30 30"/>
      </body>

      <!-- 左肩 -->
      <body name="left_shoulder" pos="0.12 0 0.14">
        <geom type="sphere" size="0.04" rgba="0.2 0.5 0.9 1"/>
        <joint type="hinge" axis="1 0 0" name="left_shoulder_pitch" range="-180 60"/>
        <!-- 左上臂 -->
        <body name="left_upper_arm" pos="0 0 -0.08">
          <geom type="capsule" fromto="0 0 0 0 0 -0.14" size="0.025" rgba="0.2 0.5 0.9 1"/>
          <joint type="hinge" axis="0 1 0" name="left_elbow" range="0 150"/>
          <!-- 左前臂 -->
          <body name="left_forearm" pos="0 0 -0.14">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.02" rgba="0.3 0.6 1 1"/>
            <joint type="hinge" axis="0 0 1" name="left_wrist" range="-90 90"/>
          </body>
        </body>
      </body>

      <!-- 右肩 -->
      <body name="right_shoulder" pos="-0.12 0 0.14">
        <geom type="sphere" size="0.04" rgba="0.2 0.5 0.9 1"/>
        <joint type="hinge" axis="1 0 0" name="right_shoulder_pitch" range="-180 60"/>
        <!-- 右上臂 -->
        <body name="right_upper_arm" pos="0 0 -0.08">
          <geom type="capsule" fromto="0 0 0 0 0 -0.14" size="0.025" rgba="0.2 0.5 0.9 1"/>
          <joint type="hinge" axis="0 1 0" name="right_elbow" range="0 150"/>
          <!-- 右前臂 -->
          <body name="right_forearm" pos="0 0 -0.14">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.02" rgba="0.3 0.6 1 1"/>
            <joint type="hinge" axis="0 0 1" name="right_wrist" range="-90 90"/>
          </body>
        </body>
      </body>

      <!-- 腰部 -->
      <body name="waist" pos="0 0 -0.18">
        <geom type="box" size="0.09 0.06 0.06" rgba="0.3 0.3 0.3 1"/>
        <joint type="hinge" axis="0 0 1" name="waist_yaw" range="-90 90"/>

        <!-- 左髋 -->
        <body name="left_hip" pos="0.06 0 -0.06">
          <geom type="sphere" size="0.04" rgba="0.8 0.2 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="left_hip_pitch" range="-120 30"/>
          <!-- 左大腿 -->
          <body name="left_thigh" pos="0 0 -0.06">
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" rgba="0.8 0.2 0.2 1"/>
            <joint type="hinge" axis="1 0 0" name="left_knee" range="0 150"/>
            <!-- 左小腿 -->
            <body name="left_shin" pos="0 0 -0.2">
              <geom type="capsule" fromto="0 0 0 0 0 -0.18" size="0.025" rgba="0.9 0.3 0.3 1"/>
              <joint type="hinge" axis="1 0 0" name="left_ankle" range="-60 60"/>
              <!-- 左脚 -->
              <body name="left_foot" pos="0 0 -0.18">
                <geom type="box" size="0.05 0.03 0.1" rgba="0.4 0.4 0.4 1"/>
              </body>
            </body>
          </body>
        </body>

        <!-- 右髋 -->
        <body name="right_hip" pos="-0.06 0 -0.06">
          <geom type="sphere" size="0.04" rgba="0.2 0.8 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="right_hip_pitch" range="-120 30"/>
          <!-- 右大腿 -->
          <body name="right_thigh" pos="0 0 -0.06">
            <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" rgba="0.2 0.8 0.2 1"/>
            <joint type="hinge" axis="1 0 0" name="right_knee" range="0 150"/>
            <!-- 右小腿 -->
            <body name="right_shin" pos="0 0 -0.2">
              <geom type="capsule" fromto="0 0 0 0 0 -0.18" size="0.025" rgba="0.3 0.9 0.3 1"/>
              <joint type="hinge" axis="1 0 0" name="right_ankle" range="-60 60"/>
              <!-- 右脚 -->
              <body name="right_foot" pos="0 0 -0.18">
                <geom type="box" size="0.05 0.03 0.1" rgba="0.4 0.4 0.4 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    @staticmethod
    def _go2_xml() -> str:
        """Go2 四足机器人简化 MJCF。"""
        return """
<mujoco model="go2_placeholder">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>

    <!-- 躯干 -->
    <body name="trunk" pos="0 0 0.35">
      <geom type="box" size="0.2 0.08 0.05" rgba="0.2 0.6 1 1"/>
      <joint type="free" name="root"/>

      <!-- 前左腿 -->
      <body name="FL_hip" pos="0.18 0.08 0">
        <geom type="sphere" size="0.03" rgba="0.8 0.2 0.2 1"/>
        <joint type="hinge" axis="1 0 0" name="FL_hip_joint" range="-60 60"/>
        <body name="FL_thigh" pos="0 0 -0.04">
          <geom type="capsule" fromto="0 0 0 0 0 -0.13" size="0.02" rgba="0.8 0.2 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="FL_thigh_joint" range="-150 10"/>
          <body name="FL_calf" pos="0 0 -0.13">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.015" rgba="0.9 0.3 0.3 1"/>
            <joint type="hinge" axis="1 0 0" name="FL_calf_joint" range="-150 10"/>
            <body name="FL_foot" pos="0 0 -0.12">
              <geom type="sphere" size="0.02" rgba="0.4 0.4 0.4 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- 前右腿 -->
      <body name="FR_hip" pos="0.18 -0.08 0">
        <geom type="sphere" size="0.03" rgba="0.2 0.8 0.2 1"/>
        <joint type="hinge" axis="1 0 0" name="FR_hip_joint" range="-60 60"/>
        <body name="FR_thigh" pos="0 0 -0.04">
          <geom type="capsule" fromto="0 0 0 0 0 -0.13" size="0.02" rgba="0.2 0.8 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="FR_thigh_joint" range="-150 10"/>
          <body name="FR_calf" pos="0 0 -0.13">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.015" rgba="0.3 0.9 0.3 1"/>
            <joint type="hinge" axis="1 0 0" name="FR_calf_joint" range="-150 10"/>
            <body name="FR_foot" pos="0 0 -0.12">
              <geom type="sphere" size="0.02" rgba="0.4 0.4 0.4 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- 后左腿 -->
      <body name="RL_hip" pos="-0.18 0.08 0">
        <geom type="sphere" size="0.03" rgba="0.8 0.2 0.2 1"/>
        <joint type="hinge" axis="1 0 0" name="RL_hip_joint" range="-60 60"/>
        <body name="RL_thigh" pos="0 0 -0.04">
          <geom type="capsule" fromto="0 0 0 0 0 -0.13" size="0.02" rgba="0.8 0.2 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="RL_thigh_joint" range="-150 10"/>
          <body name="RL_calf" pos="0 0 -0.13">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.015" rgba="0.9 0.3 0.3 1"/>
            <joint type="hinge" axis="1 0 0" name="RL_calf_joint" range="-150 10"/>
            <body name="RL_foot" pos="0 0 -0.12">
              <geom type="sphere" size="0.02" rgba="0.4 0.4 0.4 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- 后右腿 -->
      <body name="RR_hip" pos="-0.18 -0.08 0">
        <geom type="sphere" size="0.03" rgba="0.2 0.8 0.2 1"/>
        <joint type="hinge" axis="1 0 0" name="RR_hip_joint" range="-60 60"/>
        <body name="RR_thigh" pos="0 0 -0.04">
          <geom type="capsule" fromto="0 0 0 0 0 -0.13" size="0.02" rgba="0.2 0.8 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="RR_thigh_joint" range="-150 10"/>
          <body name="RR_calf" pos="0 0 -0.13">
            <geom type="capsule" fromto="0 0 0 0 0 -0.12" size="0.015" rgba="0.3 0.9 0.3 1"/>
            <joint type="hinge" axis="1 0 0" name="RR_calf_joint" range="-150 10"/>
            <body name="RR_foot" pos="0 0 -0.12">
              <geom type="sphere" size="0.02" rgba="0.4 0.4 0.4 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    @staticmethod
    def _h2_xml() -> str:
        """H2 人形机器人简化 MJCF（比 H1 更粗壮）。"""
        return """
<mujoco model="h2_placeholder">
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>

    <!-- 躯干 -->
    <body name="torso" pos="0 0 0.85">
      <geom type="box" size="0.12 0.1 0.22" rgba="0.2 0.6 1 1"/>
      <joint type="free" name="root"/>

      <!-- 头部 -->
      <body name="head" pos="0 0 0.3">
        <geom type="sphere" size="0.09" rgba="0.9 0.9 0.9 1"/>
        <joint type="hinge" axis="0 1 0" name="neck_pitch" range="-30 30"/>
      </body>

      <!-- 左肩 -->
      <body name="left_shoulder" pos="0.15 0 0.18">
        <geom type="sphere" size="0.05" rgba="0.2 0.5 0.9 1"/>
        <joint type="hinge" axis="1 0 0" name="left_shoulder_pitch" range="-180 60"/>
        <body name="left_upper_arm" pos="0 0 -0.1">
          <geom type="capsule" fromto="0 0 0 0 0 -0.18" size="0.03" rgba="0.2 0.5 0.9 1"/>
          <joint type="hinge" axis="0 1 0" name="left_elbow" range="0 150"/>
          <body name="left_forearm" pos="0 0 -0.18">
            <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" rgba="0.3 0.6 1 1"/>
            <joint type="hinge" axis="0 0 1" name="left_wrist" range="-90 90"/>
          </body>
        </body>
      </body>

      <!-- 右肩 -->
      <body name="right_shoulder" pos="-0.15 0 0.18">
        <geom type="sphere" size="0.05" rgba="0.2 0.5 0.9 1"/>
        <joint type="hinge" axis="1 0 0" name="right_shoulder_pitch" range="-180 60"/>
        <body name="right_upper_arm" pos="0 0 -0.1">
          <geom type="capsule" fromto="0 0 0 0 0 -0.18" size="0.03" rgba="0.2 0.5 0.9 1"/>
          <joint type="hinge" axis="0 1 0" name="right_elbow" range="0 150"/>
          <body name="right_forearm" pos="0 0 -0.18">
            <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.025" rgba="0.3 0.6 1 1"/>
            <joint type="hinge" axis="0 0 1" name="right_wrist" range="-90 90"/>
          </body>
        </body>
      </body>

      <!-- 腰部 -->
      <body name="waist" pos="0 0 -0.22">
        <geom type="box" size="0.11 0.08 0.08" rgba="0.3 0.3 0.3 1"/>
        <joint type="hinge" axis="0 0 1" name="waist_yaw" range="-90 90"/>

        <!-- 左髋 -->
        <body name="left_hip" pos="0.07 0 -0.08">
          <geom type="sphere" size="0.05" rgba="0.8 0.2 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="left_hip_pitch" range="-120 30"/>
          <body name="left_thigh" pos="0 0 -0.08">
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.8 0.2 0.2 1"/>
            <joint type="hinge" axis="1 0 0" name="left_knee" range="0 150"/>
            <body name="left_shin" pos="0 0 -0.25">
              <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.03" rgba="0.9 0.3 0.3 1"/>
              <joint type="hinge" axis="1 0 0" name="left_ankle" range="-60 60"/>
              <body name="left_foot" pos="0 0 -0.22">
                <geom type="box" size="0.06 0.04 0.12" rgba="0.4 0.4 0.4 1"/>
              </body>
            </body>
          </body>
        </body>

        <!-- 右髋 -->
        <body name="right_hip" pos="-0.07 0 -0.08">
          <geom type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
          <joint type="hinge" axis="1 0 0" name="right_hip_pitch" range="-120 30"/>
          <body name="right_thigh" pos="0 0 -0.08">
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
            <joint type="hinge" axis="1 0 0" name="right_knee" range="0 150"/>
            <body name="right_shin" pos="0 0 -0.25">
              <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.03" rgba="0.3 0.9 0.3 1"/>
              <joint type="hinge" axis="1 0 0" name="right_ankle" range="-60 60"/>
              <body name="right_foot" pos="0 0 -0.22">
                <geom type="box" size="0.06 0.04 0.12" rgba="0.4 0.4 0.4 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    def _load_mjcf(self, mjcf_path: str):
        """加载 MuJoCo MJCF 模型。"""
        try:
            import mujoco

            spec = mujoco.MjSpec.from_file(mjcf_path)
            self.model = spec.compile()
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)

            # 渲染网格到 Viser
            self._upload_meshes()
            print(f"  ✅ MJCF 加载: {mjcf_path}")
            print(f"     关节数: {self.model.njnt}")
            print(f"     geom 数: {self.model.ngeom}")

        except Exception as e:
            print(f"  ⚠️  MJCF 加载失败: {e}")
            print(f"     将使用占位几何体")

    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> tuple:
        """将 3x3 旋转矩阵转换为四元数 (w, x, y, z)。"""
        # 使用 Shepperd 方法
        m = mat
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return (float(w), float(x), float(y), float(z))

    def _upload_meshes(self):
        """将 MuJoCo 几何体上传到 Viser 场景。"""
        import mujoco

        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            geom_type = self.model.geom_type[i]
            pos = self.data.geom_xpos[i]
            mat = self.data.geom_xmat[i].reshape(3, 3)
            size = self.model.geom_size[i]
            rgba = self.model.geom_rgba[i]

            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # 加载 mesh 数据
                meshid = self.model.geom_dataid[i]
                mesh_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, meshid)
                vertadr = self.model.mesh_vertadr[meshid]
                vertnum = self.model.mesh_vertnum[meshid]
                faceadr = self.model.mesh_faceadr[meshid]
                facenum = self.model.mesh_facenum[meshid]
                verts = self.model.mesh_vert[vertadr:vertadr+vertnum]
                faces = self.model.mesh_face[faceadr:faceadr+facenum]

                # 构建旋转矩阵（mujoco geom_xmat 是行主序 3x3）
                rot = mat
                handle = self.server.scene.add_mesh_simple(
                    name=f"geom/{geom_name}",
                    vertices=verts.astype(np.float32),
                    faces=faces.astype(np.int32),
                    position=tuple(pos),
                    color=color,
                    wireframe=False,
                )
                # 设置旋转
                handle.wxyz = self._mat_to_quat(rot)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                handle = self.server.scene.add_box(
                    name=f"geom/{geom_name}",
                    dimensions=(float(size[0]*2), float(size[1]*2), float(size[2]*2)),
                    position=tuple(pos),
                    color=color,
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                handle = self.server.scene.add_icosphere(
                    name=f"geom/{geom_name}",
                    radius=float(size[0]),
                    position=tuple(pos),
                    color=color,
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                handle = self.server.scene.add_cylinder(
                    name=f"geom/{geom_name}",
                    height=float(size[0] * 2),
                    radius=float(size[1]),
                    position=tuple(pos),
                    color=color,
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                handle = self.server.scene.add_cylinder(
                    name=f"geom/{geom_name}",
                    height=float(size[0] * 2),
                    radius=float(size[1]),
                    position=tuple(pos),
                    color=color,
                )
            else:
                # 通用：用小球代替
                handle = self.server.scene.add_icosphere(
                    name=f"geom/{geom_name}",
                    radius=0.02,
                    position=tuple(pos),
                    color=color,
                )

            self._scene_handles[geom_name] = handle

        # 添加关节点标记
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            jnt_type = self.model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                body_id = self.model.jnt_bodyid[i]
                body_pos = self.data.xpos[body_id]
                handle = self.server.scene.add_icosphere(
                    name=f"joint/{joint_name}",
                    radius=0.015,
                    position=tuple(body_pos),
                    color=(0.0, 1.0, 0.0),
                )
                self._joint_handles[joint_name] = handle

    def update(self, qpos: np.ndarray):
        """更新机器人姿态。"""
        import mujoco

        if self.model is None:
            return

        # 应用关节位置
        nq = min(len(qpos), self.model.nq)
        self.data.qpos[:nq] = qpos[:nq]
        mujoco.mj_forward(self.model, self.data)

        # 更新几何体位置
        for geom_name, handle in self._scene_handles.items():
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                pos = self.data.geom_xpos[geom_id]
                handle.position = tuple(pos)

    def run_interactive(self):
        """启动交互式可视化（阻塞）。"""
        if not self._initialized:
            self.init()

        print(f"🌐 Viser 浏览器已启动: http://localhost:{self.port}")
        print(f"   按 Ctrl+C 停止")

        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n🔒 Viser 已关闭")
            if self.server:
                self.server.scene.clear()

    def run_with_inference(self, inference, images: dict, state: np.ndarray,
                          language: str = "perform the task", steps: int = 300):
        """
        结合推理的可视化回放。

        Args:
            inference: GR00TLocalInference 实例
            images: 初始图像
            state: 初始状态
            language: 语言指令
            steps: 回放步数
        """
        if not self._initialized:
            self.init()

        print(f"▶️  开始推理可视化 ({steps} 步)")

        for step in range(steps):
            action, info = inference.predict(
                images=images,
                state=state,
                language=language,
            )

            # 更新状态
            if self.model is not None:
                self.update(action[:self.model.nq])
            else:
                state = action

            time.sleep(1.0 / 30)  # 30 fps

        print("✅ 回放完成")

    def close(self):
        """关闭 Viser。"""
        if self.server:
            self.server = None
        self._scene_handles.clear()
        self._joint_handles.clear()
        self._initialized = False


def select_robot() -> str:
    """交互式选择机器人类型。"""
    robots = [
        ("g1", "Unitree G1 人形机器人 (29-DOF)"),
        ("h1", "Unitree H1 人形机器人 (20-DOF)"),
        ("h1_with_hand", "Unitree H1 人形机器人 (带手, 46-DOF)"),
        ("h1_2", "Unitree H1.2 人形机器人 (52-DOF)"),
        ("h2", "Unitree H2 人形机器人 (32-DOF)"),
        ("go2", "Unitree Go2 四足机器人 (12-DOF)"),
    ]
    print("\n🤖 请选择机器人类型:")
    for i, (key, desc) in enumerate(robots):
        print(f"  [{i}] {desc}")
    while True:
        try:
            choice = input("\n请输入编号 (0-5): ").strip()
            idx = int(choice)
            if 0 <= idx < len(robots):
                return robots[idx][0]
        except (ValueError, EOFError):
            pass
        print("  ⚠️  无效输入，请重新选择")


def main():
    parser = argparse.ArgumentParser(description="Viser 浏览器 3D 可视化")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型路径（用于推理可视化）")
    parser.add_argument("--robot", type=str, default=None,
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"],
                        help="机器人类型（不指定则交互式选择）")
    parser.add_argument("--mjcf-path", type=str, default=None,
                        help="MJCF 模型文件路径")
    parser.add_argument("--port", type=int, default=20006,
                        help="Viser 服务端口")
    args = parser.parse_args()

    robot = args.robot
    if robot is None:
        robot = select_robot()

    viewer = ViserViewer(port=args.port, mjcf_path=args.mjcf_path, robot=robot)
    viewer.init()
    viewer.run_interactive()


if __name__ == "__main__":
    main()
