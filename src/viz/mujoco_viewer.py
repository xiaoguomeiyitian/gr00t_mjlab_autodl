"""
mujoco_viewer.py — MuJoCo 原生桌面窗口可视化。

使用 mujoco.viewer 提供低延迟桌面窗口可视化。
支持键盘交互：空格暂停、R 重播。

依赖: pip install mujoco glfw
运行: python -m src.viz.mujoco_viewer --robot g1
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np


class MuJoCoViewer:
    """MuJoCo 原生桌面窗口可视化。"""

    def __init__(self, mjcf_path: Optional[str] = None, robot: str = "g1"):
        """
        Args:
            mjcf_path: MJCF 模型文件路径
            robot: 机器人类型（用于生成占位模型）
        """
        self.mjcf_path = mjcf_path
        self.robot = robot
        self.model = None
        self.data = None
        self._viewer = None

    def _find_robot_mjcf(self) -> Optional[str]:
        """自动查找 robot_retargeter 项目中的 MJCF 模型。"""
        from src.viz.viser_viewer import find_robot_mjcf
        return find_robot_mjcf(self.robot)

    def init(self):
        """初始化 MuJoCo 模型。"""
        try:
            import mujoco
        except ImportError:
            print("❌ 未安装 mujoco，请运行: pip install mujoco glfw")
            print("   或使用 Viser 可视化: ./start.sh viser")
            raise SystemExit(1)

        mjcf_path = self.mjcf_path
        if mjcf_path is None:
            mjcf_path = self._find_robot_mjcf()

        if mjcf_path and Path(mjcf_path).exists():
            spec = mujoco.MjSpec.from_file(mjcf_path)
            self.model = spec.compile()
            self.data = mujoco.MjData(self.model)
            print(f"  ✅ MJCF 加载: {mjcf_path}")
        else:
            # 生成简单占位模型
            self.model, self.data = self._create_placeholder_model()

        mujoco.mj_forward(self.model, self.data)
        print(f"✅ MuJoCo 模型加载完成")
        print(f"   关节数: {self.model.njnt}")
        print(f"   自由度 nq: {self.model.nq}")
        print(f"   速度维度 nv: {self.model.nv}")

    def _create_placeholder_model(self) -> tuple:
        """创建简单占位 MuJoCo 模型（用于无 MJCF 时的测试）。"""
        import mujoco

        xml = """
        <mujoco model="placeholder_robot">
            <option timestep="0.02" gravity="0 0 -9.81"/>
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
                <body name="base" pos="0 0 0.5">
                    <geom type="sphere" size="0.1" rgba="0.2 0.6 1 1"/>
                    <joint type="free" name="base_joint"/>
                    <body name="leg_L" pos="0 0.1 -0.3">
                        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0.8 0.2 0.2 1"/>
                        <joint type="hinge" axis="1 0 0" name="L_hip"/>
                    </body>
                    <body name="leg_R" pos="0 -0.1 -0.3">
                        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0.2 0.8 0.2 1"/>
                        <joint type="hinge" axis="1 0 0" name="R_hip"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    def update(self, qpos: np.ndarray):
        """更新关节位置。"""
        import mujoco

        nq = min(len(qpos), self.model.nq)
        self.data.qpos[:nq] = qpos[:nq]
        mujoco.mj_forward(self.model, self.data)

    def run_passive(self, policy_fn=None, initial_state=None, steps: int = 300, fps: int = 30):
        """
        启动被动模式可视化。

        Args:
            policy_fn: 策略函数，接受 state 返回 action (callable or None)
            initial_state: 初始状态 (qpos)
            steps: 回放步数
            fps: 帧率
        """
        import mujoco
        import mujoco.viewer

        if initial_state is not None:
            self.update(initial_state)

        dt = 1.0 / fps

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print(f"🖥️  MuJoCo 窗口已启动")
            print(f"   空格 = 暂停/继续 | R = 重播 | Esc = 退出")

            step = 0
            paused = False

            while viewer.is_running():
                if not paused:
                    if policy_fn is not None:
                        # 使用策略生成动作
                        state = self.data.qpos[:self.model.nq].copy()
                        action = policy_fn(state)
                        if action is not None:
                            nq = min(len(action), self.model.nq)
                            self.data.ctrl[:nq] = action[:nq]

                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    step += 1

                    if step >= steps:
                        print(f"\n✅ 回放完成 ({steps} 步)")
                        break

                    time.sleep(dt)

            print("🔒 MuJoCo 窗口已关闭")

    def run_interactive(self):
        """启动交互模式（自由操控）。"""
        import mujoco
        import mujoco.viewer

        mujoco.viewer.launch(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 原生桌面可视化")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型路径（用于推理可视化）")
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"],
                        help="机器人类型")
    parser.add_argument("--mjcf-path", type=str, default=None,
                        help="MJCF 模型文件路径")
    parser.add_argument("--steps", type=int, default=300,
                        help="回放步数")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率")
    args = parser.parse_args()

    viewer = MuJoCoViewer(mjcf_path=args.mjcf_path, robot=args.robot)
    viewer.init()
    viewer.run_passive(steps=args.steps, fps=args.fps)


if __name__ == "__main__":
    main()
