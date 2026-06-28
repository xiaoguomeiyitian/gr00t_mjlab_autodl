"""
mujoco_infer.py — MuJoCo 桌面窗口 + Policy Server 实时推理。

连接云端 Policy Server，获取 action，实时驱动 MuJoCo 桌面窗口中的机器人模型。

用法:
    python -m src.viz.mujoco_infer --robot g1 --host 127.0.0.1 --port 5555
    python -m src.viz.mujoco_infer --robot h1 --dataset ../Isaac-GR00T/demo_data/droid_sample

依赖: pip install mujoco glfw msgpack msgpack-numpy pyzmq
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.policy_client import GR00TClient
from src.observation_builder import ObservationBuilder
from src.lerobot_loader import LeRobotEpisodeLoader
from src.viz.mujoco_viewer import MuJoCoViewer


class MuJoCoInferLoop:
    """MuJoCo + Policy Server 实时推理循环。"""

    def __init__(
        self,
        robot: str = "g1",
        host: str = "127.0.0.1",
        port: int = 5555,
        mjcf_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        traj_id: int = 1,
        fps: int = 30,
    ):
        self.robot = robot
        self.host = host
        self.port = port
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.traj_id = traj_id
        self.fps = fps

        # 初始化 MuJoCo viewer
        self.viewer = MuJoCoViewer(mjcf_path=mjcf_path, robot=robot)
        self.viewer.init()

        # Policy client（延迟连接）
        self.client: Optional[GR00TClient] = None
        self.obs_builder: Optional[ObservationBuilder] = None
        self.dataset: Optional[LeRobotEpisodeLoader] = None

    def connect(self):
        """连接 Policy Server。"""
        self.client = GR00TClient(host=self.host, port=self.port)

        # 获取 modality config
        modality_config = self.client.get_modality_config()
        video_keys = list(modality_config.get("video", {}).keys()) if isinstance(modality_config, dict) else ["exterior_image_1_left"]
        self.obs_builder = ObservationBuilder(camera_keys=video_keys)

        # 加载数据集（如果提供）
        if self.dataset_path and Path(self.dataset_path).exists():
            self.dataset = LeRobotEpisodeLoader(
                dataset_path=self.dataset_path,
                modality_configs=modality_config,
            )
            print(f"📊 数据集加载完成: {len(self.dataset)} episodes")

    def _get_initial_state(self) -> tuple:
        """获取初始状态和图像。"""
        if self.dataset and self.traj_id < len(self.dataset):
            traj = self.dataset[self.traj_id]
            step = 0

            # 提取图像
            images = {}
            for key in ["exterior_image_1_left", "wrist_image_left"]:
                if key in traj.columns:
                    img = traj[key].iloc[step]
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    images[key] = img

            # 提取状态
            state_cols = [c for c in traj.columns if c.startswith("state.")]
            if state_cols:
                state = np.vstack([traj[c].iloc[step] for c in state_cols]).flatten().astype(np.float32)
            else:
                state = np.zeros(17, dtype=np.float32)

            return images, state
        else:
            # 默认空状态
            images = {"exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8)}
            state = np.zeros(17, dtype=np.float32)
            return images, state

    def run(self):
        """运行推理可视化循环。"""
        self.connect()

        images, state = self._get_initial_state()
        print(f"▶️  开始 MuJoCo 推理可视化 (host={self.host}:{self.port})")
        print(f"   空格 = 暂停/继续 | Esc = 退出")

        step = 0
        paused = False

        def policy_fn(current_state):
            nonlocal step, state
            # 构建观测
            obs = self.obs_builder.build(
                images=images,
                state=state,
            )
            # 推理
            action, info = self.client.get_action(obs)
            # 更新状态
            state = action[:len(state)]
            step += 1
            return action

        try:
            self.viewer.run_passive(policy_fn=policy_fn, fps=self.fps)
        except KeyboardInterrupt:
            print(f"\n🔒 MuJoCo 推理可视化已停止 ({step} 步)")
        finally:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 桌面窗口 + Policy Server 实时推理")
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"],
                        help="机器人类型")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Policy Server 地址")
    parser.add_argument("--port", type=int, default=5555,
                        help="Policy Server 端口")
    parser.add_argument("--mjcf-path", type=str, default=None,
                        help="MJCF 模型文件路径")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集路径（用于获取初始观测）")
    parser.add_argument("--embodiment-tag", type=str,
                        default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                        help="具身标签")
    parser.add_argument("--traj-id", type=int, default=1,
                        help="轨迹 ID")
    parser.add_argument("--fps", type=int, default=30,
                        help="可视化帧率")
    args = parser.parse_args()

    loop = MuJoCoInferLoop(
        robot=args.robot,
        host=args.host,
        port=args.port,
        mjcf_path=args.mjcf_path,
        dataset_path=args.dataset,
        embodiment_tag=args.embodiment_tag,
        traj_id=args.traj_id,
        fps=args.fps,
    )
    loop.run()


if __name__ == "__main__":
    main()
