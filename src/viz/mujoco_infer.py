"""
mujoco_infer.py — MuJoCo 桌面窗口 + Policy Server 实时推理。

连接云端 Policy Server，获取 action，实时驱动 MuJoCo 桌面窗口中的机器人模型。

用法:
    python -m src.viz.mujoco_infer --robot g1 --host 127.0.0.1 --port 5555

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


class MuJoCoViewer:
    """轻量 MuJoCo 查看器（内联实现）。"""

    def __init__(self, mjcf_path: str = "", robot: str = "g1"):
        self.mjcf_path = mjcf_path
        self.robot = robot
        self.model = None

    def init(self):
        try:
            import mujoco
            if self.mjcf_path and Path(self.mjcf_path).exists():
                self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
                print(f"✅ MuJoCo 模型已加载: {self.mjcf_path}")
            else:
                print("⚠️  MJCF 文件不存在，仅做推理测试")
        except ImportError:
            print("⚠️  mujoco 未安装，仅做推理测试")
            self.model = None

    def run_passive(self, policy_fn, fps=30):
        """简化版运行循环。"""
        print("⚠️  MuJoCo 窗口需要桌面环境，当前仅测试推理连接")
        # 简化：只跑几步推理测试
        import time
        # 首次 state 不能为 None，否则 obs_builder.build 会 AttributeError
        state = np.zeros(71, dtype=np.float32)
        for i in range(10):
            action = policy_fn(state)
            print(f"   Step {i}: action shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}]")
            time.sleep(1.0 / fps)


class MuJoCoInferLoop:
    """MuJoCo + Policy Server 实时推理循环。"""

    def __init__(
        self,
        robot: str = "g1",
        host: str = "127.0.0.1",
        port: int = 5555,
        mjcf_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        embodiment_tag: str = "NEW_EMBODIMENT",
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
        # modality_config 反序列化后是嵌套 dict，相机名在 ["video"]["modality_keys"]
        video_keys = ["exterior_image_1_left"]
        if isinstance(modality_config, dict):
            video_cfg = modality_config.get("video", {})
            if isinstance(video_cfg, dict):
                mk = video_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    video_keys = mk
            else:
                mk = getattr(video_cfg, "modality_keys", None)
                if mk:
                    video_keys = list(mk)
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
        from src.configs.g1_config import STATE_DIM as G1_STATE_DIM
        from src.configs.go2_config import STATE_DIM as GO2_STATE_DIM
        _state_dim_map = {"g1": G1_STATE_DIM, "go2": GO2_STATE_DIM}
        state_dim = _state_dim_map.get(self.robot, 71)

        if self.dataset and self.traj_id < len(self.dataset):
            traj = self.dataset[self.traj_id]
            step = 0

            # 提取图像：用 obs_builder 的 camera_keys
            images = {}
            cam_keys = self.obs_builder.camera_keys if self.obs_builder else []
            for key in cam_keys:
                col_candidates = [key, f"observation.images.{key}"]
                for col in col_candidates:
                    if col in traj.columns:
                        img = traj[col].iloc[step]
                        if not isinstance(img, np.ndarray):
                            img = np.array(img)
                        images[key] = img
                        break

            # 提取状态：LeRobot v2 单列 observation.state
            if "observation.state" in traj.columns:
                state = np.atleast_1d(np.array(traj["observation.state"].iloc[step], dtype=np.float32))
            else:
                state = np.zeros(state_dim, dtype=np.float32)

            return images, state
        else:
            cam_keys = self.obs_builder.camera_keys if self.obs_builder else ["front"]
            images = {k: np.zeros((224, 224, 3), dtype=np.uint8) for k in cam_keys}
            state = np.zeros(state_dim, dtype=np.float32)
            return images, state

    def _extract_action_vector(self, action_result) -> np.ndarray:
        """从推理结果提取单步动作向量 (action_dim,)。"""
        if isinstance(action_result, (tuple, list)) and len(action_result) == 2:
            action_result = action_result[0]
        if isinstance(action_result, dict):
            for key in ["joint_position_delta", "joint_position", "action"]:
                if key in action_result:
                    arr = np.asarray(action_result[key], dtype=np.float32)
                    return self._squeeze_action(arr)
            arr = np.asarray(list(action_result.values())[0], dtype=np.float32)
            return self._squeeze_action(arr)
        arr = np.asarray(action_result, dtype=np.float32)
        return self._squeeze_action(arr)

    @staticmethod
    def _squeeze_action(arr: np.ndarray) -> np.ndarray:
        while arr.ndim > 1:
            arr = arr[0]
        return arr

    def run(self):
        """运行推理可视化循环。"""
        self.connect()

        images, state = self._get_initial_state()
        print(f"▶️  开始 MuJoCo 推理可视化 (host={self.host}:{self.port})")
        print(f"   空格 = 暂停/继续 | Esc = 退出")

        step = 0

        def policy_fn(current_state):
            nonlocal step, state
            # current_state 可能为 None（首次），用 self.state
            cur = state if current_state is None else current_state
            # 构建观测
            obs = self.obs_builder.build(
                images=images,
                state=cur,
            )
            # 推理
            action_result, info = self.client.get_action(obs)
            # 从 action dict 提取单步动作向量
            action = self._extract_action_vector(action_result)
            # 更新状态：action 是 joint_position_delta，仅累加到 joint_pos 切片
            num_joints = len(action)
            if len(state) >= num_joints:
                state = state.copy()
                state[:num_joints] = state[:num_joints] + action
            else:
                state = np.atleast_1d(np.array(action, dtype=np.float32))
            step += 1
            return action

        try:
            self.viewer.run_passive(policy_fn=policy_fn, fps=self.fps)
        except KeyboardInterrupt:
            print(f"\n🔒 MuJoCo 推理可视化已停止 ({step} 步)")
        finally:
            if self.client is not None:
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
                        default="NEW_EMBODIMENT",
                        help="具身标签（微调模型用 NEW_EMBODIMENT，预训练用 OXE_DROID_*）")
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
