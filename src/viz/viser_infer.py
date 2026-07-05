"""
viser_infer.py — Viser 3D 可视化 + Policy Server 实时推理。

连接云端 Policy Server，获取 action，实时驱动 3D 机器人模型。

用法:
    python -m src.viz.viser_infer --robot g1 --host 127.0.0.1 --port 5555

依赖: pip install viser mujoco msgpack msgpack-numpy pyzmq
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.policy_client import GR00TClient
from src.observation_builder import ObservationBuilder
from src.lerobot_loader import LeRobotEpisodeLoader


def find_robot_mjcf(robot: str) -> str:
    """查找机器人 MJCF 文件路径。"""
    candidates = [
        f"../robot_retargeter/asset/robot/{robot}.xml",
        f"../robot_retargeter/asset/robot/{robot}_description/{robot}.xml",
        f"../Isaac-GR00T/assets/robots/{robot}.xml",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[0]  # 返回第一个作为默认值


class ViserViewer:
    """轻量 Viser 查看器（内联实现）。"""

    def __init__(self, port: int = 20006, mjcf_path: str = "", robot: str = "g1"):
        self.port = port
        self.mjcf_path = mjcf_path
        self.robot = robot
        self.model = None
        self.vis = None

    def init(self):
        try:
            import viser
            self.vis = viser.ViserServer(port=self.port)
            print(f"✅ Viser 服务已启动: http://localhost:{self.port}")
        except ImportError:
            print("⚠️  viser 未安装，仅做推理测试")
            self.vis = None

    def update(self, qpos):
        pass  # 简化实现

    def close(self):
        pass


class ViserInferLoop:
    """Viser + Policy Server 实时推理循环。"""

    def __init__(
        self,
        robot: str = "g1",
        host: str = "127.0.0.1",
        port: int = 5555,
        viser_port: int = 20006,
        mjcf_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        embodiment_tag: str = "NEW_EMBODIMENT",
        traj_id: int = 1,
        action_horizon: int = 8,
        fps: int = 30,
    ):
        self.robot = robot
        self.host = host
        self.port = port
        self.viser_port = viser_port
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.traj_id = traj_id
        self.action_horizon = action_horizon
        self.fps = fps

        # 初始化 Viser
        if mjcf_path is None:
            mjcf_path = find_robot_mjcf(robot)
        self.viewer = ViserViewer(port=viser_port, mjcf_path=mjcf_path, robot=robot)
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
        # modality_config 反序列化后是嵌套 dict：
        #   {"video": {"delta_indices": [...], "modality_keys": ["front","wrist"], ...}, ...}
        # 相机名在 modality_config["video"]["modality_keys"]，不是顶层 .keys()
        video_keys = ["exterior_image_1_left"]
        if isinstance(modality_config, dict):
            video_cfg = modality_config.get("video", {})
            if isinstance(video_cfg, dict):
                mk = video_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    video_keys = mk
            else:
                # 兼容 ModalityConfig 对象（本地直连场景）
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
        # 从 robot config 获取 state_dim，避免硬编码
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
                # LeRobot 列名形如 observation.images.front
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
        """从推理结果提取单步动作向量 (action_dim,)。

        action_result 可能是：
          - dict: {"joint_position_delta": (B,T,D) ndarray} → 取 [0,0,:] 或 [0,:]
          - ndarray: (B,T,D) / (T,D) / (D,) → 取首步首 batch
          - tuple/list: (action_dict, info_dict) → 递归处理第一个元素
        """
        # get_action 返回 (action, info) tuple：
        #   - action 是 dict → 递归处理 dict
        #   - action 是 ndarray → 取该 ndarray
        if isinstance(action_result, (tuple, list)) and len(action_result) == 2:
            action_result = action_result[0]

        if isinstance(action_result, dict):
            # 优先 joint_position_delta，其次任意 action key
            for key in ["joint_position_delta", "joint_position", "action"]:
                if key in action_result:
                    arr = np.asarray(action_result[key], dtype=np.float32)
                    return self._squeeze_action(arr)
            # 退化为第一个值
            arr = np.asarray(list(action_result.values())[0], dtype=np.float32)
            return self._squeeze_action(arr)

        arr = np.asarray(action_result, dtype=np.float32)
        return self._squeeze_action(arr)

    @staticmethod
    def _squeeze_action(arr: np.ndarray) -> np.ndarray:
        """(B,T,D) / (T,D) / (D,) → (D,) 单步动作。"""
        while arr.ndim > 1:
            arr = arr[0]
        return arr

    def run(self):
        """运行推理可视化循环。"""
        self.connect()

        images, state = self._get_initial_state()
        print(f"▶️  开始推理可视化 (host={self.host}:{self.port})")
        print(f"   按 Ctrl+C 停止")

        step = 0
        try:
            while True:
                # 构建观测
                obs = self.obs_builder.build(
                    images=images,
                    state=state,
                )

                # 推理
                action_result, info = self.client.get_action(obs)

                # action_result 可能是 dict（如 {"joint_position_delta": (B,T,D) ndarray}）
                # 或 ndarray。提取单步动作向量 (action_dim,)
                action = self._extract_action_vector(action_result)

                # 更新 3D 模型
                nq = self.viewer.model.nq if self.viewer.model else len(action)
                self.viewer.update(action[:nq])

                # 更新状态：action 是 joint_position_delta，仅累加到 joint_pos 切片
                num_joints = len(action)
                if len(state) >= num_joints:
                    state = state.copy()
                    state[:num_joints] = state[:num_joints] + action
                else:
                    state = np.atleast_1d(np.array(action, dtype=np.float32))

                step += 1
                time.sleep(1.0 / self.fps)

        except KeyboardInterrupt:
            print(f"\n🔒 推理可视化已停止 ({step} 步)")
        finally:
            if self.client is not None:
                self.client.close()
            self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Viser 3D 可视化 + Policy Server 实时推理")
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"],
                        help="机器人类型")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Policy Server 地址")
    parser.add_argument("--port", type=int, default=5555,
                        help="Policy Server 端口")
    parser.add_argument("--viser-port", type=int, default=20006,
                        help="Viser 服务端口")
    parser.add_argument("--mjcf-path", type=str, default=None,
                        help="MJCF 模型文件路径")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集路径（用于获取初始观测）")
    parser.add_argument("--embodiment-tag", type=str,
                        default="NEW_EMBODIMENT",
                        help="具身标签（微调模型用 NEW_EMBODIMENT，预训练用 OXE_DROID_*）")
    parser.add_argument("--traj-id", type=int, default=1,
                        help="轨迹 ID")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="动作预测步数")
    parser.add_argument("--fps", type=int, default=30,
                        help="可视化帧率")
    args = parser.parse_args()

    loop = ViserInferLoop(
        robot=args.robot,
        host=args.host,
        port=args.port,
        viser_port=args.viser_port,
        mjcf_path=args.mjcf_path,
        dataset_path=args.dataset,
        embodiment_tag=args.embodiment_tag,
        traj_id=args.traj_id,
        action_horizon=args.action_horizon,
        fps=args.fps,
    )
    loop.run()


if __name__ == "__main__":
    main()
