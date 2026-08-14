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
    """查找机器人 MJCF 文件路径。

    覆盖 robot_retargeter 的多种目录结构：
      - <robot>_description/mjcf/<robot>.xml      (g1, h1)
      - <robot>_description/<robot>.xml           (h1_2, h2)
      - <robot>_description/mjcf/<robot>_with_hand.xml  (h1_with_hand)
    """
    candidates = [
        f"../robot_retargeter/asset/robot/{robot}_description/mjcf/{robot}.xml",
        f"../robot_retargeter/asset/robot/{robot}_description/{robot}.xml",
        f"../robot_retargeter/asset/robot/{robot}_description/mjcf/{robot}_with_hand.xml",
        f"../robot_retargeter/asset/robot/{robot}_description/H2.xml" if robot == "h2" else None,
        f"../Isaac-GR00T/assets/robots/{robot}.xml",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    # 回退：在 robot_retargeter 目录下递归查找
    import glob
    matches = glob.glob(f"../robot_retargeter/asset/robot/{robot}*/**/{robot}*.xml", recursive=True)
    if matches:
        return matches[0]
    return candidates[0]  # 返回第一个作为默认值（会触发"文件不存在"提示）


def _reorder_slices(state_keys: list, local_slices: dict) -> dict:
    """按服务端 state_keys 顺序重新计算连续切片。

    假设 state 仍是按 state_keys 顺序连续拼接的向量。
    用 local_slices 中各 key 的长度推导新切片。
    """
    # 先从 local_slices 推导每个 key 的长度
    lengths = {k: (e - s) for k, (s, e) in local_slices.items()}
    reordered = {}
    offset = 0
    for k in state_keys:
        d = lengths.get(k, 0)
        if d == 0:
            # 未知 key，跳过（后续 ObservationBuilder 会零填充告警）
            continue
        reordered[k] = (offset, offset + d)
        offset += d
    return reordered


class ViserViewer:
    """Viser 3D 查看器：借鉴 robot_retargeter 的 multi_robot_visualize_viser.py 设计。

    - set_up_direction("+z") + configure_default_lights + add_grid 地面网格
    - add_batched_meshes_simple 批量渲染（高效）
    - 用 mujoco FK 后的 geom_xpos/geom_xmat 更新位姿
    - GUI: 暂停/重播/帧滑块/速度
    """

    def __init__(self, port: int = 20006, mjcf_path: str = "", robot: str = "g1"):
        self.port = port
        self.mjcf_path = mjcf_path
        self.robot = robot
        self.model = None       # mujoco MjModel
        self.data = None        # mujoco MjData
        self.vis = None         # viser.ViserServer
        self._batched_handles = []  # list[(handle, [geom_idx, ...])]
        self._qpos_addr = 0     # 铰接关节 qpos 起始地址（跳过 floating base）
        # 播放状态
        self.paused = False
        self.fps = 30

    def init(self):
        try:
            import viser
            import mujoco
        except ImportError as e:
            print(f"⚠️  依赖未安装: {e}，仅做推理测试")
            self.vis = None
            return

        # 启动 Viser（即使无模型也启动，便于浏览器查看）
        self.vis = viser.ViserServer(port=self.port, verbose=False)
        self.vis.scene.set_up_direction("+z")
        self.vis.scene.configure_default_lights(enabled=True)
        # 灰色地面网格（与 robot_retargeter 一致）
        self.vis.scene.add_grid(
            "ground",
            width=20, height=20,
            cell_size=1.0, cell_thickness=0.5, cell_color=(80, 80, 80),
            section_thickness=0.8, section_color=(50, 50, 50),
            position=(0, 0, -0.01), wxyz=(1, 0, 0, 0),
        )
        print(f"✅ Viser 服务已启动: http://localhost:{self.port}")

        # 加载 MJCF
        if not self.mjcf_path or not Path(self.mjcf_path).exists():
            print(f"⚠️  MJCF 文件不存在: {self.mjcf_path}（仅显示地面网格）")
            return

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        print(f"   MJCF: {self.mjcf_path} (nq={self.model.nq}, nbody={self.model.nbody})")

        # 找到铰接关节 qpos 起始地址（跳过 floating base 的 7 维）
        self._qpos_addr = 7 if self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0

        # 按 mesh_id 分组 geom，用 add_batched_meshes_simple 批量渲染
        # mesh_vertadr/mesh_faceadr 是起始地址，mesh_vertnum/mesh_facenum 是数量
        mesh_groups: dict = {}  # mesh_id -> [geom_idx, ...]
        for i in range(self.model.ngeom):
            gtype = int(self.model.geom_type[i])
            if gtype != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue  # 只渲染 mesh 类型 geom（与 robot_retargeter --all-geoms=False 一致）
            mesh_id = int(self.model.geom_dataid[i])
            if mesh_id < 0:
                continue
            mesh_groups.setdefault(mesh_id, []).append(i)

        for mesh_id, geom_ids in mesh_groups.items():
            if mesh_id >= self.model.nmesh:
                continue
            vert_adr = int(self.model.mesh_vertadr[mesh_id])
            vert_num = int(self.model.mesh_vertnum[mesh_id])
            face_adr = int(self.model.mesh_faceadr[mesh_id])
            face_num = int(self.model.mesh_facenum[mesh_id])
            if vert_num <= 0 or face_num <= 0:
                continue
            vertices = np.asarray(self.model.mesh_vert[vert_adr:vert_adr + vert_num], dtype=np.float32)
            faces = np.asarray(self.model.mesh_face[face_adr:face_adr + face_num], dtype=np.uint32)

            n_inst = len(geom_ids)
            positions = np.zeros((n_inst, 3), dtype=np.float32)
            wxyzs = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)
            colors = np.zeros((n_inst, 3), dtype=np.uint8)
            # 用初始 geom_xpos/geom_xmat 填充，并提取每个 geom 的 rgba 颜色
            for j, gid in enumerate(geom_ids):
                positions[j] = self.data.geom_xpos[gid]
                wxyzs[j] = _rotmat_to_wxyz(self.data.geom_xmat[gid])
                # geom_rgba 是 [r,g,b,a] 0-1 浮点，转 0-255 uint8
                rgba = self.model.geom_rgba[gid]
                colors[j] = (np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)

            handle = self.vis.scene.add_batched_meshes_simple(
                name=f"robot/mesh_{mesh_id}",
                vertices=vertices,
                faces=faces,
                batched_wxyzs=wxyzs,
                batched_positions=positions,
                batched_colors=colors,
            )
            self._batched_handles.append((handle, geom_ids))

        print(f"   已渲染 {len(self._batched_handles)} 个 mesh 组 ({sum(len(g) for _, g in self._batched_handles)} 个 geom)")

    def update(self, qpos):
        """根据关节角度更新机器人姿态。

        Args:
            qpos: 铰接关节角度（不含 floating base），长度 = model.nq - 7
        """
        if self.model is None or self.vis is None:
            return

        import mujoco
        # 把 qpos 写入 mujoco data（跳过 floating base 的 7 维）
        n_artic = self.model.nq - self._qpos_addr
        if len(qpos) >= n_artic:
            self.data.qpos[self._qpos_addr:] = qpos[:n_artic]
        else:
            self.data.qpos[self._qpos_addr:self._qpos_addr + len(qpos)] = qpos
        mujoco.mj_forward(self.model, self.data)

        # 用 geom_xpos/geom_xmat 更新每个 batched mesh 的位姿
        all_pos = self.data.geom_xpos
        all_mat = self.data.geom_xmat
        for handle, geom_ids in self._batched_handles:
            n_inst = len(geom_ids)
            pos = np.zeros((n_inst, 3), dtype=np.float32)
            wxyz = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)
            for j, gid in enumerate(geom_ids):
                pos[j] = all_pos[gid]
                wxyz[j] = _rotmat_to_wxyz(all_mat[gid])
            handle.batched_positions = pos
            handle.batched_wxyzs = wxyz

    def close(self):
        if self.vis is not None:
            try:
                self.vis.stop()
            except Exception:
                pass


def _rotmat_to_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵（展平）→ wxyz 四元数。借鉴 robot_retargeter 的实现。"""
    m = mat.flatten()
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s])
    elif m[0] > m[4] and m[0] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[0] - m[4] - m[8])
        return np.array([(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s])
    elif m[4] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[4] - m[0] - m[8])
        return np.array([(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[8] - m[0] - m[4])
        return np.array([(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s])


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
        video_keys = ["front", "wrist"]
        state_keys = []
        language_key = "annotation.human.task_description"
        num_obs_steps = 1
        if isinstance(modality_config, dict):
            video_cfg = modality_config.get("video", {})
            state_cfg = modality_config.get("state", {})
            lang_cfg = modality_config.get("language", {})
            if isinstance(video_cfg, dict):
                mk = video_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    video_keys = mk
                di = video_cfg.get("delta_indices")
                if isinstance(di, list) and di:
                    num_obs_steps = len(di)
            if isinstance(state_cfg, dict):
                mk = state_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    state_keys = mk
            if isinstance(lang_cfg, dict):
                mk = lang_cfg.get("modality_keys")
                if isinstance(mk, list) and mk:
                    language_key = mk[0]
        else:
            # 兼容 ModalityConfig 对象（本地直连场景）
            video_keys = list(getattr(modality_config.get("video"), "modality_keys", video_keys) or video_keys)
            state_keys = list(getattr(modality_config.get("state"), "modality_keys", []) or [])
            lang_mks = getattr(modality_config.get("language"), "modality_keys", None)
            if lang_mks:
                language_key = lang_mks[0]

        # state_slices：优先用机器人配置（与 modality.json 对齐），回退到单键
        from src.observation_builder import state_slices_from_config
        state_slices = state_slices_from_config(self.robot)
        # 若服务端返回的 state_keys 与本地切片不一致，以服务端为准并按顺序重排
        if state_keys and list(state_slices.keys()) != state_keys:
            # 按服务端 state_keys 顺序重新计算切片（假设仍是连续拼接）
            state_slices = _reorder_slices(state_keys, state_slices)

        self.obs_builder = ObservationBuilder(
            camera_keys=video_keys,
            state_dim=sum(e - s for s, e in state_slices.values()),
            state_slices=state_slices,
            language_key=language_key,
            num_obs_steps=num_obs_steps,
        )

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
          - dict: {"joint_pos": (B,T,D) ndarray} → 取 [0,0,:] 或 [0,:]
          - ndarray: (B,T,D) / (T,D) / (D,) → 取首步首 batch
          - tuple/list: (action_dict, info_dict) → 递归处理第一个元素
        """
        # get_action 返回 (action, info) tuple：
        #   - action 是 dict → 递归处理 dict
        #   - action 是 ndarray → 取该 ndarray
        if isinstance(action_result, (tuple, list)) and len(action_result) == 2:
            action_result = action_result[0]

        if isinstance(action_result, dict):
            # 优先 joint_pos（与 modality_config 对齐），其次旧 key
            for key in ["joint_pos", "joint_position_delta", "joint_position", "action"]:
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

        # 维护绝对关节角状态（用于驱动 Viser 模型）
        # state 是完整观测向量 (joint_pos + joint_vel + base...)，取前 num_joints 维作为初始 qpos
        from src.observation_builder import state_slices_from_config
        slices = state_slices_from_config(self.robot)
        num_joints = slices["joint_pos"][1] - slices["joint_pos"][0]
        qpos = state[:num_joints].copy() if len(state) >= num_joints else np.zeros(num_joints, dtype=np.float32)

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

                # action_result 可能是 dict（如 {"joint_pos": (B,T,D) ndarray}）
                # 或 ndarray。提取单步动作向量 (action_dim,)
                action = self._extract_action_vector(action_result)

                # action 是 joint_pos（RELATIVE delta），累加到绝对关节角
                num_act = len(action)
                if len(qpos) >= num_act:
                    qpos = qpos.copy()
                    qpos[:num_act] = qpos[:num_act] + action
                else:
                    qpos = np.atleast_1d(np.array(action, dtype=np.float32))

                # 更新 3D 模型（传入绝对关节角，不含 floating base）
                self.viewer.update(qpos)

                # 同步更新观测状态向量的 joint_pos 切片
                if len(state) >= num_act:
                    state = state.copy()
                    state[:num_act] = qpos[:num_act]

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
