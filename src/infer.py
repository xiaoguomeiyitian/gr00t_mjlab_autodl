#!/usr/bin/env python3
"""
GR00T 本地推理包装器 — 基于 unitree_rl_mjlab 仿真环境进行推理

支持 INT4/FP16/BF16 推理和 Viser 3D 可视化。

使用方法:
    # INT4 推理 (8GB GPU, 已量化的模型)
    python infer.py --robot g1 \
        --model-path models/g1_gr00t_int4 \
        --instruction "walk forward"

    # BF16 推理 (24GB+ GPU, fine-tune 输出)
    python infer.py --robot g1 \
        --model-path models/g1_gr00t

注: Isaac-GR00T N1.7 Gr00tPolicy 在 from_pretrained 时已自动加载模型保存时的精度/dtype。
    训练时是 BF16, INT4 量化后是 4-bit, 推理时无需再传 --quantize (历史参数已忽略)。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class GR00TLocalInference:
    """GR00T 本地推理包装器。

    在 unitree_rl_mjlab 仿真环境中加载 fine-tune 后的 GR00T 模型，
    接收 RGB 图像 + 本体感知 + 语言指令，输出关节动作。
    """

    # G1 29 关节顺序 (来自 g1_config.py G1_JOINT_NAMES):
    #   左腿 6, 右腿 6, 腰部 3, 左臂 7, 右臂 7 (共 29)
    _PART_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "left_leg":  (0, 6),
        "right_leg": (6, 12),
        "waist":     (12, 15),
        "left_arm":  (15, 22),
        "right_arm": (22, 29),
    }

    def __init__(
        self,
        model_path: str,
        robot: str = "g1",
        quantize: str = "auto",
        device: str = "auto",
        action_horizon: int = 16,
        execution_horizon: int = 1,
        task_id: str | None = None,
        instruction: str = "walk forward",
        viser: bool = False,
        viser_port: int = 20006,
        embodiment_tag: str | None = None,
    ):
        self.model_path = model_path
        self.robot = robot
        self.quantize = quantize
        self.device = device
        self.action_horizon = action_horizon
        # 运行时校验: execution_horizon 不能超过 action_horizon
        if execution_horizon > action_horizon:
            logger.warning(
                "execution_horizon (%d) > action_horizon (%d), 已裁剪为 %d",
                execution_horizon, action_horizon, action_horizon,
            )
        self.execution_horizon = max(1, min(execution_horizon, action_horizon))
        self.task_id = task_id or (
            "Mjlab-Velocity-Flat-Unitree-G1" if robot == "g1" else "Mjlab-Velocity-Flat-Unitree-Go2"
        )
        self.instruction = instruction
        self._viser = viser
        self._viser_port = viser_port
        self._viewer = None  # AsyncViser3DViewer 实例

        # Action chunking 队列 (缓存 GR00T 输出的多步动作)
        self._action_queue: list[np.ndarray] = []
        self._action_queue_start_pos: np.ndarray | None = None  # RELATIVE 模式累加起点

        # 加载配置
        if robot == "g1":
            from configs.g1_config import (
                G1_NUM_JOINTS, G1_DEFAULT_JOINT_ANGLES, G1_JOINT_NAMES,
            )
            self.num_joints = G1_NUM_JOINTS
            self.default_angles = np.array(
                [G1_DEFAULT_JOINT_ANGLES[n] for n in G1_JOINT_NAMES], dtype=np.float32
            )
        else:
            from configs.go2_config import (
                GO2_NUM_JOINTS, GO2_DEFAULT_JOINT_ANGLES, GO2_JOINT_NAMES,
            )
            self.num_joints = GO2_NUM_JOINTS
            self.default_angles = np.array(
                [GO2_DEFAULT_JOINT_ANGLES[n] for n in GO2_JOINT_NAMES], dtype=np.float32
            )

        # 自动检测量化模式
        if quantize == "auto":
            model_path_lower = model_path.lower()
            if "int4" in model_path_lower:
                self.quantize = "4bit"
                logger.info("自动检测: INT4 量化模型")
            elif "int8" in model_path_lower:
                self.quantize = "8bit"
                logger.info("自动检测: INT8 量化模型")
            else:
                self.quantize = "none"
                logger.info("自动检测: 全精度模型")

        # 自动检测设备
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        # 自动检测 embodiment tag (从模型 processor_config.json 读取)
        self.embodiment_tag = embodiment_tag or self._detect_embodiment_tag()

        self._policy = None

    def _detect_embodiment_tag(self) -> str:
        """从模型 processor_config.json 自动检测 embodiment tag.

        优先匹配模型名中的关键词 (如 g1_sonic → unitree_g1_sonic),
        否则返回第一个可用的 tag。
        """
        import json as _json
        from pathlib import Path as _Path

        proc_cfg_path = _Path(self.model_path) / "processor_config.json"
        if not proc_cfg_path.exists():
            logger.warning("processor_config.json 不存在, 使用 NEW_EMBODIMENT")
            return "new_embodiment"

        proc_cfg = _json.loads(proc_cfg_path.read_text())
        modality_configs = proc_cfg.get("processor_kwargs", {}).get("modality_configs", {})
        available_tags = list(modality_configs.keys())

        if not available_tags:
            logger.warning("processor_config.json 中无 modality_configs, 使用 NEW_EMBODIMENT")
            return "new_embodiment"

        # 根据模型名关键词匹配
        # 模型路径可能是 "models/g1_gr00t" 或 "models/GR00T-N1.7-G1-SONIC-LAFAN"
        model_name_lower = _Path(self.model_path).name.lower()
        # 也检查父路径 (如 "GR00T-N1.7-G1-SONIC-LAFAN" 在 models/ 下)
        parent_name = _Path(self.model_path).parent.name.lower() if _Path(self.model_path).parent else ""
        combined_name = model_name_lower + " " + parent_name

        # 第一轮: 精确匹配 (优先级最高)
        for tag in available_tags:
            tag_lower = tag.lower()
            if "g1_sonic" in combined_name and "g1_sonic" in tag_lower:
                logger.info(f"自动匹配 embodiment tag: {tag}")
                return tag

        # 第二轮: 宽泛匹配 (按优先级排序)
        # 优先匹配 unitree_ 前缀的 tag (通常是最新/最好的模型)
        for tag in available_tags:
            tag_lower = tag.lower()
            if "unitree" in tag_lower and "g1" in combined_name and "g1" in tag_lower:
                logger.info(f"自动匹配 embodiment tag: {tag}")
                return tag
            if "unitree" in tag_lower and "sonic" in combined_name and "sonic" in tag_lower:
                logger.info(f"自动匹配 embodiment tag: {tag}")
                return tag

        # 第三轮: 一般匹配
        for tag in available_tags:
            tag_lower = tag.lower()
            if "g1" in combined_name and "g1" in tag_lower:
                logger.info(f"自动匹配 embodiment tag: {tag}")
                return tag
            if "sonic" in combined_name and "sonic" in tag_lower:
                logger.info(f"自动匹配 embodiment tag: {tag}")
                return tag

        # 回退: 返回第一个 tag
        selected = available_tags[0]
        logger.warning(f"无法精确匹配 embodiment tag, 使用第一个: {selected}")
        logger.warning(f"可用 tags: {available_tags}")
        return selected

    def load(self):
        """加载 GR00T 模型 (Isaac-GR00T N1.7 Gr00tPolicy)."""
        logger.info("加载 GR00T 模型: %s", self.model_path)
        logger.info("  机器人: %s (%d joints)", self.robot, self.num_joints)
        logger.info("  量化: %s", self.quantize)
        logger.info("  设备: %s", self.device)
        logger.info("  embodiment_tag: %s", self.embodiment_tag)

        try:
            # 官方 Isaac-GR00T (N1.7) 策略实现
            from gr00t.policy.gr00t_policy import Gr00tPolicy
        except ImportError as e:
            logger.error("无法导入 Isaac-GR00T: %s", e)
            logger.info("请确保已安装 Isaac-GR00T 并将 gr00t/ 加入 PYTHONPATH:")
            logger.info("  export PYTHONPATH=/root/unitree/Isaac-GR00T:$PYTHONPATH")
            raise

        # 注意: GR00T 模型保存时已经包含量化信息 (INT4/BF16/FP16),
        # 不需要在加载时再次指定 dtype 或 quantize。
        # Gr00tPolicy 签名: (embodiment_tag, model_path, *, device, strict)
        # embodiment_tag 使用从 processor_config.json 自动检测的 tag
        self._policy = Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=str(self.model_path),
            device=str(self.device),
            strict=True,
        )
        # ── 使用 PolicyHorizonSpec 统一 horizon 参数 (Isaac-GR00T 新特性) ──
        try:
            from gr00t.eval._horizon_contract import PolicyHorizonSpec
            self._horizon_spec = PolicyHorizonSpec.from_policy(self._policy)
            logger.info("  PolicyHorizonSpec: action_horizon=%d, n_action_steps=%d",
                        self._horizon_spec.action_horizon, self._horizon_spec.n_action_steps)
            # 同步 action_horizon (以权威源为准)
            self.action_horizon = self._horizon_spec.action_horizon
        except ImportError:
            # 旧版 Isaac-GR00T 无 PolicyHorizonSpec, 回退到手动管理
            logger.info("  PolicyHorizonSpec 不可用 (旧版 GR00T), 使用手动 horizon")
            self._horizon_spec = None
        # ModalityConfigs: self._policy.modality_configs["action"].action_configs[0].rep
        action_modality = self._policy.modality_configs.get("action")
        if action_modality is None or not action_modality.action_configs:
            raise RuntimeError("Policy 中找不到 action modality_configs, 模型可能不是 fine-tune 后产物")
        self._action_rep = action_modality.action_configs[0].rep
        self._action_keys = action_modality.modality_keys
        logger.info("✅ GR00T 模型加载成功")
        logger.info("  action.rep  = %s", self._action_rep)
        logger.info("  action.keys = %s", self._action_keys)

    def _build_policy_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """把 *单步* 仿真观测转换为 Gr00tPolicy 期望的 batched 形式.

        所有 key 名称均来自加载模型时 self._policy.modality_configs，
        确保与 processor_config.json 中定义的 modality_keys 完全一致。

        Returns:
            dict, 三个顶级 key:
              video:   {view_name: np.ndarray (B=1, T=1, H, W, 3) uint8}
              state:   {state_name: np.ndarray (B=1, T=1, D) float32}
              language:{lang_name: list[list[str]]  shape (B=1, T=1)}
        """
        assert self._policy is not None, "Policy 未加载, 请先调用 load()"
        modality_configs = self._policy.modality_configs

        # ── video ──
        # 使用模型定义的第一个 video key (如 \"ego_view\")
        video_keys = modality_configs["video"].modality_keys
        video_key = video_keys[0]
        frame = obs.get(f"video.{video_key}", obs.get("video.front_view"))
        if frame is None:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
        if frame.ndim == 3:
            frame = frame[None, None, ...]  # (H,W,3) -> (1,1,H,W,3)
        elif frame.ndim == 4:
            frame = frame[None, ...]        # (T,H,W,3) -> (1,T,H,W,3)
        elif frame.ndim == 5 and frame.shape[0] != 1:
            frame = frame[:1]
        video = {video_key: frame.astype(np.uint8)}

        # ── state ──
        # Gr00tPolicy 要求 state[key] 形状 (B, T, D), 这里 B=T=1
        # 使用模型定义的 state modality_keys
        def _to_btd(x):
            arr = np.asarray(x, dtype=np.float32)
            return arr.reshape(1, 1, -1)

        # 从 obs 中提取完整的 joint_pos (如果可用)
        # unitree_g1_sonic 需要将完整 joint_pos 分解为 body parts
        full_joint_pos = obs.get("state.joint_pos")  # 已包含 default_angles (绝对位置)
        full_joint_vel = obs.get("state.joint_vel")

        state = {}
        for model_state_key in modality_configs["state"].modality_keys:
            val = None

            # 1. 如果是 body part 类型 key, 从 full_joint_pos 中切片
            if model_state_key in self._PART_RANGES and full_joint_pos is not None:
                start, end = self._PART_RANGES[model_state_key]
                jp = np.asarray(full_joint_pos, dtype=np.float32)
                if jp.ndim == 0:
                    jp = jp.reshape(1)
                val = jp[start:end]
            # 2. 如果是 joint_vel body part, 从 full_joint_vel 中切片
            elif model_state_key in self._PART_RANGES and full_joint_vel is not None:
                start, end = self._PART_RANGES[model_state_key]
                jv = np.asarray(full_joint_vel, dtype=np.float32)
                if jv.ndim == 0:
                    jv = jv.reshape(1)
                val = jv[start:end]
            # 3. projected_gravity
            elif model_state_key == "projected_gravity":
                pg = obs.get("state.projected_gravity")
                if pg is not None:
                    val = np.asarray(pg, dtype=np.float32).flatten()
                else:
                    val = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            # 4. hand keys: 从 obs 中尝试获取
            elif "hand" in model_state_key:
                # 尝试从 obs 获取手部数据
                hand_key = f"state.{model_state_key}"
                val = obs.get(hand_key)
                if val is None:
                    val = np.zeros(6, dtype=np.float32)
                else:
                    val = np.asarray(val, dtype=np.float32).flatten()

            # 5. fallback: 用 default_angles 填充
            if val is None:
                if model_state_key in self._PART_RANGES:
                    start, end = self._PART_RANGES[model_state_key]
                    val = self.default_angles[start:end].copy()
                else:
                    val = np.zeros(1, dtype=np.float32)

            state[model_state_key] = _to_btd(val)

        # ── language ──
        lang_key = self._policy.language_key
        # 兼容: 仍允许 obs 覆盖 (e.g. 数据回放场景每 step 有不同 instruction)
        instruction = obs.get(
            "annotation.language.action_text",
            obs.get("instruction", self.instruction),
        )
        language = {lang_key: [[str(instruction)]]}

        return {"video": video, "state": state, "language": language}

    def _get_default_subset(self, model_state_key: str) -> np.ndarray:
        """从 default_angles 中提取对应 body part 的关节值.

        unitree_g1_sonic 的 state keys: left_leg(6), right_leg(6),
        waist(3), left_arm(7), right_arm(7), left_hand(?), right_hand(?),
        projected_gravity(3).

        G1 29 关节顺序 (来自 g1_config.py G1_JOINT_NAMES):
          左腿 6, 右腿 6, 腰部 3, 左臂 7, 右臂 7 (共 29)
        """
        if model_state_key in self._PART_RANGES:
            start, end = self._PART_RANGES[model_state_key]
            return self.default_angles[start:end].copy()
        # hand / projected_gravity: 返回零值
        if "hand" in model_state_key:
            return np.zeros(6, dtype=np.float32)  # 手部自由度估计
        if "projected_gravity" in model_state_key:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return np.zeros(1, dtype=np.float32)

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """从观测生成动作 (单步关节目标).

        支持 action chunking: 当 execution_horizon > 1 时, 内部维护一个队列,
        每次调用 GR00T 输出多步, 顺序执行前 execution_horizon 步。
        - RELATIVE 模式: 累积 delta 应用到当前 joint_pos
        - ABSOLUTE 模式: 直接作为目标
        - DELTA 模式: 同 ABSOLUTE (视为目标)

        对于多 action key 模型 (如 unitree_g1_sonic 的 motion_token + hand_joints),
        将所有 action key 的结果合并为一个 (num_joints,) 的关节目标。

        Args:
            obs: 单步观测字典, 同上 _build_policy_observation 输入格式

        Returns:
            joint_targets: (num_joints,) float32, 关节目标位置
        """
        if self._policy is None:
            self.load()

        from gr00t.data.types import ActionRepresentation

        # ── action chunking: 队列非空, 顺序消费 ──
        if self._action_queue:
            return self._pop_cached_action(obs, ActionRepresentation)

        # ── 队列为空: 调 GR00T 重新规划 ──
        policy_obs = self._build_policy_observation(obs)

        # ── 运行时观测格式校验 (Isaac-GR00T 新特性) ──
        try:
            self._policy.check_observation(policy_obs, strict=True)
        except (AssertionError, ValueError) as e:
            logger.warning("观测格式校验失败: %s (继续推理, 但结果可能不正确)", e)

        action_dict, _info = self._policy._get_action(policy_obs)  # dict[(B, T, D)]

        # 合并所有 action key 的结果
        joint_seq = self._merge_action_keys(action_dict)

        # 缓存到队列 (最多 execution_horizon 步)
        n_cache = min(self.execution_horizon, joint_seq.shape[0])
        self._action_queue = [joint_seq[t].astype(np.float32) for t in range(n_cache)]

        # RELATIVE 模式: 记录累加起点 (执行第一步时的 joint_pos)
        self._action_queue_start_pos = np.asarray(
            obs["state.joint_pos"], dtype=np.float32
        ).copy()

        # 返回第一步
        return self._pop_cached_action(obs, ActionRepresentation)

    def _merge_action_keys(self, action_dict: dict[str, Any]) -> np.ndarray:
        """将多个 action key 的输出合并为一个 (T, num_joints) 数组.

        不同模型的 action key 组合不同:
        - real_g1_relative_eef_relative_joints: 9 个 key (eef + hand + arm + waist + ...)
        - unitree_g1_sonic: 3 个 key (motion_token + left_hand_joints + right_hand_joints)

        Args:
            action_dict: {key: (B, T, D)} 来自 Gr00tPolicy._get_action

        Returns:
            (T, num_joints) 合并后的关节动作序列
        """
        # 单 action key 情况: 直接使用
        if len(self._action_keys) == 1:
            first_key = self._action_keys[0]
            return action_dict[first_key][0]  # (T, D)

        # 多 action key 情况
        action_configs = self._policy.modality_configs["action"].action_configs
        rep = self._action_rep

        # 初始化输出 (T, num_joints), 用当前 default_angles 作为基准
        T = action_dict[self._action_keys[0]].shape[1]
        output = np.tile(self.default_angles, (T, 1))  # (T, 29)

        # 策略: 遍历 action_configs (有序), 按 state_key 映射到关节
        # action_configs 的顺序与 modality_keys 中的 action 部分对应
        for ac in action_configs:
            if ac is None:
                continue

            # 找到这个 action config 对应的 action_dict key
            # action_configs 的顺序对应 action modality_keys 中有 state_key 的部分
            action_key = ac.state_key
            if action_key is None:
                # state_key 为 null 的 key 是独立的 action output (如 motion_token)
                # 需要找到对应的 action_dict key
                continue

            # 在 action_dict 中查找对应的数据
            if action_key in action_dict:
                action_data = action_dict[action_key][0]  # (T, D)
                self._apply_action_to_joints(output, action_data, ac)

        # 处理 state_key 为 null 的 action key (如 motion_token)
        # 这些 key 的 action 数据不包含在 action_configs 中
        # 按优先级: 优先使用维度最高的 key (通常包含全身动作)
        null_state_keys = []
        for ak in self._action_keys:
            if ak not in action_dict:
                continue
            # 检查这个 key 是否已经被 action_configs 处理过
            handled = False
            for ac in action_configs:
                if ac is not None and ac.state_key == ak:
                    handled = True
                    break
            if not handled:
                null_state_keys.append(ak)

        # 按维度降序排列, 优先使用维度最高的 key
        null_state_keys.sort(key=lambda k: action_dict[k].shape[1], reverse=True)

        for ak in null_state_keys:
            action_data = action_dict[ak][0]  # (T, D)
            D = action_data.shape[1]
            if D >= self.num_joints:
                # 这个 key 包含全部关节, 直接使用
                output[:, :self.num_joints] = action_data[:, :self.num_joints]
                break  # 已覆盖全部关节, 不需要其他 key
            else:
                # 维度不够, 按顺序填充未覆盖的关节
                # 找到第一个未被 default_angles 覆盖的位置
                output[:, :D] = action_data

        return output

    def _apply_action_to_joints(
        self, output: np.ndarray, action_data: np.ndarray, ac: Any
    ) -> None:
        """将单个 action key 的数据映射到输出关节数组中.

        Args:
            output: (T, num_joints) 输出数组, 原地修改
            action_data: (T, D) 单个 action key 的数据
            ac: ActionConfig (含 rep, state_key 等)
        """
        from gr00t.data.types import ActionRepresentation

        sk = ac.state_key
        D = action_data.shape[1]

        # 根据 state_key 名称映射到 G1_JOINT_NAMES 的索引
        _joint_name_map = {}
        for i, name in enumerate(self._get_g1_joint_names()):
            _joint_name_map[name] = i

        # 解析 state_key 对应的关节范围
        _part_joint_names = {
            "left_leg": [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            ],
            "right_leg": [
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            ],
            "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            "left_arm": [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ],
            "left_hand": [f"left_hand_joint_{i}" for i in range(D)],
            "right_hand": [f"right_hand_joint_{i}" for i in range(D)],
        }

        joint_names = _part_joint_names.get(sk)
        if joint_names is None:
            return

        indices = []
        for jn in joint_names:
            if jn in _joint_name_map:
                indices.append(_joint_name_map[jn])

        n_apply = min(len(indices), D)
        if ac.rep == ActionRepresentation.RELATIVE:
            # RELATIVE: action_data 是 delta, 需要加到当前位置
            # 使用 default_angles 作为基准
            for i in range(n_apply):
                output[:, indices[i]] = output[:, indices[i]] * 0 + action_data[:, i]
                # 注意: 这里简化为直接使用 action_data 作为目标
                # 实际 RELATIVE 应该是 current_pos + delta, 但 current_pos 在 _pop_cached_action 中处理
        else:
            # ABSOLUTE: 直接作为目标
            for i in range(n_apply):
                output[:, indices[i]] = action_data[:, i]

    def _get_g1_joint_names(self) -> list[str]:
        """获取 G1 关节名称列表."""
        try:
            from configs.g1_config import G1_JOINT_NAMES
            return G1_JOINT_NAMES
        except ImportError:
            # fallback: 生成通用名称
            return [f"joint_{i}" for i in range(self.num_joints)]

    def _pop_cached_action(
        self, obs: dict[str, Any], ActionRepresentation: type
    ) -> np.ndarray:
        """从队列中取出一步并按 ActionRepresentation 转换为关节目标.

        对于多 action key 模型, _action_queue 中缓存的是已合并的 (num_joints,) 数组。
        """
        if not self._action_queue:
            raise RuntimeError("action queue empty — call get_action() first")

        # 取步索引 (累积 offset)
        idx = self.execution_horizon - len(self._action_queue)
        action_step = self._action_queue.pop(0)  # (num_joints,)

        rep = self._action_rep
        if rep == ActionRepresentation.RELATIVE:
            # RELATIVE: action_step 是 delta, 加到当前 joint_pos
            current_pos = np.asarray(obs["state.joint_pos"], dtype=np.float32)
            return current_pos + action_step
        # ABSOLUTE 或 DELTA: 直接作为目标
        return action_step.astype(np.float32)

    def reset_action_queue(self) -> None:
        """清空 action chunking 队列 (每次新 episode / reset 后调用)."""
        self._action_queue.clear()
        self._action_queue_start_pos = None

    def run_inference_loop(
        self,
        instruction: str = "walk forward",
        max_steps: int = 200,
        show_viewer: bool = False,
    ):
        self.instruction = instruction
        logger.info("=" * 60)
        logger.info("GR00T 本地推理 (基于 unitree_rl_mjlab)")
        logger.info("  指令: %s", instruction)
        logger.info("  任务: %s", self.task_id)
        logger.info("  最大步数: %d", max_steps)
        logger.info("=" * 60)

        # ── 尝试加载 unitree_rl_mjlab 环境 ────────────────────────────
        env = None
        try:
            import torch as _torch  # noqa: F401
            rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
            if rl_mjlab_root.exists() and str(rl_mjlab_root) not in sys.path:
                sys.path.insert(0, str(rl_mjlab_root))

            from mjlab.tasks.registry import load_env_cfg
            from mjlab.envs import ManagerBasedRlEnv

            # 触发任务注册
            import src.tasks  # noqa: F401

            env_cfg = load_env_cfg(self.task_id, play=True)
            # 推理用单环境
            env_cfg.scene.num_envs = 1
            # GR00T 使用 256x256 图像 (image_target_size in processor_config.json)
            env_cfg.viewer.width = 256
            env_cfg.viewer.height = 256
            env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device, render_mode="rgb_array")
            logger.info("✅ unitree_rl_mjlab 环境创建成功 (rgb_array)")
        except ImportError as e:
            logger.warning("依赖未安装: %s", e)
            logger.info("回退到纯推理模式 (无物理仿真)")
            env = None
        except Exception as e:
            logger.warning("mjlab 环境创建失败: %s", e)
            logger.info("回退到纯推理模式 (无物理仿真)")
            env = None

        # ── viser 3D viewer (可选) ──────────────────────────────────
        if self._viser and env is not None:
            try:
                from viser_3d_viewer import AsyncViser3DViewer
                self._viewer = AsyncViser3DViewer(env=env, port=self._viser_port)
                self._viewer.start()
                logger.info("🌐 Viser 3D viewer: %s", self._viewer.url)
            except ImportError:
                logger.warning("viser_3d_viewer 未安装, 跳过 3D 可视化")
            except Exception as e:
                logger.warning("Viser 3D viewer 启动失败: %s", e)

        # ── 推理循环 ─────────────────────────────────────────────────
        # 懒加载共享渲染 + obs 工具
        from mjlab_env import get_per_key_obs, render_frame  # type: ignore

        for step in range(max_steps):
            # 获取观测
            if env is not None:
                try:
                    if step == 0:
                        env.reset()
                        self.reset_action_queue()
                    obs = self._env_obs_to_dict(env, step)
                    # 从 mjlab env 拿一帧 RGB (256x256 for GR00T)
                    frame = render_frame(env, height=256, width=256)
                    if frame is not None:
                        # 使用正确的 video key (ego_view)
                        if self._policy is not None:
                            video_key = self._policy.modality_configs["video"].modality_keys[0]
                            obs[f"video.{video_key}"] = frame
                        else:
                            obs["video.ego_view"] = frame
                except Exception as e:
                    if step == 0:
                        logger.debug("mjlab obs 获取失败: %s", e)
                    obs = self._mock_observation(step)
            else:
                obs = self._mock_observation(step)

            # GR00T 推理
            try:
                joint_targets = self.get_action(obs)
            except Exception as e:
                logger.warning("GR00T 推理失败 (step %d): %s", step, e)
                joint_targets = self.default_angles.copy()

            # 执行动作
            if env is not None:
                try:
                    import torch as _torch
                    action_tensor = _torch.from_numpy(
                        joint_targets
                    ).unsqueeze(0).to(self.device)
                    env.step(action_tensor)
                except Exception as e:
                    if step == 0:
                        logger.debug("env.step 失败: %s", e)

            # ── viser 3D 更新 ─────────────────────────────────────
            if self._viewer is not None:
                self._viewer.update()

            if (step + 1) % 50 == 0:
                logger.info("推理进度: %d/%d", step + 1, max_steps)

        # 关闭 viewer + 环境
        if self._viewer is not None:
            self._viewer.stop()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        logger.info("✅ 推理完成! 共 %d 步", max_steps)

    def _env_obs_to_dict(self, env_or_obs: Any, step: int) -> dict[str, Any]:
        mock = self._mock_observation(step)
        if env_or_obs is None:
            return mock

        # 1) 是 env 实例: 用共享工具拆 per-key obs
        is_env = hasattr(env_or_obs, "unwrapped") and hasattr(
            env_or_obs.unwrapped, "observation_manager")
        if is_env:
            try:
                from mjlab_env import get_per_key_obs, to_numpy  # type: ignore
                raw = get_per_key_obs(env_or_obs)
            except Exception as e:
                logger.debug("get_per_key_obs 失败: %s", e)
                raw = {}
        # 2) 已是 per-key dict
        elif isinstance(env_or_obs, dict) or hasattr(env_or_obs, "get"):
            raw = dict(env_or_obs) if isinstance(env_or_obs, dict) else env_or_obs
        else:
            raw = {}

        def _np(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
            if arr.ndim >= 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            return arr.astype(np.float32)

        # 与 collect_data.py 保持对称, 否则训练/推理分布不一致
        # mjlab 返回的是 rel (=joint_pos - default), 还原为 abs
        jp_raw = raw.get("joint_pos_rel", raw.get("joint_pos"))
        jv_raw = raw.get("joint_vel_rel", raw.get("joint_vel"))
        bp = raw.get("base_pos") or raw.get("root_link_pos_w")
        bq = raw.get("base_quat") or raw.get("root_link_quat_w")
        blv = raw.get("base_lin_vel") or raw.get("root_link_lin_vel_w")
        bav = raw.get("base_ang_vel") or raw.get("root_link_ang_vel_w")

        if jp_raw is not None:
            jp = _np(jp_raw)
            # mjlab 返回的是 rel (=joint_pos - default), 还原为绝对位置
            mock["state.joint_pos"] = jp + self.default_angles
        if jv_raw is not None:
            mock["state.joint_vel"] = _np(jv_raw)
        if bp is not None:
            mock["state.base_pos"] = _np(bp)
        if bq is not None:
            mock["state.base_quat"] = _np(bq)
        if blv is not None:
            mock["state.base_lin_vel"] = _np(blv)
        if bav is not None:
            mock["state.base_ang_vel"] = _np(bav)

        return mock

    def _mock_observation(self, step: int) -> dict[str, Any]:
        """生成模拟观测 (无仿真环境时使用)。

        注意: 使用 ego_view 而非 front_view, 256x256 图像尺寸。
        """
        t = step * 0.02
        # 确定性种子, 保证可复现
        rng = np.random.RandomState(42 + step)
        return {
            "video.ego_view": np.zeros((256, 256, 3), dtype=np.uint8),
            "state.joint_pos": self.default_angles +
                rng.randn(self.num_joints).astype(np.float32) * 0.01,
            "state.joint_vel": np.zeros(self.num_joints, dtype=np.float32),
            "state.base_pos": np.array(
                [0.5 * t, 0.0, 0.8 if self.robot == "g1" else 0.32], dtype=np.float32
            ),
            "state.base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "state.base_lin_vel": np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "state.base_ang_vel": np.zeros(3, dtype=np.float32),
            "annotation.language.action_text": "walk forward",
        }


def main():
    parser = argparse.ArgumentParser(description="GR00T 本地推理包装器")
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "go2"])
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--task", type=str, default=None,
                        help="unitree_rl_mjlab 任务 ID (默认: Unitree-{Robot}-Flat)")
    parser.add_argument("--instruction", type=str, default="walk forward")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--quantize", type=str, default="auto",
                        choices=["auto", "none", "4bit", "8bit"],
                        help="(历史参数, 已忽略 — 推理 dtype 由模型保存时决定)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--viser", action="store_true",
                        help="启用 Viser 浏览器 3D 可视化 (默认端口 20006)")
    parser.add_argument("--viser-port", type=int, default=20006,
                        help="Viser 服务器端口 (默认: 20006)")
    parser.add_argument("--execution-horizon", type=int, default=1,
                        help="Action chunking: 每次 GR00T 输出多步动作, "
                             "顺序执行前 N 步再重新规划 (1=每步规划)")
    parser.add_argument("--embodiment-tag", type=str, default=None,
                        help="指定 embodiment tag (默认从模型 processor_config.json 自动检测)")
    args = parser.parse_args()

    inference = GR00TLocalInference(
        model_path=args.model_path,
        robot=args.robot,
        quantize=args.quantize,
        device=args.device,
        execution_horizon=args.execution_horizon,
        task_id=args.task,
        instruction=args.instruction,
        viser=args.viser,
        viser_port=args.viser_port,
        embodiment_tag=args.embodiment_tag,
    )
    inference.run_inference_loop(
        instruction=args.instruction,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
    )


if __name__ == "__main__":
    main()
