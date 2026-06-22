#!/usr/bin/env python3
"""
GR00T 本地推理包装器 — 基于 unitree_rl_mjlab 仿真环境进行推理

支持:
  - INT4 量化模型推理 (RTX 2080 8GB)
  - FP16 / BF16 全精度推理 (RTX 4090 24GB)
  - mjlab Viser 3D 可视化回放

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
from typing import Any

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

    def __init__(
        self,
        model_path: str,
        robot: str = "g1",
        quantize: str = "auto",
        device: str = "auto",
        action_horizon: int = 16,
        task_id: str | None = None,
    ):
        self.model_path = model_path
        self.robot = robot
        self.quantize = quantize
        self.device = device
        self.action_horizon = action_horizon
        self.task_id = task_id or (
            "Unitree-G1-Flat" if robot == "g1" else "Unitree-Go2-Flat"
        )

        # 加载配置
        if robot == "g1":
            from configs.g1_config import (
                G1_NUM_JOINTS, G1_EMBODIMENT_TAG,
                G1_DEFAULT_JOINT_ANGLES, G1_JOINT_NAMES,
            )
            self.num_joints = G1_NUM_JOINTS
            self.embodiment_tag = G1_EMBODIMENT_TAG
            self.default_angles = np.array(
                [G1_DEFAULT_JOINT_ANGLES[n] for n in G1_JOINT_NAMES], dtype=np.float32
            )
        else:
            from configs.go2_config import (
                GO2_NUM_JOINTS, GO2_EMBODIMENT_TAG,
                GO2_DEFAULT_JOINT_ANGLES, GO2_JOINT_NAMES,
            )
            self.num_joints = GO2_NUM_JOINTS
            self.embodiment_tag = GO2_EMBODIMENT_TAG
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

        self._policy = None

    def load(self):
        """加载 GR00T 模型 (Isaac-GR00T N1.7 Gr00tPolicy)."""
        logger.info("加载 GR00T 模型: %s", self.model_path)
        logger.info("  机器人: %s (%d joints)", self.robot, self.num_joints)
        logger.info("  量化: %s", self.quantize)
        logger.info("  设备: %s", self.device)

        try:
            # 官方 Isaac-GR00T (N1.7) 策略实现
            from gr00t.policy.gr00t_policy import Gr00tPolicy
            from gr00t.data.embodiment_tags import EmbodimentTag
        except ImportError as e:
            logger.error("无法导入 Isaac-GR00T: %s", e)
            logger.info("请确保已安装 Isaac-GR00T 并将 gr00t/ 加入 PYTHONPATH:")
            logger.info("  export PYTHONPATH=/root/unitree/Isaac-GR00T:$PYTHONPATH")
            raise

        # 注意: GR00T 模型保存时已经包含量化信息 (INT4/BF16/FP16),
        # 不需要在加载时再次指定 dtype 或 quantize。
        # Gr00tPolicy 签名: (embodiment_tag, model_path, *, device, strict)
        self._policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=str(self.model_path),
            device=str(self.device),
            strict=True,
        )
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

        Returns:
            dict, 三个顶级 key:
              video:   {view_name: np.ndarray (B=1, T=1, H, W, 3) uint8}
              state:   {state_name: np.ndarray (B=1, T=1, D) float32}
              language:{lang_name: list[list[str]]  shape (B=1, T=1)}
        """
        # ── video ──
        frame = obs.get("video.front_view")
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if frame.ndim == 3:
            frame = frame[None, None, ...]  # (H,W,3) -> (1,1,H,W,3)
        elif frame.ndim == 4:
            frame = frame[None, ...]        # (T,H,W,3) -> (1,T,H,W,3)
        elif frame.ndim == 5 and frame.shape[0] != 1:
            frame = frame[:1]
        video = {"front_view": frame.astype(np.uint8)}

        # ── state ──
        # Gr00tPolicy 要求 state[key] 形状 (B, T, D), 这里 B=T=1
        def _to_btd(x):
            arr = np.asarray(x, dtype=np.float32)
            return arr.reshape(1, 1, -1)

        state = {
            "joint_pos": _to_btd(obs["state.joint_pos"]),
            "joint_vel": _to_btd(obs["state.joint_vel"]),
            "base_pos":  _to_btd(obs["state.base_pos"]),
            "base_quat": _to_btd(obs["state.base_quat"]),
            "base_lin_vel": _to_btd(obs["state.base_lin_vel"]),
            "base_ang_vel": _to_btd(obs["state.base_ang_vel"]),
        }

        # ── language ──
        # 需要与 ModalityConfig.language.modality_keys[0] 一致
        lang_key = self._policy.language_key
        instruction = obs.get(
            "annotation.language.action_text",
            obs.get("instruction", "walk forward"),
        )
        language = {lang_key: [[str(instruction)]]}

        return {"video": video, "state": state, "language": language}

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """从观测生成动作 (单步关节目标).

        Args:
            obs: 单步观测字典, 同上 _build_policy_observation 输入格式

        Returns:
            joint_targets: (num_joints,) float32, 关节目标位置
        """
        if self._policy is None:
            self.load()

        policy_obs = self._build_policy_observation(obs)
        action_dict, _info = self._policy._get_action(policy_obs)  # dict[np.ndarray (B,T,D)]

        # 输出是 (B, T, D), 取 [0, 0, :]
        first_key = self._action_keys[0]
        delta = action_dict[first_key][0, 0]  # (D_action,)

        # 当前 joint_pos 用于 RELATIVE 模式
        current_pos = np.asarray(obs["state.joint_pos"], dtype=np.float32)

        # ActionRepresentation: RELATIVE / ABSOLUTE / DELTA
        from gr00t.data.types import ActionRepresentation  # local import
        rep = self._action_rep
        if rep == ActionRepresentation.RELATIVE:
            return current_pos + delta
        # ABSOLUTE 或 DELTA 直接作为目标
        return delta.astype(np.float32)

    def run_inference_loop(
        self,
        instruction: str = "walk forward",
        max_steps: int = 200,
        show_viewer: bool = False,
    ):
        """运行完整推理循环 (在 unitree_rl_mjlab 仿真中)。

        Args:
            instruction: 语言指令
            max_steps: 最大步数
            show_viewer: 是否显示 mjlab 可视化
        """
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
            # 默认 224x224 视频尺寸 (匹配 GR00T)
            env_cfg.viewer.width = 224
            env_cfg.viewer.height = 224
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

        # ── 推理循环 ─────────────────────────────────────────────────
        # 懒加载共享渲染 + obs 工具
        from mjlab_env import get_per_key_obs, render_frame  # type: ignore

        for step in range(max_steps):
            # 获取观测
            if env is not None:
                try:
                    if step == 0:
                        env.reset()
                    obs = self._env_obs_to_dict(env, step)
                    # ⭐ 真实渲染: 从 mjlab env 拿一帧 RGB
                    frame = render_frame(env, height=224, width=224)
                    if frame is not None:
                        obs["video.front_view"] = frame
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

            if (step + 1) % 50 == 0:
                logger.info("推理进度: %d/%d", step + 1, max_steps)

        # 关闭环境
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        logger.info("✅ 推理完成! 共 %d 步", max_steps)

    def _env_obs_to_dict(self, env_or_obs: Any, step: int) -> dict[str, Any]:
        """从 unitree_rl_mjlab env/obs 提取 GR00T 格式数据.

        输入可以是:
          - ManagerBasedRlEnv 实例 → 通过 get_per_key_obs() 拆 term
          - dict / TensorDict → 已有 per-key 观测
          - None → fallback mock
        """
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

        # joint_pos (相对或绝对)
        jp = raw.get("joint_pos_rel", raw.get("joint_pos"))
        jv = raw.get("joint_vel_rel", raw.get("joint_vel"))
        bp = raw.get("base_pos") or raw.get("root_link_pos_w")
        bq = raw.get("base_quat") or raw.get("root_link_quat_w")
        blv = raw.get("base_lin_vel") or raw.get("root_link_lin_vel_w")
        bav = raw.get("base_ang_vel") or raw.get("root_link_ang_vel_w")

        if jp is not None:
            mock["state.joint_pos"] = _np(jp)
        if jv is not None:
            mock["state.joint_vel"] = _np(jv)
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
        """生成模拟观测 (无仿真环境时使用)。"""
        t = step * 0.02
        return {
            "video.front_view": np.zeros((224, 224, 3), dtype=np.uint8),
            "state.joint_pos": self.default_angles +
                np.random.randn(self.num_joints).astype(np.float32) * 0.01,
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
    args = parser.parse_args()

    inference = GR00TLocalInference(
        model_path=args.model_path,
        robot=args.robot,
        quantize=args.quantize,
        device=args.device,
        task_id=args.task,
    )
    inference.run_inference_loop(
        instruction=args.instruction,
        max_steps=args.max_steps,
        show_viewer=args.show_viewer,
    )


if __name__ == "__main__":
    main()
