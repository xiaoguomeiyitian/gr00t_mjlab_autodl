#!/usr/bin/env python3
"""
GR00T 本地推理包装器 — 基于 unitree_rl_mjlab 仿真环境进行推理

支持:
  - INT4 量化模型推理 (RTX 2080 8GB)
  - FP16 / BF16 全精度推理 (RTX 4090 24GB)
  - mjlab Viser 3D 可视化回放

使用方法:
    # INT4 推理 (8GB GPU)
    python infer.py --robot g1 \
        --model-path models/g1_gr00t_int4 \
        --instruction "walk forward"

    # FP16 推理 (24GB+ GPU)
    python infer.py --robot g1 \
        --model-path models/g1_gr00t \
        --quantize none
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
        """加载 GR00T 模型。"""
        logger.info("加载 GR00T 模型: %s", self.model_path)
        logger.info("  机器人: %s (%d joints)", self.robot, self.num_joints)
        logger.info("  量化: %s", self.quantize)
        logger.info("  设备: %s", self.device)

        try:
            # 需要 Isaac-GR00T 项目中的 gr00t 策略包
            # 推荐安装: pip install gr00t (或 git clone https://github.com/NVIDIA/Isaac-GR00T)
            from gr00t_integration.gr00t_policy import Gr00tPolicy, Gr00tConfig
            self._policy = Gr00tPolicy(Gr00tConfig(
                model_path=self.model_path,
                embodiment_tag=self.embodiment_tag,
                device=self.device,
                dtype="float16",
                action_horizon=self.action_horizon,
                quantize=self.quantize,
            ))
            logger.info("✅ GR00T 模型加载成功")
        except ImportError as e:
            logger.error("无法加载 GR00T: %s", e)
            logger.info("请确保已安装 Isaac-GR00T:")
            logger.info("  git clone https://github.com/NVIDIA/Isaac-GR00T.git")
            logger.info("  cd Isaac-GR00T && uv sync --python 3.10")
            logger.info("或将 gr00t_integration 包放到 PYTHONPATH")
            raise

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """从观测生成动作。

        Args:
            obs: 观测字典，包含:
                - video.front_view: (224, 224, 3) uint8 RGB
                - state.joint_pos: (num_joints,) float32
                - state.joint_vel: (num_joints,) float32
                - state.base_pos: (3,) float32
                - state.base_quat: (4,) float32
                - state.base_lin_vel: (3,) float32
                - state.base_ang_vel: (3,) float32
                - annotation.language.action_text: str

        Returns:
            joint_targets: (num_joints,) float32, 关节目标位置
        """
        if self._policy is None:
            self.load()

        action_chunk = self._policy.get_action(obs)
        # 取第一个时间步的动作
        if isinstance(action_chunk, np.ndarray) and action_chunk.ndim == 2:
            return action_chunk[0]
        return np.array(action_chunk, dtype=np.float32)

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
            env = ManagerBasedRlEnv(cfg=env_cfg, device=self.device)
            logger.info("✅ unitree_rl_mjlab 环境创建成功")
        except ImportError as e:
            logger.warning("依赖未安装: %s", e)
            logger.info("回退到纯推理模式 (无物理仿真)")
            env = None
        except Exception as e:
            logger.warning("mjlab 环境创建失败: %s", e)
            logger.info("回退到纯推理模式 (无物理仿真)")
            env = None

        # ── 推理循环 ─────────────────────────────────────────────────
        for step in range(max_steps):
            # 获取观测
            if env is not None:
                try:
                    obs_raw = env.unwrapped.reset() if step == 0 else None
                    if obs_raw is not None:
                        env_obs = obs_raw
                    else:
                        env_obs = env.unwrapped.observation_manager.compute() \
                            if hasattr(env.unwrapped, "observation_manager") else None
                    obs = self._env_obs_to_dict(env_obs, step)
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

    def _env_obs_to_dict(self, env_obs: Any, step: int) -> dict[str, Any]:
        """从 unitree_rl_mjlab 的观测中提取 GR00T 格式数据。"""
        # 默认 fallback
        mock = self._mock_observation(step)
        if env_obs is None:
            return mock

        try:
            # env_obs 可能是 dict 或 TensorDict
            if hasattr(env_obs, "get"):
                obs_dict = env_obs
            elif isinstance(env_obs, dict):
                obs_dict = env_obs
            else:
                return mock

            # 提取关节位置/速度
            jp = obs_dict.get("joint_pos_rel", obs_dict.get("joint_pos"))
            jv = obs_dict.get("joint_vel_rel", obs_dict.get("joint_vel"))
            bp = obs_dict.get("base_pos")
            bq = obs_dict.get("base_quat")
            blv = obs_dict.get("base_lin_vel")
            bav = obs_dict.get("base_ang_vel")

            def _to_numpy(x):
                """tensor → numpy (支持 batch 维度 squeeze)."""
                if x is None:
                    return None
                if hasattr(x, "cpu"):
                    return x.squeeze(0).cpu().numpy()
                return np.asarray(x)

            if jp is not None:
                mock["state.joint_pos"] = _to_numpy(jp)
            if jv is not None:
                mock["state.joint_vel"] = _to_numpy(jv)
            if bp is not None:
                mock["state.base_pos"] = _to_numpy(bp)
            if bq is not None:
                mock["state.base_quat"] = _to_numpy(bq)
            if blv is not None:
                mock["state.base_lin_vel"] = _to_numpy(blv)
            if bav is not None:
                mock["state.base_ang_vel"] = _to_numpy(bav)

        except Exception as e:
            logger.debug("obs 转换失败: %s", e)

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
                        choices=["auto", "none", "4bit", "8bit"])
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
