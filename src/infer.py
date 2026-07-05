"""infer.py — 本地推理包装器。"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


class ActionChunkBuffer:
    """动作块缓冲区：GR00T 一次预测多步，逐步消费。"""

    def __init__(self):
        self.buffer: Optional[np.ndarray] = None
        self.cursor: int = 0

    def pop(self) -> Optional[np.ndarray]:
        if self.buffer is None or self.cursor >= len(self.buffer):
            return None
        action = self.buffer[self.cursor]
        self.cursor += 1
        return action

    def push(self, actions: np.ndarray):
        self.buffer = actions
        self.cursor = 0

    @property
    def is_empty(self) -> bool:
        return self.buffer is None or self.cursor >= len(self.buffer)

    def clear(self):
        self.buffer = None
        self.cursor = 0


class GR00TLocalInference:
    """GR00T 本地推理封装。"""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "auto",
        action_horizon: int = 16,
        num_obs_steps: int = 1,
        robot: str = "g1",
    ):
        self.model_path = Path(model_path)
        self.embodiment_tag = embodiment_tag
        self.action_horizon = action_horizon
        self.num_obs_steps = num_obs_steps
        self.robot = robot
        self._closed = False

        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.quant_mode = self._detect_quant_mode()
        self._action_buffer = ActionChunkBuffer()
        self.policy = self._load_policy()

    def _detect_quant_mode(self) -> str:
        # 检测 safetensors 内是否有 .quant 后缀的 key（LUT 量化标志）
        try:
            from safetensors import safe_open
            for sf in self.model_path.glob("*.safetensors"):
                with safe_open(str(sf), framework="numpy") as f:
                    for key in f.keys():
                        if key.endswith(".quant"):
                            return "int4_lut"
                        if key.endswith(".absmax"):
                            return "int4_lut"
        except Exception:
            pass

        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            quantization_config = config.get("quantization_config", {})
            if quantization_config.get("load_in_4bit"):
                return "int4_bnb"
            if quantization_config.get("load_in_8bit"):
                return "int8"

        return "none"

    def _load_policy(self):
        import sys

        isaac_path = str(self.model_path.parent.parent / "Isaac-GR00T")
        if os.path.isdir(isaac_path) and isaac_path not in sys.path:
            sys.path.insert(0, isaac_path)

        try:
            from gr00t.policy.server_client import Gr00tPolicy

            print(f"📦 加载模型: {self.model_path}")
            print(f"   量化模式: {self.quant_mode}")
            print(f"   设备: {self.device}")

            policy = Gr00tPolicy(
                model_path=str(self.model_path),
                device=self.device,
                embodiment_tag=self.embodiment_tag,
            )

            print(f"✅ 模型加载完成")
            return policy

        except ImportError as e:
            print(f"❌ 无法导入 Isaac-GR00T: {e}")
            print(f"   请确保 Isaac-GR00T 已安装")
            raise

    def predict(
        self,
        images: dict,
        state: np.ndarray,
        language: str = "perform the task",
        use_buffer: bool = True,
    ) -> tuple:
        t0 = time.time()

        if use_buffer:
            buffered_action = self._action_buffer.pop()
            if buffered_action is not None:
                return buffered_action, {
                    "latency_ms": 0,
                    "source": "buffer",
                }

        observation = self._build_observation(images, state, language)

        # torch 仅用于 no_grad 上下文，若未安装则跳过
        try:
            import torch
            with torch.no_grad():
                result = self.policy.get_action(observation)
        except ModuleNotFoundError:
            result = self.policy.get_action(observation)

        if isinstance(result, tuple):
            action_data = result[0]
            extra_info = result[1] if len(result) > 1 else {}
        elif isinstance(result, dict):
            action_data = result.get("action", result)
            extra_info = result
        else:
            action_data = result
            extra_info = {}

        # 统一提取完整 action chunk（horizon, action_dim），由 predict 决定取哪步
        action_chunk = self._extract_action_chunk(action_data)

        if use_buffer and action_chunk.ndim == 2 and action_chunk.shape[0] > 1:
            # 缓存剩余步，逐步消费
            self._action_buffer.push(action_chunk[1:])
            first_action = action_chunk[0]
        else:
            first_action = action_chunk if action_chunk.ndim == 1 else action_chunk[0]

        latency_ms = (time.time() - t0) * 1000

        info = {
            "latency_ms": latency_ms,
            "source": "model",
            "quant_mode": self.quant_mode,
            "device": self.device,
            **extra_info,
        }

        return first_action, info

    def _build_observation(self, images: dict, state: np.ndarray, language: str) -> dict:
        """构建 GR00T 观测格式（复用 ObservationBuilder，符合 check_observation 契约）。"""
        from src.observation_builder import ObservationBuilder, state_slices_from_config
        if not hasattr(self, "_obs_builder") or self._obs_builder is None:
            state_dim = state.shape[-1] if state is not None else 71
            self._obs_builder = ObservationBuilder(
                camera_keys=list(images.keys()) if images else None,
                state_dim=state_dim,
                state_slices=state_slices_from_config(self.robot),
                num_obs_steps=self.num_obs_steps,
            )
        return self._obs_builder.build(images=images, state=state, language=language)

    def _extract_action_chunk(self, action_data) -> np.ndarray:
        """从推理结果中提取完整 action chunk（horizon, action_dim）或 (action_dim,)。"""
        if isinstance(action_data, np.ndarray):
            return action_data
        elif isinstance(action_data, dict):
            for key in ["action", "joint_position_delta", "joint_position"]:
                if key in action_data:
                    val = action_data[key]
                    if isinstance(val, np.ndarray):
                        return val
                    return np.atleast_1d(np.array(val))
            # dict 本身可能是单步
            return np.atleast_1d(np.array(list(action_data.values())[0]))
        elif isinstance(action_data, (list, tuple)):
            arr = np.array(action_data)
            return arr if arr.ndim >= 1 else arr.reshape(1)
        return np.zeros(self.action_horizon if self.action_horizon else 1, dtype=np.float32)

    def _extract_action(self, action_data) -> np.ndarray:
        """从推理结果中提取单步动作（兼容旧接口）。"""
        chunk = self._extract_action_chunk(action_data)
        if chunk.ndim == 2 and chunk.shape[0] > 0:
            return chunk[0]
        return chunk

    def reset_buffer(self):
        """重置动作缓冲区。"""
        self._action_buffer.clear()

    def close(self):
        """关闭推理。"""
        if not self._closed:
            if hasattr(self, "policy") and self.policy is not None:
                try:
                    self.policy.close()
                except Exception:
                    pass
            self._closed = True
            print("🔒 推理已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
