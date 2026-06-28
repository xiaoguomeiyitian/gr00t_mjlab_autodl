"""
infer.py — 本地推理包装器。

包装 Isaac-GR00T Gr00tPolicy，提供简洁的推理接口。
支持 BF16 / INT4 / INT8 量化模型，自动检测。

用法:
    # 在本地机器上（需要 torch + Isaac-GR00T 环境）
    from src.infer import GR00TLocalInference

    inference = GR00TLocalInference(model_path="./checkpoints/g1_int4")
    action = inference.predict(images=..., state=..., language="walk forward")
    inference.close()
"""

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
        """取出下一个动作。"""
        if self.buffer is None or self.cursor >= len(self.buffer):
            return None
        action = self.buffer[self.cursor]
        self.cursor += 1
        return action

    def push(self, actions: np.ndarray):
        """推入一组新动作。"""
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
    ):
        """
        初始化本地推理。

        Args:
            model_path: 模型路径（本地目录）
            embodiment_tag: 具身标签
            device: 设备 ("auto" / "cuda" / "cpu")
            action_horizon: 动作预测步数
            num_obs_steps: 观测历史步数
        """
        self.model_path = Path(model_path)
        self.embodiment_tag = embodiment_tag
        self.action_horizon = action_horizon
        self.num_obs_steps = num_obs_steps
        self._closed = False

        # 检测设备
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 检测量化模式
        self.quant_mode = self._detect_quant_mode()

        # 动作缓冲区
        self._action_buffer = ActionChunkBuffer()

        # 加载模型
        self.policy = self._load_policy()

    def _detect_quant_mode(self) -> str:
        """从模型目录检测量化模式。"""
        # 检查是否有 .quant 文件
        quant_files = list(self.model_path.glob("*.quant"))
        if quant_files:
            return "int4_lut"

        # 检查 BitsAndBytes 配置
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
        """加载 Isaac-GR00T Gr00tPolicy。"""
        import sys

        # 确保 Isaac-GR00T 在 sys.path 中
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
        """
        执行推理，返回动作。

        Args:
            images: {"camera_name": np.ndarray (H,W,3) uint8}
            state: (state_dim,) float32 关节状态
            language: 语言指令
            use_buffer: 是否使用动作缓冲区（action chunking）

        Returns:
            action: (action_dim,) float32
            info: dict 包含推理信息
        """
        t0 = time.time()

        # 尝试从缓冲区取动作
        if use_buffer:
            buffered_action = self._action_buffer.pop()
            if buffered_action is not None:
                return buffered_action, {
                    "latency_ms": 0,
                    "source": "buffer",
                }

        # 构建观测
        observation = self._build_observation(images, state, language)

        # 调用 policy
        import torch
        with torch.no_grad():
            result = self.policy.get_action(observation)

        # 解析结果
        if isinstance(result, tuple):
            action_data = result[0]
            extra_info = result[1] if len(result) > 1 else {}
        elif isinstance(result, dict):
            action_data = result.get("action", result)
            extra_info = result
        else:
            action_data = result
            extra_info = {}

        # 提取动作数组
        action = self._extract_action(action_data)

        # 填充缓冲区（如果有多个时间步）
        if use_buffer and isinstance(action_data, np.ndarray) and action_data.ndim == 2 and action_data.shape[0] > 1:
            self._action_buffer.push(action_data[1:])
            first_action = action_data[0]
        else:
            first_action = action if isinstance(action, np.ndarray) else np.array(action)

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
        """构建 GR00T 观测格式。"""
        video = {}
        for key, img in images.items():
            video[key] = img

        return {
            "video": video,
            "state": state.reshape(1, -1),
            "language": language,
        }

    def _extract_action(self, action_data) -> np.ndarray:
        """从推理结果中提取动作数组。"""
        if isinstance(action_data, np.ndarray):
            if action_data.ndim == 2:
                return action_data[0]
            return action_data
        elif isinstance(action_data, dict):
            for key in ["action", "joint_position_delta", "joint_position"]:
                if key in action_data:
                    val = action_data[key]
                    if isinstance(val, np.ndarray) and val.ndim == 2:
                        return val[0]
                    return val
        elif isinstance(action_data, (list, tuple)):
            return np.array(action_data[0]) if len(action_data) > 0 else np.zeros(29)
        return np.zeros(29)

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
