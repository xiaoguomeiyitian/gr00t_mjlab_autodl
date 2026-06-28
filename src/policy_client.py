"""
GR00T Policy Client — 封装与云端 Policy Server 的 ZMQ 通信。

用法:
    client = GR00TClient(host="127.0.0.1", port=5555)
    modality_config = client.get_modality_config()
    action, info = client.get_action(obs)
    client.close()
"""

import os
import sys
from typing import Optional

import numpy as np

# 自动查找 Isaac-GR00T 路径
_ISAAC_GR00T_PATH = os.path.join(os.path.dirname(__file__), "../../Isaac-GR00T")
if os.path.exists(_ISAAC_GR00T_PATH) and _ISAAC_GR00T_PATH not in sys.path:
    sys.path.insert(0, _ISAAC_GR00T_PATH)

from gr00t.policy.server_client import PolicyClient


class GR00TClient:
    """GR00T 云端推理客户端。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: int = 30000):
        """
        初始化客户端。

        Args:
            host: Policy Server 地址（通过 SSH 隧道后通常为 127.0.0.1）
            port: Policy Server 端口
            timeout: 请求超时（毫秒）
        """
        self.client = PolicyClient(host=host, port=port, timeout=timeout)
        print(f"✅ 已连接 GR00T Policy Server ({host}:{port})")

    def get_modality_config(self) -> dict:
        """获取模态配置（state/action/video key 映射）"""
        return self.client.get_modality_config()

    def get_action(self, obs: dict) -> tuple:
        """
        发送观测，接收动作。

        Args:
            obs: 观测字典，格式:
                {
                    "video": {"camera_key": np.ndarray},  # (H, W, 3) uint8
                    "state": np.ndarray,                   # (state_dim,) float32
                    "language": {"language_instruction": [["指令"]]}
                }

        Returns:
            (action, info): 动作数组 (action_dim,) + 附加信息
        """
        return self.client.get_action(obs)

    def close(self):
        """关闭连接"""
        self.client.close()
        print("🔌 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
