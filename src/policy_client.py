"""
GR00T Policy Client — 纯 ZMQ + msgpack 实现，不依赖 Isaac-GR00T 导入。

直接实现与云端 Policy Server 的通信协议，无需安装 torch/transformers 等重量级依赖。

用法:
    client = GR00TClient(host="127.0.0.1", port=5555)
    modality_config = client.get_modality_config()
    action, info = client.get_action(obs)
    client.close()
"""

from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class _MsgSerializer:
    """msgpack_numpy 序列化器（与 Isaac-GR00T PolicyServer 兼容）。"""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=mnp.encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


class GR00TClient:
    """GR00T 云端推理客户端（纯 ZMQ 实现）。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 30000):
        """
        初始化客户端。

        Args:
            host: Policy Server 地址（通过 SSH 隧道后通常为 127.0.0.1）
            port: Policy Server 端口
            timeout_ms: 请求超时（毫秒）
        """
        self._closed = False
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()
        print(f"✅ 已连接 GR00T Policy Server ({host}:{port})")

    def _init_socket(self):
        """初始化/重建 socket"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call_endpoint(self, endpoint: str, data: dict = None, requires_input: bool = True) -> Any:
        """调用云端 endpoint"""
        request = {"endpoint": endpoint}
        if requires_input and data is not None:
            request["data"] = data

        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise TimeoutError(f"请求超时 ({self.timeout_ms}ms)，请确认云端 Server 已启动且 SSH 隧道正常")

        if message == b"ERROR":
            raise RuntimeError("Server error. 请确认云端运行了正确的 Policy Server。")

        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        """检查服务器是否可达"""
        try:
            self._call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, TimeoutError):
            self._init_socket()
            return False

    def get_modality_config(self) -> dict:
        """获取模态配置（state/action/video key 映射）"""
        return self._call_endpoint("get_modality_config", requires_input=False)

    def get_action(self, obs: dict, options: dict = None) -> tuple:
        """
        发送观测，接收动作。

        Args:
            obs: 观测字典，格式:
                {
                    "video": {"camera_key": np.ndarray},  # (H, W, 3) uint8
                    "state": np.ndarray,                   # (1, state_dim) float32
                    "language": {"language_instruction": [["指令"]]}
                }
            options: 可选参数

        Returns:
            (action, info): 动作数组 + 附加信息
        """
        response = self._call_endpoint(
            "get_action",
            {"observation": obs, "options": options},
        )
        return tuple(response) if isinstance(response, list) else response

    def close(self):
        """关闭连接"""
        if self._closed:
            return
        self._closed = True
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass
        print("🔌 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
