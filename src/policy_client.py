"""
GR00T Policy Client — 纯 ZMQ + msgpack 实现，不依赖 Isaac-GR00T 导入。
"""

from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


class _MsgSerializer:
    """msgpack_numpy 序列化器（与 Isaac-GR00T PolicyServer 兼容）。

    对齐官方 gr00t.policy.server_client.MsgSerializer 的安全防护：
    拒绝 object-dtype ndarray，避免 msgpack_numpy 走 pickle 路径。
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        import functools

        def _safe_encode(obj, chain=None):
            if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
                raise TypeError(
                    f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                    f"msgpack_numpy would invoke pickle. Convert to a concrete "
                    f"numeric dtype before sending."
                )
            return mnp.encode(obj, chain=chain)

        default = functools.partial(_safe_encode, chain=None)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        import functools

        def _safe_decode(obj, chain=None):
            if isinstance(obj, dict):
                nd_val = obj.get(b"nd", obj.get("nd"))
                kind_val = obj.get(b"kind", obj.get("kind"))
                if nd_val and kind_val in (b"O", "O"):
                    raise ValueError(
                        "Refusing to decode object-dtype ndarray payload (pickle-bearing); "
                        "convert to a concrete numeric dtype before sending."
                    )
            return mnp.decode(obj, chain=chain)

        object_hook = functools.partial(_safe_decode, chain=None)
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)


class GR00TClient:
    """GR00T 云端推理客户端（纯 ZMQ 实现）。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 30000):
        self._closed = False
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()
        print(f"✅ 已连接 GR00T Policy Server ({host}:{port})")

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call_endpoint(self, endpoint: str, data: dict = None, requires_input: bool = True) -> Any:
        request = {"endpoint": endpoint}
        if requires_input and data is not None:
            request["data"] = data

        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # 超时后必须关闭旧 socket 再重建，避免句柄泄漏
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self._init_socket()
            raise TimeoutError(f"请求超时 ({self.timeout_ms}ms)，请确认云端 Server 已启动且 SSH 隧道正常")

        if message == b"ERROR":
            # REQ socket 在收到 ERROR 后状态机可能异常，重建 socket 保证下次请求可用
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self._init_socket()
            raise RuntimeError("Server error. 请确认云端运行了正确的 Policy Server。")

        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self._call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, TimeoutError):
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self._init_socket()
            return False

    def get_modality_config(self) -> dict:
        return self._call_endpoint("get_modality_config", requires_input=False)

    def get_action(self, obs: dict, options: dict = None) -> tuple:
        response = self._call_endpoint(
            "get_action",
            {"observation": obs, "options": options},
        )
        return tuple(response) if isinstance(response, list) else response

    def close(self):
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
