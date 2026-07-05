"""测试 ZMQ 客户端模块。"""

import pytest
import numpy as np

from src.policy_client import _MsgSerializer, GR00TClient


class TestMsgSerializer:
    """_MsgSerializer 测试。"""

    def test_roundtrip_dict(self):
        data = {"key": "value", "number": 42}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        assert deserialized == data

    def test_roundtrip_ndarray(self):
        data = {"array": np.array([1.0, 2.0, 3.0])}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        np.testing.assert_array_equal(deserialized["array"], data["array"])

    def test_roundtrip_nested(self):
        data = {
            "video": {"front": np.zeros((224, 224, 3), dtype=np.uint8)},
            "state": np.ones(71, dtype=np.float32),
            "language": "walk forward",
        }
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        np.testing.assert_array_equal(deserialized["state"], data["state"])
        assert deserialized["language"] == "walk forward"

    def test_roundtrip_large_array(self):
        data = {"large": np.random.randn(1000, 1000).astype(np.float32)}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        np.testing.assert_array_almost_equal(deserialized["large"], data["large"])

    def test_empty_dict(self):
        data = {}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        assert deserialized == {}

    def test_none_value(self):
        data = {"key": None}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        assert deserialized["key"] is None


class TestGR00TClient:
    """GR00TClient 测试（不需要实际服务器）。"""

    def test_init(self):
        client = GR00TClient(host="127.0.0.1", port=5555)
        assert client.host == "127.0.0.1"
        assert client.port == 5555
        assert client.timeout_ms == 30000
        client.close()

    def test_init_custom_timeout(self):
        client = GR00TClient(host="127.0.0.1", port=5556, timeout_ms=5000)
        assert client.port == 5556
        assert client.timeout_ms == 5000
        client.close()

    def test_close(self):
        client = GR00TClient()
        client.close()
        assert client._closed

    def test_context_manager(self):
        with GR00TClient() as client:
            assert not client._closed
        assert client._closed

    def test_serializer_internal(self):
        """测试内部序列化器。"""
        client = GR00TClient()
        data = {"test": np.array([1, 2, 3])}
        serialized = _MsgSerializer.to_bytes(data)
        deserialized = _MsgSerializer.from_bytes(serialized)
        np.testing.assert_array_equal(deserialized["test"], data["test"])
        client.close()


class TestGR00TClientMock:
    """用 mock socket 测试 _call_endpoint 核心路径。"""

    def test_call_endpoint_success(self, monkeypatch):
        """正常请求-响应路径。"""
        import src.policy_client as pc

        client = GR00TClient.__new__(GR00TClient)
        client._closed = False
        client.host = "127.0.0.1"
        client.port = 5555
        client.timeout_ms = 30000

        # mock zmq context + socket
        class FakeSocket:
            def __init__(self):
                self.sent = None
                self.recv_data = _MsgSerializer.to_bytes({"result": "ok"})

            def setsockopt(self, *a, **kw):
                pass

            def send(self, data):
                self.sent = data

            def recv(self):
                return self.recv_data

            def close(self, linger=0):
                pass

            def connect(self, addr):
                pass

        fake_socket = FakeSocket()

        class FakeContext:
            def socket(self, sock_type):
                return fake_socket

            def term(self):
                pass

        client.context = FakeContext()
        client.socket = fake_socket

        result = client._call_endpoint("get_action", {"observation": {}})
        assert result == {"result": "ok"}

    def test_call_endpoint_error_response(self, monkeypatch):
        """server 返回 b"ERROR" 时抛 RuntimeError 并重建 socket。"""
        import src.policy_client as pc

        client = GR00TClient.__new__(GR00TClient)
        client._closed = False
        client.host = "127.0.0.1"
        client.port = 5555
        client.timeout_ms = 30000

        rebuilt = {"count": 0}

        class FakeSocket:
            def __init__(self, recv_data=b"ERROR"):
                self.recv_data = recv_data
                self.closed = False

            def setsockopt(self, *a, **kw):
                pass

            def send(self, data):
                pass

            def recv(self):
                return self.recv_data

            def close(self, linger=0):
                self.closed = True
                rebuilt["count"] += 1

            def connect(self, addr):
                pass

        current_socket = FakeSocket(b"ERROR")

        class FakeContext:
            def __init__(self):
                self.socket_type = None

            def socket(self, sock_type):
                return current_socket

            def term(self):
                pass

        client.context = FakeContext()
        client.socket = current_socket

        with pytest.raises(RuntimeError, match="Server error"):
            client._call_endpoint("ping", requires_input=False)

        # ERROR 后应重建 socket（关闭旧的）
        assert rebuilt["count"] >= 1

    def test_call_endpoint_error_dict(self, monkeypatch):
        """server 返回含 error key 的 dict 时抛 RuntimeError。"""
        client = GR00TClient.__new__(GR00TClient)
        client._closed = False
        client.host = "127.0.0.1"
        client.port = 5555
        client.timeout_ms = 30000

        class FakeSocket:
            def __init__(self):
                self.recv_data = _MsgSerializer.to_bytes({"error": "model not loaded"})

            def setsockopt(self, *a, **kw):
                pass

            def send(self, data):
                pass

            def recv(self):
                return self.recv_data

            def close(self, linger=0):
                pass

            def connect(self, addr):
                pass

        fake_socket = FakeSocket()

        class FakeContext:
            def socket(self, sock_type):
                return fake_socket

            def term(self):
                pass

        client.context = FakeContext()
        client.socket = fake_socket

        with pytest.raises(RuntimeError, match="model not loaded"):
            client._call_endpoint("get_action", {"observation": {}})

    def test_ping_returns_false_on_timeout(self, monkeypatch):
        """ping 超时时返回 False 并重建 socket。"""
        import zmq

        client = GR00TClient.__new__(GR00TClient)
        client._closed = False
        client.host = "127.0.0.1"
        client.port = 5555
        client.timeout_ms = 100

        closed = {"count": 0}

        class FakeSocket:
            def setsockopt(self, *a, **kw):
                pass

            def send(self, data):
                raise zmq.error.Again()

            def recv(self):
                raise zmq.error.Again()

            def close(self, linger=0):
                closed["count"] += 1

            def connect(self, addr):
                pass

        current_socket = FakeSocket()

        class FakeContext:
            def socket(self, sock_type):
                return current_socket

            def term(self):
                pass

        client.context = FakeContext()
        client.socket = current_socket

        result = client.ping()
        assert result is False
        # 超时后应关闭旧 socket
        assert closed["count"] >= 1
