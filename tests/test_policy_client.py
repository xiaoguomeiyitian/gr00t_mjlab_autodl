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
