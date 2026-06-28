"""测试 NF4 量化核心模块。"""

import numpy as np
import pytest

from src.quantize_safetensors import quantize_to_nf4, NF4_TABLE, BLOCK_SIZE


class TestNF4Table:
    """NF4 查找表测试。"""

    def test_table_length(self):
        assert len(NF4_TABLE) == 16

    def test_table_range(self):
        assert NF4_TABLE.min() == pytest.approx(-1.0)
        assert NF4_TABLE.max() == pytest.approx(1.0)

    def test_table_sorted(self):
        for i in range(len(NF4_TABLE) - 1):
            assert NF4_TABLE[i] < NF4_TABLE[i + 1]

    def test_table_symmetric_zero(self):
        """NF4 表包含 0。"""
        assert 0.0 in NF4_TABLE


class TestBlockSize:
    """块大小常量测试。"""

    def test_block_size_positive(self):
        assert BLOCK_SIZE > 0

    def test_block_size_power_of_two(self):
        assert BLOCK_SIZE & (BLOCK_SIZE - 1) == 0


class TestQuantizeToNF4:
    """quantize_to_nf4 函数测试。"""

    def test_output_shape_aligned(self):
        """对齐的矩阵（列数是 BLOCK_SIZE 倍数）。"""
        weight = np.random.randn(128, 256).astype(np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        assert quantized.shape == (128, 128)  # 256 / 2
        assert absmax.shape == (128, 4)  # 256 / 64

    def test_output_shape_non_aligned(self):
        """非对齐的矩阵。"""
        weight = np.random.randn(100, 100).astype(np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        assert quantized.shape[0] == 100
        assert quantized.shape[1] == 50  # 100 / 2

    def test_output_shape_small(self):
        """极小矩阵。"""
        weight = np.random.randn(2, 3).astype(np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        assert quantized.shape[0] == 2

    def test_quantized_dtype(self):
        weight = np.random.randn(64, 64).astype(np.float32)
        quantized, _ = quantize_to_nf4(weight)
        assert quantized.dtype == np.uint8

    def test_absmax_positive(self):
        weight = np.random.randn(64, 64).astype(np.float32)
        _, absmax = quantize_to_nf4(weight)
        assert np.all(absmax > 0)

    def test_compression_ratio(self):
        """验证压缩比合理。"""
        weight = np.random.randn(256, 512).astype(np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        original = weight.nbytes
        compressed = quantized.nbytes + absmax.nbytes
        ratio = original / compressed
        assert 3.0 < ratio < 10.0

    def test_different_dtypes(self):
        """支持 float16 / bfloat16 / float32 输入。"""
        for dtype in [np.float32, np.float16]:
            weight = np.random.randn(32, 64).astype(dtype)
            quantized, absmax = quantize_to_nf4(weight)
            assert quantized.shape == (32, 32)

    def test_deterministic(self):
        """相同输入产生相同输出。"""
        weight = np.random.randn(32, 64).astype(np.float32)
        q1, a1 = quantize_to_nf4(weight)
        q2, a2 = quantize_to_nf4(weight)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_zero_weight(self):
        """全零权重不崩溃。"""
        weight = np.zeros((32, 64), dtype=np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        assert quantized.shape == (32, 32)

    def test_large_weight(self):
        """大矩阵不崩溃。"""
        weight = np.random.randn(1024, 1024).astype(np.float32)
        quantized, absmax = quantize_to_nf4(weight)
        assert quantized.shape == (1024, 512)
