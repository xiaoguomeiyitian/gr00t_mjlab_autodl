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


class TestDequantizeRoundTrip:
    """反量化 + round-trip 测试。"""

    def test_round_trip_shape(self):
        """反量化后形状与原始一致。"""
        from src.quantize_safetensors import dequantize_from_nf4
        weight = np.random.randn(64, 128).astype(np.float32)
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        assert deq.shape == weight.shape

    def test_round_trip_mse(self):
        """round-trip 量化误差合理（MSE 应远小于原始方差）。"""
        from src.quantize_safetensors import dequantize_from_nf4
        np.random.seed(42)
        weight = np.random.randn(256, 512).astype(np.float32)
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        mse = np.mean((weight - deq) ** 2)
        var = np.var(weight)
        # 量化误差应远小于原始方差
        assert mse < var * 0.1, f"MSE={mse}, var={var}"

    def test_round_trip_odd_columns(self):
        """奇数列 round-trip（测试 padding 分支）。"""
        from src.quantize_safetensors import dequantize_from_nf4
        weight = np.random.randn(32, 65).astype(np.float32)  # 奇数列
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        assert deq.shape == (32, 65)
        mse = np.mean((weight - deq) ** 2)
        var = np.var(weight)
        assert mse < var * 0.1

    def test_round_trip_small_block(self):
        """列数 < BLOCK_SIZE 的边界（单块）。"""
        from src.quantize_safetensors import dequantize_from_nf4
        weight = np.random.randn(8, 1).astype(np.float32)  # 1 列
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        assert deq.shape == (8, 1)

    def test_round_trip_63_columns(self):
        """列数略小于 BLOCK_SIZE。"""
        from src.quantize_safetensors import dequantize_from_nf4
        weight = np.random.randn(16, 63).astype(np.float32)
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        assert deq.shape == (16, 63)

    def test_round_trip_zero_weight(self):
        """全零权重 round-trip。"""
        from src.quantize_safetensors import dequantize_from_nf4
        weight = np.zeros((32, 64), dtype=np.float32)
        packed, absmax = quantize_to_nf4(weight)
        deq = dequantize_from_nf4(packed, absmax, weight.shape)
        # 全零权重反量化后应接近 0
        assert np.max(np.abs(deq)) < 1e-5

    def test_packed_nibble_range(self):
        """packed 值每个 nibble 在 0-15 范围内。"""
        weight = np.random.randn(64, 128).astype(np.float32)
        packed, _ = quantize_to_nf4(weight)
        # 低 nibble
        assert np.all((packed & 0x0F) >= 0)
        assert np.all((packed & 0x0F) <= 15)
        # 高 nibble
        assert np.all((packed >> 4) >= 0)
        assert np.all((packed >> 4) <= 15)

    def test_absmax_correctness(self):
        """absmax 等于块内真实最大绝对值。"""
        weight = np.abs(np.random.randn(4, 128).astype(np.float32))
        # 构造已知 absmax：每块设为不同量级
        weight[:, :64] *= 10
        weight[:, 64:128] *= 0.1
        _, absmax = quantize_to_nf4(weight)
        # 块 0 的 absmax 应接近 10，块 1 接近 0.1
        assert absmax[0, 0] > 5  # 块 0
        assert absmax[0, 1] < 1  # 块 1


class TestQuantizeSafetensorsFile:
    """quantize_safetensors_file 文件级测试。"""

    def test_file_level_quantize(self, temp_dir):
        """文件级量化：跳过 1D，量化 2D。"""
        from src.quantize_safetensors import quantize_safetensors_file
        from safetensors.numpy import save_file, safe_open

        model_path = temp_dir / "model.safetensors"
        weights = {
            "linear.weight": np.random.randn(64, 128).astype(np.float32),
            "linear.bias": np.random.randn(64).astype(np.float32),  # 1D 跳过
        }
        save_file(weights, str(model_path))

        out_path = temp_dir / "model_int4.safetensors"
        stats = quantize_safetensors_file(str(model_path), str(out_path), verbose=False)

        assert stats["total_tensors"] == 2
        assert stats["quantized_tensors"] == 1
        assert stats["skipped_tensors"] == 1
        assert out_path.exists()

        # 验证输出 key 命名
        with safe_open(str(out_path), framework="numpy") as f:
            keys = list(f.keys())
        assert "linear.weight.quant" in keys
        assert "linear.weight.absmax" in keys
        assert "linear.weight.shape" in keys
        assert "linear.bias" in keys  # 跳过的原样保留
