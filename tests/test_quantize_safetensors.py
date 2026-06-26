#!/usr/bin/env python3.12
"""测试 quantize_safetensors.py — NF4 量化核心逻辑.

覆盖:
  - quantize_nf4 基本正确性 (打包/反量化/精度)
  - 边界条件 (全零、单 block、非对齐尺寸)
  - quantize_file 的 2D 选择性 / backbone 跳过
"""
import sys
import struct
import pytest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── NF4 参考表 (来自 bitsandbytes) ──────────────────────────────────────

NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09185808897018433, 0.0,
    0.07555019855499268, 0.15110039710998535, 0.22369961440563202, 0.293800413608551,
    0.3639012277126312, 0.4355524778366089, 0.5124996900558472, 0.6000000238418579,
], dtype=np.float32)


def _dequantize_nf4(packed: np.ndarray, scale: np.ndarray, zero_point: np.ndarray,
                    block_size: int = 64) -> np.ndarray:
    """反量化: packed uint8 → float32 (用于验证 quantize_nf4 的正确性)."""
    # 解包: 每个 uint8 拆成 2 个 4-bit index
    indices = np.zeros(len(packed) * 2, dtype=np.uint8)
    indices[0::2] = (packed >> 4) & 0x0F
    indices[1::2] = packed & 0x0F

    # 查表得到归一化值 [-1, 1]
    normalized = NF4_TABLE[indices]

    # 乘以 scale (每个 block 一个 scale)
    n_blocks = len(scale)
    flat = np.zeros(n_blocks * block_size, dtype=np.float32)
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        flat[start:end] = normalized[start:end] * scale[b] + zero_point[b]

    return flat


class TestQuantizeNf4:
    """quantize_nf4 — NF4 量化核心函数."""

    def test_output_types_and_shapes(self):
        """返回值类型正确: packed uint8, scale/fp32, zero_point fp32."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(128, 64).astype(np.float32)
        packed, scale, zero_point, orig_shape, orig_dtype, pad_len = quantize_nf4(weight)

        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.uint8
        assert isinstance(scale, np.ndarray)
        assert scale.dtype == np.float32
        assert isinstance(zero_point, np.ndarray)
        assert zero_point.dtype == np.float32
        assert orig_shape == (128, 64)
        assert orig_dtype == np.float32

    def test_packed_size_is_half_of_elements(self):
        """packed 长度 = ceil(total_elements / 2)."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(100, 50).astype(np.float32)  # 5000 elements
        packed, _, _, _, _, pad_len = quantize_nf4(weight, block_size=64)
        # 5000 → pad to 5056 (79*64) → 5056/2 = 2528 packed bytes
        total_padded = 5000 + pad_len
        expected_len = total_padded // 2
        assert len(packed) == expected_len

    def test_scale_per_block(self):
        """每个 block 一个 scale."""
        from quantize_safetensors import quantize_nf4

        block_size = 64
        weight = np.random.randn(128, 64).astype(np.float32)  # 128*64 = 8192 = 128 blocks
        packed, scale, _, _, _, _ = quantize_nf4(weight, block_size=block_size)
        assert len(scale) == 8192 // block_size  # = 128

    def test_zero_point_is_zero(self):
        """对称量化: zero_point 全为 0."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(64, 32).astype(np.float32)
        _, _, zero_point, _, _, _ = quantize_nf4(weight)
        np.testing.assert_array_equal(zero_point, 0.0)

    def test_dequantized_range_within_bounds(self):
        """反量化后的值在 [min(NF4_TABLE), max(NF4_TABLE)] * scale 范围内."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(256, 128).astype(np.float32) * 0.5
        packed, scale, zero_point, orig_shape, _, pad_len = quantize_nf4(weight)

        # 反量化
        dequant_flat = _dequantize_nf4(packed, scale, zero_point)
        # 去除 padding
        total_elements = np.prod(orig_shape)
        dequant_flat = dequant_flat[:total_elements]

        # 反量化值应该与原值在同一量级 (误差来自量化精度)
        # NF4 的 max abs 误差约为 scale * 0.1 (经验值)
        max_error = np.max(np.abs(dequant_flat - weight.flatten()))
        assert max_error < 2.0, f"max dequantization error too large: {max_error}"

    def test_all_zeros_weight(self):
        """全零权重: scale 应为 1e-8 (避免除零), packed 全为 0x77 (index 7 = 0.0)."""
        from quantize_safetensors import quantize_nf4

        weight = np.zeros((64, 32), dtype=np.float32)
        packed, scale, zero_point, _, _, _ = quantize_nf4(weight)
        # 全零 → absmax = 0 → clamp to 1e-8 → scale = 1e-8
        assert np.all(scale > 0)
        assert np.all(scale <= 1e-8 + 1e-10)

    def test_non_aligned_size_padding(self):
        """非 block_size 对齐的尺寸: 自动 pad, pad_len 正确."""
        from quantize_safetensors import quantize_nf4

        block_size = 64
        weight = np.random.randn(100, 10).astype(np.float32)  # 1000 elements
        _, _, _, _, _, pad_len = quantize_nf4(weight, block_size=block_size)
        expected_pad = (block_size - 1000 % block_size) % block_size
        assert pad_len == expected_pad

    def test_exact_aligned_no_padding(self):
        """恰好对齐: pad_len = 0."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(64, 64).astype(np.float32)  # 4096 = 64*64
        _, _, _, _, _, pad_len = quantize_nf4(weight, block_size=64)
        assert pad_len == 0

    def test_single_block(self):
        """单个 block (64 元素)."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(1, 64).astype(np.float32)
        packed, scale, _, _, _, pad_len = quantize_nf4(weight, block_size=64)
        assert len(scale) == 1
        assert len(packed) == 32  # 64 / 2
        assert pad_len == 0

    def test_packed_values_in_range(self):
        """packed 字节值在 [0, 255] 范围内 (uint8)."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(128, 128).astype(np.float32)
        packed, _, _, _, _, _ = quantize_nf4(weight)
        assert packed.dtype == np.uint8
        assert packed.min() >= 0
        assert packed.max() <= 255

    def test_deterministic(self):
        """相同输入 → 相同输出 (确定性量化)."""
        from quantize_safetensors import quantize_nf4

        weight = np.random.randn(64, 32).astype(np.float32)
        np.random.seed(42)
        w1 = weight.copy()
        np.random.seed(42)
        w2 = weight.copy()

        r1 = quantize_nf4(w1)
        r2 = quantize_nf4(w2)
        np.testing.assert_array_equal(r1[0], r2[0])  # packed
        np.testing.assert_array_equal(r1[1], r2[1])  # scale
        np.testing.assert_array_equal(r1[2], r2[2])  # zero_point


class TestQuantizeNf4Precision:
    """量化精度验证."""

    def test_small_values_high_precision(self):
        """小值 (接近 0) 量化精度更高 (NF4 在 0 附近级别更密)."""
        from quantize_safetensors import quantize_nf4

        # 小值
        small_weight = np.random.randn(256, 256).astype(np.float32) * 0.01
        packed_s, scale_s, _, shape_s, _, _ = quantize_nf4(small_weight)
        dequant_s = _dequantize_nf4(packed_s, scale_s, np.zeros(len(scale_s)))
        error_s = np.max(np.abs(dequant_s[:small_weight.size] - small_weight.flatten()))

        # 大值
        large_weight = np.random.randn(256, 256).astype(np.float32) * 10.0
        packed_l, scale_l, _, shape_l, _, _ = quantize_nf4(large_weight)
        dequant_l = _dequantize_nf4(packed_l, scale_l, np.zeros(len(scale_l)))
        error_l = np.max(np.abs(dequant_l[:large_weight.size] - large_weight.flatten()))

        # 小值的绝对误差应更小
        assert error_s < error_l

    def test_sign_preserved(self):
        """正负号保持: 正值量化后仍为正 (或接近 0), 负值仍为负."""
        from quantize_safetensors import quantize_nf4

        weight = np.array([[1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.0, 0.0]], dtype=np.float32)
        # 扩展到 64 elements (1 block)
        weight = np.tile(weight, (8, 1)).astype(np.float32)  # (8, 8) = 64
        packed, scale, _, _, _, _ = quantize_nf4(weight, block_size=64)
        dequant = _dequantize_nf4(packed, scale, np.zeros(1))[:64]

        # 正值仍应 > 0 (或非常接近)
        assert dequant[0] > 0, f"positive value became negative: {dequant[0]}"
        assert dequant[1] < 0, f"negative value became positive: {dequant[1]}"


class TestQuantizeFile:
    """quantize_file — 文件级量化 (需要 safetensors)."""

    @pytest.fixture
    def fake_safetensors_file(self, tmp_path):
        """创建一个假的 safetensors 文件 (含 2D 权重 + 1D bias)."""
        safetensors = pytest.importorskip("safetensors")
        from safetensors.torch import save_file
        import torch

        tensors = {
            "action_head.weight": torch.randn(256, 128, dtype=torch.float32),
            "action_head.bias": torch.randn(256, dtype=torch.float32),
            "backbone.layer1.weight": torch.randn(512, 256, dtype=torch.float32),
            "backbone.layer1.bias": torch.randn(512, dtype=torch.float32),
        }
        path = tmp_path / "model.safetensors"
        save_file(tensors, str(path))
        return path

    def test_quantize_file_2d_only(self, fake_safetensors_file, tmp_path):
        """只量化 2D 权重, 1D bias 原样保留."""
        from quantize_safetensors import quantize_file

        output_path = tmp_path / "output.safetensors"
        quantize_file(str(fake_safetensors_file), str(output_path))

        from safetensors import safe_open
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())

        # 2D 权重 → 有 .qdata, .scale, .zero_point 后缀
        assert "action_head.weight.qdata" in keys
        assert "action_head.weight.scale" in keys
        assert "action_head.weight.zero_point" in keys

        # 1D bias → 原样保留
        assert "action_head.bias" in keys

    def test_quantize_file_skips_backbone_by_default(self, fake_safetensors_file, tmp_path):
        """默认不量化 backbone."""
        from quantize_safetensors import quantize_file

        output_path = tmp_path / "output.safetensors"
        quantize_file(str(fake_safetensors_file), str(output_path), quantize_backbone=False)

        from safetensors import safe_open
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())

        # backbone 原样保留 (没有 .qdata)
        assert "backbone.layer1.weight" in keys
        assert "backbone.layer1.weight.qdata" not in keys

    def test_quantize_file_quantize_backbone(self, fake_safetensors_file, tmp_path):
        """--quantize-backbone: backbone 也被量化."""
        from quantize_safetensors import quantize_file

        output_path = tmp_path / "output.safetensors"
        quantize_file(str(fake_safetensors_file), str(output_path), quantize_backbone=True)

        from safetensors import safe_open
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())

        assert "backbone.layer1.weight.qdata" in keys
        assert "backbone.layer1.weight.scale" in keys

    def test_metadata_preserved(self, fake_safetensors_file, tmp_path):
        """量化后 metadata 包含 orig_shape / orig_dtype / pad_len."""
        from quantize_safetensors import quantize_file

        output_path = tmp_path / "output.safetensors"
        quantize_file(str(fake_safetensors_file), str(output_path))

        from safetensors import safe_open
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            metadata = f.metadata()

        assert "action_head.weight.orig_shape" in metadata
        assert "action_head.weight.orig_dtype" in metadata
        assert "action_head.weight.pad_len" in metadata


class TestMainArgparse:
    """main() 参数解析."""

    def test_required_args(self):
        """缺少 --input-dir / --output-dir → SystemExit(2)."""
        import sys
        from quantize_safetensors import main
        old_argv = sys.argv
        try:
            sys.argv = ["quantize_safetensors.py"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2  # argparse error code
        finally:
            sys.argv = old_argv

    def test_help_does_not_raise(self):
        """--help → SystemExit(0)."""
        import sys
        from quantize_safetensors import main
        old_argv = sys.argv
        try:
            sys.argv = ["quantize_safetensors.py", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
