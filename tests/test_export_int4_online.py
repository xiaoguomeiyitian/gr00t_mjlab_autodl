"""测试 export_int4.py（在线 INT4 量化主入口，LUT 路径）。"""

import numpy as np
import pytest
from safetensors.numpy import save_file, safe_open

from src.export_int4 import export_int4


def _make_model(temp_dir, with_bias=True):
    """创建模拟模型目录（含 safetensors）。"""
    model_dir = temp_dir / "model"
    model_dir.mkdir(exist_ok=True)
    weights = {
        "linear.weight": np.random.randn(64, 128).astype(np.float32),
    }
    if with_bias:
        weights["linear.bias"] = np.random.randn(64).astype(np.float32)
    save_file(weights, str(model_dir / "model.safetensors"))
    return model_dir


class TestExportInt4:
    """export_int4 函数测试（device=cpu 走 LUT 路径）。"""

    def test_lut_path_basic(self, temp_dir):
        """device=cpu 时走 LUT 路径。"""
        model_dir = _make_model(temp_dir)
        out_dir = temp_dir / "out"

        stats = export_int4(
            model_path=str(model_dir),
            output_dir=str(out_dir),
            device="cpu",
            verbose=False,
        )

        assert stats["method"] == "lut_nf4"
        assert stats["quantized_tensors"] >= 1
        assert out_dir.exists()
        # 输出含 safetensors
        assert list(out_dir.glob("*.safetensors"))

    def test_lut_skips_1d_and_bias(self, temp_dir):
        """1D 张量（bias）和含 bias 模式的 key 被跳过。"""
        model_dir = _make_model(temp_dir, with_bias=True)
        out_dir = temp_dir / "out"

        stats = export_int4(
            model_path=str(model_dir),
            output_dir=str(out_dir),
            device="cpu",
            verbose=False,
        )

        # linear.weight (2D) 量化，linear.bias (1D + bias 模式) 跳过
        assert stats["quantized_tensors"] == 1
        assert stats["skipped_tensors"] == 1

    def test_lut_output_keys(self, temp_dir):
        """输出 safetensors 含 .quant/.absmax/.shape key。"""
        model_dir = _make_model(temp_dir, with_bias=False)
        out_dir = temp_dir / "out"

        export_int4(
            model_path=str(model_dir),
            output_dir=str(out_dir),
            device="cpu",
            verbose=False,
        )

        out_sf = list(out_dir.glob("*.safetensors"))[0]
        with safe_open(str(out_sf), framework="numpy") as f:
            keys = list(f.keys())
        assert "linear.weight.quant" in keys
        assert "linear.weight.absmax" in keys
        assert "linear.weight.shape" in keys

    def test_empty_model_raises(self, temp_dir):
        """空模型目录抛 FileNotFoundError。"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        out_dir = temp_dir / "out"

        with pytest.raises(FileNotFoundError):
            export_int4(
                model_path=str(empty_dir),
                output_dir=str(out_dir),
                device="cpu",
                verbose=False,
            )

    def test_default_output_dir(self, temp_dir):
        """未指定 output_dir 时自动创建 <model>_int4。"""
        model_dir = _make_model(temp_dir, with_bias=False)
        # 不传 output_dir
        stats = export_int4(
            model_path=str(model_dir),
            device="cpu",
            verbose=False,
        )
        expected = model_dir.parent / "model_int4"
        assert expected.exists()
        assert stats["method"] == "lut_nf4"
