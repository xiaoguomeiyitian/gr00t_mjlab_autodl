"""测试 INT4 量化导出模块。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.export_int4_offline import export_int4_offline


class TestExportInt4Offline:
    """export_int4_offline 测试。"""

    def test_quantizes_2d_weights(self, sample_safetensors):
        output_dir = str(sample_safetensors.parent / "int4_output")
        stats = export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            verbose=False,
        )
        assert stats["quantized"] == 3  # 3 个 2D 权重
        assert stats["skipped"] == 2    # 2 个 1D 层

    def test_creates_output_dir(self, sample_safetensors):
        output_dir = str(sample_safetensors.parent / "int4_output2")
        export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            verbose=False,
        )
        assert os.path.isdir(output_dir)

    def test_creates_quantized_safetensors(self, sample_safetensors):
        output_dir = str(sample_safetensors.parent / "int4_output3")
        export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            verbose=False,
        )
        safetensors_files = list(Path(output_dir).glob("*.safetensors"))
        assert len(safetensors_files) == 1

    def test_compression_ratio(self, sample_safetensors):
        output_dir = str(sample_safetensors.parent / "int4_output4")
        stats = export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            verbose=False,
        )
        assert stats["compression"] > 1.0

    def test_copies_config(self, sample_safetensors):
        output_dir = str(sample_safetensors.parent / "int4_output5")
        export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            verbose=False,
        )
        assert (Path(output_dir) / "config.json").exists()

    def test_exclude_backbone(self, sample_safetensors):
        """排除 backbone 时不应量化 cosmos/reason 等层。"""
        # 添加 backbone 权重
        from safetensors.numpy import save_file
        backbone_weights = {
            "cosmos.encoder.weight": np.random.randn(64, 128).astype(np.float32),
            "reason.attn.weight": np.random.randn(64, 128).astype(np.float32),
        }
        save_file(backbone_weights, str(sample_safetensors / "backbone.safetensors"))

        output_dir = str(sample_safetensors.parent / "int4_output6")
        stats = export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            exclude_backbone=True,
            verbose=False,
        )
        # backbone 层应被跳过: cosmos + reason + layernorm + bias = 4
        assert stats["skipped"] >= 4

    def test_no_exclude_backbone(self, sample_safetensors):
        """不排除 backbone 时应量化更多层。"""
        from safetensors.numpy import save_file
        backbone_weights = {
            "cosmos.encoder.weight": np.random.randn(64, 128).astype(np.float32),
        }
        save_file(backbone_weights, str(sample_safetensors / "backbone.safetensors"))

        output_dir = str(sample_safetensors.parent / "int4_output7")
        stats = export_int4_offline(
            model_path=str(sample_safetensors),
            output_dir=output_dir,
            exclude_backbone=False,
            verbose=False,
        )
        # 应量化 4 层（3 original + 1 backbone）
        assert stats["quantized"] == 4

    def test_missing_model_dir(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            export_int4_offline(
                model_path=str(temp_dir / "nonexistent"),
                verbose=False,
            )

    def test_no_safetensors_files(self, temp_dir):
        empty_dir = temp_dir / "empty_model"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="未找到 safetensors"):
            export_int4_offline(
                model_path=str(empty_dir),
                verbose=False,
            )
