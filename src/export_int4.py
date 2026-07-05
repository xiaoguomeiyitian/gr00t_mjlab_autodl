"""export_int4.py — INT4 量化导出主入口。"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

# 共享的排除模式（norm/bias/embedding 等不量化的层）
_EXCLUDE_PATTERNS = (
    "layernorm", "layer_norm", "bias", "embedding",
    "patch_embed", "pos_embed", "cls_token",
)


def export_int4(
    model_path: str,
    output_dir: Optional[str] = None,
    quantize_backbone: bool = False,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / f"{model_path.name}_int4"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # BitsAndBytes 4-bit 量化需要 CUDA，CPU 环境直接走查找表方案
    use_bnb = device != "cpu"

    if verbose:
        print(f"📦 INT4 量化导出")
        print(f"   输入模型: {model_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   设备: {device}")
        print(f"   量化 backbone: {quantize_backbone}")

    try:
        if not use_bnb:
            raise ImportError("device=cpu 不支持 BitsAndBytes 4-bit，使用查找表方案")
        stats = _export_via_bnb(
            model_path=str(model_path), output_dir=str(output_dir),
            quantize_backbone=quantize_backbone, device=device, verbose=verbose,
        )
    except ImportError:
        if verbose:
            print("⚠️  BitsAndBytes 不可用，使用查找表方案")
        stats = _export_via_lut(
            model_path=str(model_path), output_dir=str(output_dir), verbose=verbose,
        )

    return stats


def _export_via_bnb(
    model_path: str, output_dir: str, quantize_backbone: bool = False,
    device: str = "cuda", verbose: bool = True,
) -> dict:
    from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    if verbose:
        print("  🔄 加载模型（4-bit 量化模式）...")

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )

    if verbose:
        print("  ✅ 模型加载完成")

    if verbose:
        print("  💾 保存量化模型...")

    model.save_pretrained(output_dir)

    for f in ["processor_config.json", "preprocessor_config.json", "config.json"]:
        src = Path(model_path) / f
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(Path(output_dir) / f))

    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(output_dir)
        for f in fn
    )

    stats = {
        "method": "bitsandbytes_4bit",
        "total_params": total_params,
        "output_size_mb": total_size / (1024 * 1024),
        "output_dir": output_dir,
    }

    if verbose:
        print(f"\n📊 量化完成:")
        print(f"   方法: BitsAndBytes NF4")
        print(f"   参数量: {total_params:,}")
        print(f"   输出大小: {stats['output_size_mb']:.1f} MB")

    return stats


def _export_via_lut(
    model_path: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """通过查找表进行 safetensors 级别 NF4 量化（无需 transformers）。"""
    from safetensors import safe_open
    from safetensors.numpy import save_file
    from src.quantize_safetensors import quantize_to_nf4

    model_path = Path(model_path)

    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"未找到 safetensors 文件: {model_path}")

    if verbose:
        print(f"  📂 找到 {len(safetensors_files)} 个 safetensors 文件")

    total_quantized = 0
    total_skipped = 0

    for sf_path in safetensors_files:
        tensors = {}
        file_quantized = 0
        file_skipped = 0
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                should_skip = any(p in key.lower() for p in _EXCLUDE_PATTERNS) or tensor.ndim != 2

                if should_skip:
                    tensors[key] = tensor
                    file_skipped += 1
                else:
                    q, absmax = quantize_to_nf4(tensor)
                    tensors[f"{key}.quant"] = q
                    tensors[f"{key}.absmax"] = absmax
                    tensors[f"{key}.shape"] = np.array(tensor.shape, dtype=np.int32)
                    file_quantized += 1

        out_path = Path(output_dir) / sf_path.name
        save_file(tensors, str(out_path))
        total_quantized += file_quantized
        total_skipped += file_skipped
        if verbose:
            print(f"  ✅ {sf_path.name}: {file_quantized} quantized, {file_skipped} skipped")

    # 复制配置文件
    import shutil
    for f in ["config.json", "processor_config.json", "preprocessor_config.json"]:
        src = model_path / f
        if src.exists():
            shutil.copy2(str(src), str(Path(output_dir) / f))

    total_size = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fn in os.walk(output_dir)
        for fn in fn
    )

    stats = {
        "method": "lut_nf4",
        "quantized_tensors": total_quantized,
        "skipped_tensors": total_skipped,
        "output_size_mb": total_size / (1024 * 1024),
        "output_dir": output_dir,
    }

    if verbose:
        print(f"\n📊 量化完成:")
        print(f"   方法: 查找表 NF4")
        print(f"   已量化: {total_quantized}")
        print(f"   跳过: {total_skipped}")
        print(f"   输出大小: {stats['output_size_mb']:.1f} MB")

    return stats


def main():
    parser = argparse.ArgumentParser(description="INT4 量化导出")
    parser.add_argument("--model-path", type=str, required=True,
                        help="BF16 模型路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--quantize-backbone", action="store_true",
                        help="量化 backbone（默认只量化 diffusion head）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto/cuda/cpu)")
    args = parser.parse_args()

    export_int4(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantize_backbone=args.quantize_backbone,
        device=args.device,
    )


if __name__ == "__main__":
    import numpy as np  # needed for _export_via_lut
    main()
