"""
export_int4.py — INT4 量化导出主入口。

使用 BitsAndBytesConfig + AutoModel 进行后训练量化（PTQ）。
输入：BF16 模型（~7GB）
输出：INT4 量化模型（~1.5GB，5-15 分钟）

用法:
    python -m src.export_int4 --model-path ./checkpoints/g1_finetune --output-dir ./checkpoints/g1_int4
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def export_int4(
    model_path: str,
    output_dir: Optional[str] = None,
    quantize_backbone: bool = False,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    将 BF16 模型量化为 INT4。

    Args:
        model_path: BF16 模型路径（本地目录或 HuggingFace ID）
        output_dir: 输出目录（默认在 model_path 后缀 _int4）
        quantize_backbone: 是否量化 backbone（默认只量化 diffusion head）
        device: 设备 ("auto" / "cuda" / "cpu")
        verbose: 打印详细信息

    Returns:
        统计信息字典
    """
    import torch
    from safetensors.torch import load_file, save_file

    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / f"{model_path.name}_int4"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"📦 INT4 量化导出")
        print(f"   输入模型: {model_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   设备: {device}")
        print(f"   量化 backbone: {quantize_backbone}")

    # ─── 方案 A: 通过 BitsAndBytesConfig ───
    try:
        stats = _export_via_bnb(
            model_path=str(model_path),
            output_dir=str(output_dir),
            quantize_backbone=quantize_backbone,
            device=device,
            verbose=verbose,
        )
    except ImportError:
        if verbose:
            print("⚠️  BitsAndBytes 不可用，使用查找表方案")
        stats = _export_via_lut(
            model_path=str(model_path),
            output_dir=str(output_dir),
            verbose=verbose,
        )

    return stats


def _export_via_bnb(
    model_path: str,
    output_dir: str,
    quantize_backbone: bool = False,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """通过 BitsAndBytesConfig 进行 INT4 量化。"""
    from transformers import AutoModelForVision2Seq, BitsAndBytesConfig

    # 配置量化
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

    # 保存量化后的模型
    if verbose:
        print("  💾 保存量化模型...")

    model.save_pretrained(output_dir)

    # 复制 processor_config 等配置文件
    for f in ["processor_config.json", "preprocessor_config.json", "config.json"]:
        src = Path(model_path) / f
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(Path(output_dir) / f))

    # 统计
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
    from src.quantize_safetensors import quantize_to_nf4, BLOCK_SIZE

    model_path = Path(model_path)

    # 找到 safetensors 文件
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"未找到 safetensors 文件: {model_path}")

    if verbose:
        print(f"  📂 找到 {len(safetensors_files)} 个 safetensors 文件")

    exclude_patterns = ["layernorm", "layer_norm", "bias", "embedding",
                        "patch_embed", "pos_embed", "cls_token"]

    total_quantized = 0
    total_skipped = 0

    for sf_path in safetensors_files:
        tensors = {}
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                should_skip = any(p in key.lower() for p in exclude_patterns) or tensor.ndim != 2

                if should_skip:
                    tensors[key] = tensor
                    total_skipped += 1
                else:
                    q, absmax = quantize_to_nf4(tensor)
                    tensors[f"{key}.quant"] = q
                    tensors[f"{key}.absmax"] = absmax
                    tensors[f"{key}.shape"] = np.array(tensor.shape, dtype=np.int32)
                    total_quantized += 1

        out_path = Path(output_dir) / sf_path.name
        save_file(tensors, str(out_path))
        if verbose:
            print(f"  ✅ {sf_path.name}: {total_quantized} quantized, {total_skipped} skipped")

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


# ─────────────────── CLI ───────────────────
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
