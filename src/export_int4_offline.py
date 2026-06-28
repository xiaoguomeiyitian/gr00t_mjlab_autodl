"""
export_int4_offline.py — INT4 离线量化（跳过 backbone 下载）。

在无网络环境下对本地 safetensors 进行 NF4 量化。
适用于 AutoDL 断网后或离线部署场景。

用法:
    python -m src.export_int4_offline --model-path ./checkpoints/g1_finetune --output-dir ./checkpoints/g1_int4
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from src.quantize_safetensors import quantize_to_nf4, NF4_TABLE, BLOCK_SIZE


def export_int4_offline(
    model_path: str,
    output_dir: Optional[str] = None,
    exclude_backbone: bool = True,
    verbose: bool = True,
) -> dict:
    """
    离线模式 INT4 量化。

    逐文件扫描 safetensors，对 2D 权重做 NF4 查找表量化。
    不需要 torch / transformers / BitsAndBytes。

    Args:
        model_path: BF16 模型目录
        output_dir: 输出目录
        exclude_backbone: 跳过 backbone 层（Cosmos-Reason2）
        verbose: 打印详情
    """
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / f"{model_path.name}_int4_offline"
    else:
        output_dir = Path(output_dir)

    # 先检查输入
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"未找到 safetensors: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # backbone 关键词
    backbone_patterns = [
        "cosmos", "reason", "backbone", "vision_encoder",
        "image_encoder", "patch_embed", "cls_token",
        "pos_embed", "rel_pos", "attn.0", "attn.1",
    ]

    # 通用跳过模式
    skip_patterns = [
        "layernorm", "layer_norm", "bias",
        "embedding", "norm.", "ln_",
    ]

    if verbose:
        print(f"📦 离线 INT4 量化")
        print(f"   输入: {model_path} ({len(safetensors_files)} 文件)")
        print(f"   输出: {output_dir}")
        print(f"   排除 backbone: {exclude_backbone}")

    stats = {
        "total_tensors": 0,
        "quantized": 0,
        "skipped": 0,
        "original_mb": 0,
        "quantized_mb": 0,
    }

    for sf_idx, sf_path in enumerate(safetensors_files):
        tensors = {}
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                stats["total_tensors"] += 1
                stats["original_mb"] += tensor.nbytes / (1024 * 1024)

                should_skip = False

                # 跳过非 2D
                if tensor.ndim != 2:
                    should_skip = True

                # 跳过通用模式
                if not should_skip:
                    for pattern in skip_patterns:
                        if pattern in key.lower():
                            should_skip = True
                            break

                # 跳过 backbone
                if not should_skip and exclude_backbone:
                    for pattern in backbone_patterns:
                        if pattern in key.lower():
                            should_skip = True
                            break

                if should_skip:
                    tensors[key] = tensor
                    stats["skipped"] += 1
                else:
                    q, absmax = quantize_to_nf4(tensor)
                    tensors[f"{key}.quant"] = q
                    tensors[f"{key}.absmax"] = absmax
                    tensors[f"{key}.shape"] = np.array(tensor.shape, dtype=np.int32)
                    stats["quantized"] += 1

        out_path = output_dir / sf_path.name
        save_file(tensors, str(out_path))

        if verbose:
            print(f"  ✅ [{sf_idx+1}/{len(safetensors_files)}] {sf_path.name}")

    # 复制配置文件
    import shutil
    for fname in ["config.json", "processor_config.json", "preprocessor_config.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))

    stats["quantized_mb"] = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(output_dir)
        for fn in fns
    ) / (1024 * 1024)
    stats["compression"] = stats["original_mb"] / max(stats["quantized_mb"], 0.01)

    if verbose:
        print(f"\n📊 离线量化完成:")
        print(f"   总张量: {stats['total_tensors']}")
        print(f"   已量化: {stats['quantized']}")
        print(f"   跳过: {stats['skipped']}")
        print(f"   原始: {stats['original_mb']:.1f} MB")
        print(f"   量化后: {stats['quantized_mb']:.1f} MB")
        print(f"   压缩比: {stats['compression']:.1f}x")

    return stats


def main():
    parser = argparse.ArgumentParser(description="离线 INT4 量化")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-exclude-backbone", action="store_true",
                        help="不排除 backbone（量化全部层）")
    args = parser.parse_args()

    export_int4_offline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        exclude_backbone=not args.no_exclude_backbone,
    )


if __name__ == "__main__":
    main()
