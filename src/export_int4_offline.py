"""export_int4_offline.py — INT4 离线量化（跳过 backbone）。"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from src.quantize_safetensors import quantize_to_nf4


def export_int4_offline(
    model_path: str,
    output_dir: Optional[str] = None,
    exclude_backbone: bool = True,
    verbose: bool = True,
) -> dict:
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / f"{model_path.name}_int4_offline"
    else:
        output_dir = Path(output_dir)

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"未找到 safetensors: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_patterns = [
        "cosmos", "reason", "backbone", "vision_encoder",
        "image_encoder", "patch_embed", "cls_token",
        "pos_embed", "rel_pos",
        # 用精确前缀避免误伤扩散头：backbone.attn.0 / backbone.attn.1
        "backbone.attn.0", "backbone.attn.1",
    ]

    skip_patterns = [
        "layernorm", "layer_norm", "bias",
        "embedding", "norm", "ln_",
    ]

    if verbose:
        print(f"📦 离线 INT4 量化")
        print(f"   输入: {model_path} ({len(safetensors_files)} 文件)")
        print(f"   输出: {output_dir}")
        print(f"   排除 backbone: {exclude_backbone}")

    stats = {"total_tensors": 0, "quantized": 0, "skipped": 0, "original_mb": 0, "quantized_mb": 0}

    for sf_idx, sf_path in enumerate(safetensors_files):
        tensors = {}
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                stats["total_tensors"] += 1
                stats["original_mb"] += tensor.nbytes / (1024 * 1024)

                should_skip = False

                if tensor.ndim != 2:
                    should_skip = True

                if not should_skip:
                    for pattern in skip_patterns:
                        if pattern in key.lower():
                            should_skip = True
                            break

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
