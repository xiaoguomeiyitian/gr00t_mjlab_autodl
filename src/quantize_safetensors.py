"""
quantize_safetensors.py — 直接 safetensors NF4 量化核心。

读取 safetensors 文件，对权重进行 NF4 查找表量化。
适用于内存受限环境，无需加载完整模型。

用法:
    python -m src.quantize_safetensors --input model.safetensors --output model_int4.safetensors
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np


# NF4 查找表（16 个量化值，QLoRA 原始值）
NF4_TABLE = np.array([
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0,
], dtype=np.float32)

# NF4 量化块大小
BLOCK_SIZE = 64


def quantize_to_nf4(weight: np.ndarray) -> tuple:
    """
    将权重矩阵量化为 NF4 格式。

    Args:
        weight: 原始权重 (m, n) float16/bfloat16/float32

    Returns:
        quantized: 量化后的 uint8 数组 (m, n//2)，每字节存 2 个 NF4 值
        absmax: 每个块的最大绝对值 (m, n//BLOCK_SIZE)
    """
    original_shape = weight.shape
    original_dtype = weight.dtype

    # 逐行量化，避免 padding 对齐问题
    m, n = original_shape
    n_blocks_per_row = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_padded = n_blocks_per_row * BLOCK_SIZE
    pad_len = n_padded - n

    # Pad 每行
    if pad_len > 0:
        padded = np.pad(weight.astype(np.float32), ((0, 0), (0, pad_len)))
    else:
        padded = weight.astype(np.float32)

    # Reshape 为 (m, n_blocks_per_row, BLOCK_SIZE)
    blocks = padded.reshape(m, n_blocks_per_row, BLOCK_SIZE)

    # 计算每个块的 absmax
    absmax = np.max(np.abs(blocks), axis=2)  # (m, n_blocks_per_row)
    absmax = np.maximum(absmax, 1e-8)  # 避免除零

    # 归一化到 [-1, 1]
    normalized = blocks / absmax[:, :, None]

    # 找到最近的 NF4 值
    indices = np.zeros((m, n_padded), dtype=np.int32)
    for row in range(m):
        for col in range(n_blocks_per_row):
            block = normalized[row, col]
            dists = np.abs(block[:, None] - NF4_TABLE[None, :])
            start = col * BLOCK_SIZE
            indices[row, start:start + BLOCK_SIZE] = np.argmin(dists, axis=1)

    # 裁剪到原始列数
    indices = indices[:, :n]

    # 确保列数为偶数（最后一个值填 0）
    if n % 2 != 0:
        indices = np.pad(indices, ((0, 0), (0, 1)), constant_values=0)

    # 每字节存 2 个 NF4 值（低 4 位 + 高 4 位）
    packed = indices[:, 0::2].astype(np.uint8) | (indices[:, 1::2].astype(np.uint8) << 4)

    # absmax 只保留需要的列
    absmax = absmax[:, :n_blocks_per_row]

    return packed, absmax


def quantize_safetensors_file(
    input_path: str,
    output_path: Optional[str] = None,
    exclude_patterns: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """
    量化 safetensors 文件中的所有权重。

    Args:
        input_path: 输入 safetensors 文件路径
        output_path: 输出文件路径（默认在原文件名加 _int4 后缀）
        exclude_patterns: 要跳过的 key 模式列表
        verbose: 是否打印详细信息

    Returns:
        统计信息字典
    """
    try:
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError:
        raise ImportError("需要安装 safetensors: pip install safetensors")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_int4.safetensors"
    else:
        output_path = Path(output_path)

    exclude_patterns = exclude_patterns or [
        "layernorm", "layer_norm", "bias", "embedding",
        "patch_embed", "pos_embed", "cls_token",
    ]

    if verbose:
        print(f"📦 NF4 量化: {input_path}")
        print(f"   输出: {output_path}")

    stats = {
        "total_tensors": 0,
        "quantized_tensors": 0,
        "skipped_tensors": 0,
        "original_size_mb": 0,
        "quantized_size_mb": 0,
        "details": [],
    }

    quantized_tensors = {}
    input_size = input_path.stat().st_size
    stats["original_size_mb"] = input_size / (1024 * 1024)

    with safe_open(str(input_path), framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            stats["total_tensors"] += 1

            # 判断是否需要量化
            should_skip = False
            for pattern in exclude_patterns:
                if pattern.lower() in key.lower():
                    should_skip = True
                    break

            # 只量化 2D 权重
            if tensor.ndim != 2:
                should_skip = True

            if should_skip:
                quantized_tensors[key] = tensor
                stats["skipped_tensors"] += 1
                if verbose:
                    print(f"   ⏭️  跳过: {key} (shape={tensor.shape})")
            else:
                q, absmax = quantize_to_nf4(tensor)
                quantized_tensors[f"{key}.quant"] = q
                quantized_tensors[f"{key}.absmax"] = absmax
                quantized_tensors[f"{key}.shape"] = np.array(tensor.shape, dtype=np.int32)
                stats["quantized_tensors"] += 1
                ratio = tensor.nbytes / (q.nbytes + absmax.nbytes + 12)
                stats["details"].append({
                    "key": key,
                    "shape": list(tensor.shape),
                    "ratio": f"{ratio:.1f}x",
                })
                if verbose:
                    print(f"   ✅ 量化: {key} (shape={tensor.shape}, ratio={ratio:.1f}x)")

    # 保存
    save_file(quantized_tensors, str(output_path))
    stats["quantized_size_mb"] = output_path.stat().st_size / (1024 * 1024)
    stats["compression_ratio"] = stats["original_size_mb"] / max(stats["quantized_size_mb"], 0.01)

    if verbose:
        print(f"\n📊 量化完成:")
        print(f"   总张量: {stats['total_tensors']}")
        print(f"   已量化: {stats['quantized_tensors']}")
        print(f"   跳过: {stats['skipped_tensors']}")
        print(f"   原始: {stats['original_size_mb']:.1f} MB")
        print(f"   量化后: {stats['quantized_size_mb']:.1f} MB")
        print(f"   压缩比: {stats['compression_ratio']:.1f}x")

    return stats


# ─────────────────── CLI ───────────────────
def main():
    parser = argparse.ArgumentParser(description="safetensors NF4 量化")
    parser.add_argument("--input", type=str, required=True, help="输入 safetensors 文件")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--exclude", type=str, nargs="*", default=None,
                        help="要跳过的 key 模式")
    args = parser.parse_args()

    quantize_safetensors_file(
        input_path=args.input,
        output_path=args.output,
        exclude_patterns=args.exclude,
    )


if __name__ == "__main__":
    main()
