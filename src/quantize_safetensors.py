"""
quantize_safetensors.py — 直接 safetensors NF4 量化核心。

读取 safetensors 文件，对权重进行 NF4 查找表量化。
适用于内存受限环境，无需加载完整模型。
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


NF4_TABLE = np.array([
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0,
], dtype=np.float32)

BLOCK_SIZE = 64


def quantize_to_nf4(weight: np.ndarray) -> tuple:
    original_shape = weight.shape
    original_dtype = weight.dtype

    m, n = original_shape
    n_blocks_per_row = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_padded = n_blocks_per_row * BLOCK_SIZE
    pad_len = n_padded - n

    if pad_len > 0:
        padded = np.pad(weight.astype(np.float32), ((0, 0), (0, pad_len)))
    else:
        padded = weight.astype(np.float32)

    blocks = padded.reshape(m, n_blocks_per_row, BLOCK_SIZE)

    absmax = np.max(np.abs(blocks), axis=2)
    absmax = np.maximum(absmax, 1e-8)

    normalized = blocks / absmax[:, :, None]

    # 向量化：一次性广播计算所有块到 NF4 表的最近邻索引
    # normalized: (m, n_blocks, 64), NF4_TABLE: (16,)
    # dists: (m, n_blocks, 64, 16)
    dists = np.abs(normalized[..., None] - NF4_TABLE[None, None, None, :])
    indices = np.argmin(dists, axis=-1).astype(np.int32)  # (m, n_blocks, 64)
    indices = indices.reshape(m, n_padded)  # (m, n_padded)

    indices = indices[:, :n]

    if n % 2 != 0:
        indices = np.pad(indices, ((0, 0), (0, 1)), constant_values=0)

    packed = indices[:, 0::2].astype(np.uint8) | (indices[:, 1::2].astype(np.uint8) << 4)

    return packed, absmax


def dequantize_from_nf4(packed: np.ndarray, absmax: np.ndarray, original_shape: tuple) -> np.ndarray:
    """从 NF4 量化结果反量化为 float32 权重。

    Args:
        packed: (m, ceil(n/2)) uint8，每字节含 2 个 4-bit 索引
        absmax: (m, n_blocks_per_row) float32，每块的绝对最大值
        original_shape: 原始权重形状 (m, n)

    Returns:
        weight: (m, n) float32
    """
    m, n = original_shape
    n_blocks_per_row = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_padded = n_blocks_per_row * BLOCK_SIZE

    # 解包 4-bit 索引
    low = (packed & 0x0F).astype(np.int32)
    high = (packed >> 4).astype(np.int32)
    indices = np.zeros((m, packed.shape[1] * 2), dtype=np.int32)
    indices[:, 0::2] = low
    indices[:, 1::2] = high
    # 解包后长度可能 < n_padded（奇数列时），需 pad 到 n_padded
    if indices.shape[1] < n_padded:
        indices = np.pad(indices, ((0, 0), (0, n_padded - indices.shape[1])), constant_values=0)
    indices = indices[:, :n_padded]

    # 查表
    blocks = NF4_TABLE[indices]  # (m, n_padded)
    blocks = blocks.reshape(m, n_blocks_per_row, BLOCK_SIZE)

    # 反归一化
    dequant_blocks = blocks * absmax[:, :, None]
    dequant = dequant_blocks.reshape(m, n_padded)
    dequant = dequant[:, :n]

    return dequant.astype(np.float32)


def quantize_safetensors_file(
    input_path: str,
    output_path: Optional[str] = None,
    exclude_patterns: Optional[list] = None,
    verbose: bool = True,
) -> dict:
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

            should_skip = False
            for pattern in exclude_patterns:
                if pattern.lower() in key.lower():
                    should_skip = True
                    break

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
                ratio = tensor.nbytes / (q.nbytes + absmax.nbytes + np.array(tensor.shape, dtype=np.int32).nbytes)
                stats["details"].append({
                    "key": key,
                    "shape": list(tensor.shape),
                    "ratio": f"{ratio:.1f}x",
                })
                if verbose:
                    print(f"   ✅ 量化: {key} (shape={tensor.shape}, ratio={ratio:.1f}x)")

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
