#!/usr/bin/env python3
"""
直接量化 safetensors 文件为 INT4 (NF4)。

不依赖 transformers 模型加载，直接读取 safetensors 文件，
对每个权重矩阵进行 NF4 量化，保存为 safetensors 格式。

用法:
  python quantize_safetensors.py \
      --input-dir /path/to/GR00T-N1.7-3B \
      --output-dir /path/to/GR00T-N1.7-3B-INT4 \
      --quantize-backbone  # 是否也量化 backbone (默认只量化 action head)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quantize_nf4(weight: np.ndarray, block_size: int = 64) -> tuple:
    """对单个权重矩阵进行 NF4 量化。

    Args:
        weight: 输入权重 (float32 numpy array)
        block_size: NF4 block 大小 (默认 64)

    Returns:
        (quantized_data, scale, zero_point)
        - quantized_data: uint8 array, 每 2 个值打包成 1 个 byte (4-bit)
        - scale: float32 array, 每个 block 一个 scale
        - zero_point: float32 array, 每个 block 一个 zero_point
    """
    # NF4 量化表 (NormalFloat4)
    NF4_TABLE = np.array([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09185808897018433, 0.0,
        0.07555019855499268, 0.15110039710998535, 0.22369961440563202, 0.293800413608551,
        0.3639012277126312, 0.4355524778366089, 0.5124996900558472, 0.6000000238418579
    ], dtype=np.float32)

    original_shape = weight.shape
    original_dtype = weight.dtype

    # 转为 float32
    if original_dtype != np.float32:
        weight = weight.astype(np.float32)

    # 展平
    flat = weight.flatten()

    # 补齐到 block_size 的倍数
    n = len(flat)
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

    n_blocks = len(flat) // block_size
    flat_blocks = flat.reshape(n_blocks, block_size)

    # 对每个 block 计算 absmax，然后归一ize到 [-1, 1]
    absmax = np.abs(flat_blocks).max(axis=1, keepdims=True)
    absmax = np.maximum(absmax, 1e-8)  # 避免除零

    # 归一化到 [-1, 1]
    normalized = flat_blocks / absmax

    # 找到最近的 NF4 值
    # 计算每个值到 NF4_TABLE 中所有值的距离
    # normalized shape: (n_blocks, block_size)
    # NF4_TABLE shape: (16,)
    # 广播: (n_blocks, block_size, 1) - (1, 1, 16) -> (n_blocks, block_size, 16)
    diff = normalized[:, :, np.newaxis] - NF4_TABLE[np.newaxis, np.newaxis, :]
    indices = np.argmin(np.abs(diff), axis=2)  # (n_blocks, block_size)

    # 打包: 每 2 个 4-bit 值打包成 1 个 uint8
    # 偶数索引在高 4 位，奇数索引在低 4 位
    indices_flat = indices.reshape(-1)
    packed = np.zeros(len(indices_flat) // 2, dtype=np.uint8)
    packed = ((indices_flat[0::2] & 0x0F) << 4) | (indices_flat[1::2] & 0x0F)

    # scale 和 zero_point
    scale = absmax.flatten() / 1.0  # NF4 范围是 [-1, 1]，所以 scale = absmax
    # zero_point 对于对称量化是 0
    zero_point = np.zeros(n_blocks, dtype=np.float32)

    return packed, scale, zero_point, original_shape, original_dtype, pad_len


def quantize_file(input_path: str, output_path: str, quantize_backbone: bool = False):
    """量化单个 safetensors 文件。"""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        logger.error("需要安装: pip install safetensors")
        sys.exit(1)

    import torch

    logger.info(f"处理: {input_path}")

    tensors_to_save = {}
    metadata = {}

    with safe_open(input_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        logger.info(f"  共 {len(keys)} 个 tensor")

        backbone_count = 0
        action_count = 0
        skipped_count = 0

        for key in keys:
            tensor = f.get_tensor(key)

            # 判断是否量化
            is_backbone = key.startswith("backbone.")
            if is_backbone and not quantize_backbone:
                # 不量化 backbone，直接复制
                tensors_to_save[key] = tensor
                skipped_count += 1
                continue

            # 只量化 2D 权重矩阵 (忽略 bias, layernorm 等)
            if tensor.ndim != 2:
                tensors_to_save[key] = tensor
                skipped_count += 1
                continue

            # 转为 float32 numpy (bf16 不直接支持 numpy)
            if tensor.dtype == torch.bfloat16:
                np_tensor = tensor.float().numpy()
            else:
                np_tensor = tensor.numpy()

            # 量化
            packed, scale, zero_point, orig_shape, orig_dtype, pad_len = quantize_nf4(np_tensor)

            # 保存量化后的数据
            tensors_to_save[key + ".qdata"] = torch.from_numpy(packed)
            tensors_to_save[key + ".scale"] = torch.from_numpy(scale)
            tensors_to_save[key + ".zero_point"] = torch.from_numpy(zero_point)
            metadata[key + ".orig_shape"] = json.dumps(list(orig_shape))
            metadata[key + ".orig_dtype"] = str(orig_dtype)
            metadata[key + ".pad_len"] = str(pad_len)

            if is_backbone:
                backbone_count += 1
            else:
                action_count += 1

        logger.info(f"  量化: backbone={backbone_count}, action_head={action_count}, 跳过={skipped_count}")

    # 保存
    save_file(tensors_to_save, output_path, metadata=metadata)
    logger.info(f"  保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="直接量化 safetensors 文件为 INT4 (NF4)",
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入目录 (含 *.safetensors)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--quantize-backbone", action="store_true",
                        help="是否也量化 backbone (默认只量化 action head)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有 safetensors 文件
    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        logger.error(f"未找到 safetensors 文件: {input_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(safetensor_files)} 个 safetensors 文件")

    # 复制非 safetensors 文件到输出目录
    for f in input_dir.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            import shutil
            shutil.copy2(f, output_dir / f.name)
            logger.info(f"  复制: {f.name}")

    # 复制 model.safetensors.index.json 并修改
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        import shutil
        shutil.copy2(index_path, output_dir / "model.safetensors.index.json")

    # 量化每个 safetensors 文件
    for sf in safetensor_files:
        output_path = output_dir / sf.name
        quantize_file(str(sf), str(output_path), args.quantize_backbone)

    # 统计大小
    input_size = sum(f.stat().st_size for f in input_dir.rglob("*") if f.is_file())
    output_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info("=" * 60)
    logger.info(f"✅ 量化完成!")
    logger.info(f"  输入: {input_dir} ({input_size/1024/1024/1024:.2f} GB)")
    logger.info(f"  输出: {output_dir} ({output_size/1024/1024/1024:.2f} GB)")
    logger.info(f"  压缩比: {input_size/output_size:.1f}x")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
