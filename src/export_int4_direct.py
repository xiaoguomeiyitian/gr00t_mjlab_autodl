#!/usr/bin/env python3
"""
INT4 量化导出 (直接模式) — 不通过 AutoModel.from_pretrained，
直接读取 safetensors 文件并量化 Linear 层的权重。

用于以下场景:
  - backbone 是 gated repo 无法下载
  - 只需要量化 action head 权重
  - 当前机器没有 GPU (CPU 量化，会很慢)

原理:
  - 读取 safetensors 文件中的 bf16 权重
  - 对每个 Linear 层的 weight 做 NF4 量化 (使用 bitsandbytes)
  - 保存为 safetensors 格式 + config.json

使用方法:
    python export_int4_direct.py \
        --model-dir models/g1_gr00t \
        --output-dir models/g1_gr00t_int4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quantize_linear_weight(weight, bnb_quant_func):
    """量化单个 Linear 层的 weight 为 INT4.

    Args:
        weight: numpy array, shape (out_features, in_features), dtype float32
        bnb_quant_func: bitsandbytes 量化函数

    Returns:
        quantized_weight: numpy array, int8 存储 (4-bit 打包)
        scale: float32 缩放因子
        zero_point: float32 零点
    """
    import torch
    import numpy as np

    # 转为 float16 tensor (模拟 BF16 精度)
    t = torch.from_numpy(weight).half()  # (out, in)

    # bitsandbytes 的 int4 量化
    # 使用 functional.quantize_4bit
    from bitsandbytes.functional import quantize_4bit

    # quantize_4bit 期望输入 shape: (N, )
    # 我们需要 reshape 为 2D
    quantized, quant_state = quantize_4bit(
        t,
        compress_statistics=True,
        quant_type="nf4",
    )

    return quantized, quant_state


def export_int4_direct(model_dir: str, output_dir: str) -> str:
    """直接读取 safetensors 并量化 Linear 层."""
    import numpy as np
    import torch

    model_path_obj = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INT4 量化导出 (直接模式)")
    logger.info("  输入: %s", model_dir)
    logger.info("  输出: %s", output_dir)
    logger.info("=" * 60)

    # ── 1. 读取模型索引 ──────────────────────────────────────────────
    index_path = model_path_obj / "model.safetensors.index.json"
    if not index_path.exists():
        logger.error("找不到 model.safetensors.index.json: %s", index_path)
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    logger.info("模型包含 %d 个权重文件", len(set(weight_map.values())))

    # ── 2. 读取 config.json ───────────────────────────────────────────
    config_path = model_path_obj / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # 修改 config 以标记为 INT4
        config["quantization_config"] = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
        }
        # 保存修改后的 config
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("✅ config.json 已保存 (含 quantization_config)")

    # ── 3. 保存其他配置文件 ─────────────────────────────────────────
    for fname in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
        src = model_path_obj / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / fname)
            logger.info("✅ %s 已复制", fname)

    # ── 4. 量化权重 ──────────────────────────────────────────────────
    from bitsandbytes.functional import quantize_4bit
    from safetensors import safe_open
    import safetensors.torch as st_torch

    # 收集所有 safetensors 文件
    safetensor_files = sorted(set(weight_map.values()))
    logger.info("需要处理 %d 个 safetensors 文件", len(safetensor_files))
    logger.debug("文件列表: %s", safetensor_files)

    total_params = 0
    quantized_params = 0
    start_time = time.time()

    # 用于存储量化后的权重
    all_quantized_weights = {}

    for sf_idx, sf_name in enumerate(safetensor_files):
        sf_path = model_path_obj / sf_name
        logger.info("[%d/%d] 处理: %s", sf_idx + 1, len(safetensor_files), sf_name)

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            logger.info("  → %d 个张量", len(keys))

            for key in keys:
                tensor = f.get_tensor(key)
                total_params += tensor.numel()

                # 只量化 2D 权重 (Linear 层)
                if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
                    # 转为 float16 再量化
                    t_fp16 = tensor.half().float()  # 保持精度

                    # NF4 量化
                    quantized, quant_state = quantize_4bit(
                        t_fp16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                    all_quantized_weights[key] = quantized
                    quantized_params += tensor.numel()
                    logger.debug("  ✓ %s: %s → INT4", key, tuple(tensor.shape))
                else:
                    # 非 2D 张量直接保存
                    all_quantized_weights[key] = tensor
                    logger.debug("  - %s: %s (跳过)", key, tuple(tensor.shape))

        # 每处理完一个文件，保存一次（避免内存溢出）
        if len(all_quantized_weights) > 0:
            output_sf = output_path / sf_name
            # 注意: bitsandbytes 的量化张量不能直接保存为 safetensors
            # 我们需要用 PyTorch 的 save 格式
            logger.info("  → 保存量化权重: %s", output_sf)
            # 使用 torch.save 而不是 safetensors.save_file
            # 因为量化张量是特殊的格式
            st_torch.save_file(all_quantized_weights, str(output_sf))
            all_quantized_weights.clear()

    elapsed = time.time() - start_time

    # ── 5. 统计 ──────────────────────────────────────────────────────
    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 ** 2)
    file_count = sum(1 for _ in output_path.rglob("*") if _.is_file())

    logger.info("=" * 60)
    logger.info("✅ INT4 量化导出完成!")
    logger.info("  路径: %s", output_path)
    logger.info("  大小: %.0f MB (%d files)", size_mb, file_count)
    logger.info("  总参数: %d", total_params)
    logger.info("  量化参数: %d (%.1f%%)", quantized_params, 100 * quantized_params / max(total_params, 1))
    logger.info("  耗时: %.1f 秒", elapsed)
    logger.info("=" * 60)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="GR00T INT4 量化导出 (直接模式, 不依赖 backbone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="输入目录 (BF16 完整模型, 含 config.json + *.safetensors)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="INT4 模型输出目录")
    args = parser.parse_args()

    export_int4_direct(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
