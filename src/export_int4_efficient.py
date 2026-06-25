#!/usr/bin/env python3
"""
INT4 量化导出 (高效模式) — 使用 safetensors 内存映射逐层量化，
避免将整个模型加载到内存。

专门用于:
  - 内存受限环境 (4GB RAM)
  - backbone 是 gated repo 无法下载
  - 只需要量化 action head 权重

原理:
  - 使用 safetensors.safe_open 的内存映射模式逐层读取
  - 对每个 2D 权重做 NF4 量化
  - 使用 PyTorch 的 quantized tensor 格式保存

使用方法:
    python export_int4_efficient.py \
        --model-dir models/g1_gr00t \
        --output-dir models/g1_gr00t_int4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def export_int4_efficient(model_dir: str, output_dir: str) -> str:
    """使用内存映射逐层量化 action head 权重为 INT4."""
    import torch
    from safetensors import safe_open
    from bitsandbytes.functional import quantize_4bit

    model_path_obj = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INT4 量化导出 (高效模式, 内存映射)")
    logger.info("  输入: %s", model_dir)
    logger.info("  输出: %s", output_dir)
    logger.info("=" * 60)

    # ── 1. 读取模型索引 ──────────────────────────────────────────────
    index_path = model_path_obj / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})

    # ── 2. 筛选 action head 的 key ───────────────────────────────────
    # action head 的 key 不包含 backbone 相关的路径
    backbone_keywords = ['backbone', 'qwen', 'model.model.backbone', 'model.model.model']
    action_head_keys = [
        k for k in weight_map
        if not any(kw in k for kw in backbone_keywords)
    ]
    logger.info("总权重数: %d", len(weight_map))
    logger.info("Action head 权重数: %d", len(action_head_keys))

    # 按文件分组
    file_to_keys = {}
    for key in action_head_keys:
        sf_file = weight_map[key]
        file_to_keys.setdefault(sf_file, []).append(key)

    # ── 3. 保存配置文件 ──────────────────────────────────────────────
    # config.json
    config_src = model_path_obj / "config.json"
    if config_src.exists():
        with open(config_src) as f:
            config = json.load(f)
        config["quantization_config"] = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("✅ config.json 已保存")

    # 其他配置文件
    for fname in ["processor_config.json", "statistics.json", "embodiment_id.json"]:
        src = model_path_obj / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            logger.info("✅ %s 已复制", fname)

    # ── 4. 创建新的 index.json ───────────────────────────────────────
    new_weight_map = {}
    for key in action_head_keys:
        # 保持相对路径不变
        new_weight_map[key] = weight_map[key]

    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    logger.info("✅ model.safetensors.index.json 已保存")

    # ── 5. 逐文件量化 ───────────────────────────────────────────────
    total_params = 0
    quantized_params = 0
    start_time = time.time()

    for sf_idx, (sf_name, keys) in enumerate(sorted(file_to_keys.items())):
        sf_path = model_path_obj / sf_name
        out_sf_path = output_path / sf_name

        logger.info("[%d/%d] 处理: %s (%d keys)",
                     sf_idx + 1, len(file_to_keys), sf_name, len(keys))

        # 使用 safe_open 的内存映射模式
        quantized_tensors = {}

        with safe_open(sf_path, framework="pt", device="cpu") as f:
            all_keys_in_file = list(f.keys())
            logger.info("  文件总张量: %d", len(all_keys_in_file))

            for key in keys:
                if key not in all_keys_in_file:
                    continue

                tensor = f.get_tensor(key)
                total_params += tensor.numel()

                # 只量化 2D 权重 (Linear 层)
                if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1:
                    # 转为 float16 再量化
                    t_fp16 = tensor.half().float()

                    # NF4 量化
                    quantized, quant_state = quantize_4bit(
                        t_fp16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                    quantized_tensors[key] = quantized
                    quantized_params += tensor.numel()
                else:
                    # 非 2D 张量直接保存
                    quantized_tensors[key] = tensor

        # 保存量化后的张量
        import safetensors.torch as st_torch
        st_torch.save_file(quantized_tensors, str(out_sf_path))
        logger.info("  → 已保存: %s", out_sf_path)

        # 释放内存
        del quantized_tensors

    elapsed = time.time() - start_time

    # ── 6. 统计 ──────────────────────────────────────────────────────
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
        description="GR00T INT4 量化导出 (高效模式, 内存映射)",
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    export_int4_efficient(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
