#!/usr/bin/env python3
"""
INT4 量化导出 — 将 GR00T fine-tune 后的模型量化为 INT4

在 AutoDL 云端训练完成后运行，将全精度模型转为 INT4 量化版本 (~1.5GB)，
方便在 RTX 2080 8GB 等低显存 GPU 上推理。

使用方法:
    python export_int4.py --model-dir /workspace/models/g1_gr00t \
        --output-dir /workspace/models/g1_gr00t_int4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def export_int4(
    model_dir: str,
    output_dir: str,
    model_path: str = "nvidia/GR00T-N1.7-3B",
) -> str:
    """将 fine-tune 后的模型量化为 INT4。

    Args:
        model_dir: fine-tune 输出目录 (含 adapter 或完整权重)
        output_dir: INT4 量化模型输出目录
        model_path: 基础模型 HuggingFace ID (用于加载 tokenizer)

    Returns:
        output_dir: 输出目录路径
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error("需要安装: pip install bitsandbytes accelerate peft transformers")
        sys.exit(1)

    model_path_obj = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INT4 量化导出")
    logger.info("  输入: %s", model_dir)
    logger.info("  输出: %s", output_dir)
    logger.info("  基础模型: %s", model_path)
    logger.info("=" * 60)

    # ── 1. 加载基础模型 (INT4) ──────────────────────────────────────────
    step("加载基础模型 (INT4 量化)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("✅ 基础模型加载成功")
    except Exception as e:
        logger.error("基础模型加载失败: %s", e)
        logger.info("尝试使用 transformers 兼容模式...")
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
        )

    # ── 2. 加载 LoRA adapter (如果有) ───────────────────────────────────
    model = base
    adapter_config = model_path_obj / "adapter_config.json"
    if adapter_config.exists():
        step("检测到 LoRA adapter, 合并权重...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(model_path_obj))
        model = model.merge_and_unload()
        logger.info("✅ LoRA adapter 合并完成")
    else:
        logger.info("未检测到 LoRA adapter, 使用全量权重")

    # ── 3. 保存 INT4 模型 ───────────────────────────────────────────────
    step("保存 INT4 量化模型...")
    model.save_pretrained(str(output_path))

    # 保存 tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.save_pretrained(str(output_path))
        logger.info("✅ Tokenizer 保存完成")
    except Exception as e:
        logger.warning("Tokenizer 保存失败: %s (不影响推理)", e)

    # ── 4. 统计大小 ─────────────────────────────────────────────────────
    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 ** 2)
    file_count = sum(1 for _ in output_path.rglob("*") if _.is_file())

    logger.info("=" * 60)
    logger.info("✅ INT4 量化导出完成!")
    logger.info("  路径: %s", output_dir)
    logger.info("  大小: %.0f MB (%d files)", size_mb, file_count)
    logger.info("=" * 60)

    return str(output_path)


def step(msg: str):
    logger.info("[→] %s", msg)


def main():
    parser = argparse.ArgumentParser(description="GR00T INT4 量化导出")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="fine-tune 输出目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="INT4 模型输出目录")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.7-3B",
                        help="基础模型 HuggingFace ID")
    args = parser.parse_args()

    export_int4(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
