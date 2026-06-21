#!/usr/bin/env python3
"""
LoRA 合并工具 — 将 LoRA adapter 合并到基础模型, 保存为完整 FP16 模型

在 AutoDL 云端训练完成后运行, 产生一个可直接加载的全量 FP16 模型,
无需在推理时再加载 LoRA, 适合在 RTX 4090 24GB+ 等高显存 GPU 上运行。

使用方法:
    python merge_lora.py \\
        --base-model /workspace/models/GR00T-N1-1.7-3B \\
        --lora-path /workspace/models/g1_gr00t \\
        --output-dir /workspace/models/g1_gr00t_full_fp16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_dir: str,
) -> str:
    """将 LoRA adapter 合并到基础模型, 保存为完整 FP16 模型。

    Args:
        base_model_path: 基础 GR00T 模型路径 (HuggingFace 格式)
        lora_path: LoRA adapter 目录 (含 adapter_config.json)
        output_dir: 输出目录 (保存合并后的完整模型)

    Returns:
        output_dir: 输出目录路径
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("需要安装: pip install peft transformers accelerate")
        sys.exit(1)

    base_path = Path(base_model_path)
    lora_path_obj = Path(lora_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LoRA 合并 → 完整 FP16 模型")
    logger.info("  基础模型: %s", base_model_path)
    logger.info("  LoRA:     %s", lora_path)
    logger.info("  输出:     %s", output_dir)
    logger.info("=" * 60)

    # ── 1. 加载基础模型 (CPU 上即可, 不消耗 GPU 显存) ─────────────────
    step("加载基础模型 (FP16, CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    logger.info("✅ 基础模型加载成功")

    # ── 2. 加载并合并 LoRA ──────────────────────────────────────────────
    adapter_config = lora_path_obj / "adapter_config.json"
    if adapter_config.exists():
        step("检测到 LoRA adapter, 合并权重...")
        model = PeftModel.from_pretrained(base, str(lora_path_obj))
        model = model.merge_and_unload()
        logger.info("✅ LoRA adapter 合并完成")
    else:
        warn_msg = (
            f"未检测到 LoRA adapter ({adapter_config}), "
            "直接复制基础模型"
        )
        logger.warning(warn_msg)
        model = base

    # ── 3. 保存完整 FP16 模型 ──────────────────────────────────────────
    step("保存完整 FP16 模型...")
    model.save_pretrained(str(output_path), safe_serialization=True)

    # ── 4. 保存 tokenizer ───────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_path), trust_remote_code=True
        )
        tokenizer.save_pretrained(str(output_path))
        logger.info("✅ Tokenizer 保存完成")
    except Exception as e:
        logger.warning("Tokenizer 保存失败: %s (不影响推理)", e)

    # ── 5. 统计大小 ─────────────────────────────────────────────────────
    size_gb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    file_count = sum(1 for _ in output_path.rglob("*") if _.is_file())

    logger.info("=" * 60)
    logger.info("✅ LoRA 合并完成!")
    logger.info("  路径: %s", output_dir)
    logger.info("  大小: %.2f GB (%d files)", size_gb, file_count)
    logger.info("=" * 60)

    return str(output_path)


def step(msg: str):
    logger.info("[→] %s", msg)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 合并 → 完整 FP16 模型 (适合高显存 GPU 推理)"
    )
    parser.add_argument("--base-model", type=str, required=True,
                        help="基础 GR00T 模型路径")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="LoRA adapter 目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="完整 FP16 模型输出目录")
    args = parser.parse_args()

    merge_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
