#!/usr/bin/env python3
"""
INT4 量化导出 — 将 GR00T fine-tune 后的模型量化为 INT4 (NF4 + double-quant)

**纯后训练量化 (Post-Training Quantization)** —— 不需要重新训练,
仅将 FP16 权重转换为 INT4 表示 (NF4 + double-quant), 8GB 显存即可完成。

支持三种输入模式:
  1) LoRA adapter 目录 (含 adapter_config.json)
       → 加载基础模型 + 合并 LoRA → INT4 量化
  2) FP16 全量模型目录 (含 config.json + *.safetensors)
       → 直接 INT4 量化 (无需再加载基础模型)
  3) HuggingFace ID (默认 nvidia/GR00T-N1.7-3B)
       → 远程下载 + INT4 量化 (需要联网)

使用方法:
    # 场景 A: 已有 FP16 全量模型 (本地下载 + 本地量化) ⭐ 推荐
    python export_int4.py \\
        --input-type fp16 \\
        --model-dir /root/models/g1_gr00t_full_fp16 \\
        --output-dir /root/models/g1_gr00t_int4

    # 场景 B: 仅有 LoRA adapter (云端量化场景)
    python export_int4.py \\
        --input-type lora \\
        --model-dir /root/models/g1_gr00t \\
        --output-dir /root/models/g1_gr00t_int4

    # 场景 C: 离线环境 (禁用 HF 在线检查)
    python export_int4.py \\
        --input-type fp16 \\
        --model-dir /root/models/g1_gr00t_full_fp16 \\
        --output-dir /root/models/g1_gr00t_int4 \\
        --offline
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
    input_type: str = "auto",
    offline: bool = False,
    device_map: str = "auto",
) -> str:
    """将 fine-tune 后的模型量化为 INT4。

    Args:
        model_dir: 输入目录
            - LoRA 模式:  含 adapter_config.json 的 adapter 目录
            - FP16 模式:  含 config.json + *.safetensors 的完整模型目录
        output_dir: INT4 量化模型输出目录
        model_path: HuggingFace 模型 ID (仅 LoRA 模式 / 远程模式需要)
        input_type: 输入类型 ("auto" | "lora" | "fp16")
            - "auto": 根据 model_dir 是否含 adapter_config.json 自动判断
            - "lora": 显式指定为 LoRA adapter (需联网下载基础模型)
            - "fp16": 显式指定为完整 FP16 模型 (无需联网)
        offline: 离线模式 (TRANSFORMERS_OFFLINE=1, 禁止 HF 在线下载)
        device_map: 设备映射 ("auto" | "cpu" | "cuda:0")

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

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ── 0. 自动检测输入类型 ─────────────────────────────────────────────
    if input_type == "auto":
        if (model_path_obj / "adapter_config.json").exists():
            input_type = "lora"
        elif (model_path_obj / "config.json").exists():
            input_type = "fp16"
        else:
            logger.error("无法自动判断输入类型: %s", model_dir)
            logger.error("  请用 --input-type 显式指定 (lora | fp16)")
            sys.exit(1)
        logger.info("自动检测输入类型: %s", input_type)

    # ── 离线模式 ──────────────────────────────────────────────────────
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        logger.info("离线模式: 禁止 HuggingFace 在线下载")

    # ── 1. 加载模型 (INT4 量化) ───────────────────────────────────────
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    if input_type == "fp16":
        # ── 场景 A: 本地 FP16 全量模型直接量化 ─────────────────────────
        logger.info("=" * 60)
        logger.info("输入: FP16 全量模型 (本地, 无需联网)")
        logger.info("  路径: %s", model_dir)
        logger.info("=" * 60)

        step("加载 FP16 模型并直接量化为 INT4...")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_path_obj),
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info("✅ FP16 模型加载并量化完成")
        except Exception as e:
            logger.error("FP16 模型加载失败: %s", e)
            sys.exit(1)

        # FP16 全量模型已含合并后的权重, 无需再 merge LoRA
        model = base
        tokenizer_source = str(model_path_obj)

    else:
        # ── 场景 B: LoRA adapter → 合并 + 量化 ─────────────────────────
        logger.info("=" * 60)
        logger.info("输入: LoRA adapter (需联网加载基础模型)")
        logger.info("  Adapter: %s", model_dir)
        logger.info("  Base:   %s", model_path)
        logger.info("=" * 60)

        step("加载基础模型 (INT4 量化)...")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info("✅ 基础模型加载成功")
        except Exception as e:
            logger.error("基础模型加载失败: %s", e)
            logger.info("提示: 如果是 LoRA 模式, 请用 --input-type fp16 切换到本地 FP16 输入")
            logger.info("提示: 或在本地先运行 merge_lora.py 生成完整 FP16 模型")
            sys.exit(1)

        # ── 加载并合并 LoRA ──────────────────────────────────────────
        step("检测到 LoRA adapter, 合并权重...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(model_path_obj))
            model = model.merge_and_unload()
            logger.info("✅ LoRA adapter 合并完成")
        except Exception as e:
            logger.error("LoRA 合并失败: %s", e)
            sys.exit(1)

        tokenizer_source = model_path

    # ── 2. 保存 INT4 模型 ───────────────────────────────────────────────
    step("保存 INT4 量化模型...")
    model.save_pretrained(str(output_path))

    # 保存 tokenizer (优先从本地 FP16 模型读, 失败则从 HF 读)
    try:
        tok = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            local_files_only=offline,
        )
        tok.save_pretrained(str(output_path))
        logger.info("✅ Tokenizer 保存完成 (源: %s)", tokenizer_source)
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
    parser = argparse.ArgumentParser(
        description="GR00T INT4 量化导出 (PTQ, 无需重新训练)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # ⭐ 本地 FP16 → INT4 (推荐, 无需联网)
  python export_int4.py --input-type fp16 \\
      --model-dir models/g1_gr00t_full_fp16 \\
      --output-dir models/g1_gr00t_int4

  # LoRA → 合并 + INT4 (云端量化场景)
  python export_int4.py --input-type lora \\
      --model-dir models/g1_gr00t \\
      --output-dir models/g1_gr00t_int4

  # 离线模式
  python export_int4.py --input-type fp16 --offline \\
      --model-dir models/g1_gr00t_full_fp16 \\
      --output-dir models/g1_gr00t_int4
        """,
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="输入目录 (FP16 全量模型 OR LoRA adapter)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="INT4 模型输出目录")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.7-3B",
                        help="基础模型 HuggingFace ID (仅 --input-type lora 有效)")
    parser.add_argument("--input-type", type=str, default="auto",
                        choices=["auto", "lora", "fp16"],
                        help="输入类型 (默认 auto 自动检测)")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式: 禁止 HF 在线下载 (TRANSFORMERS_OFFLINE=1)")
    parser.add_argument("--device-map", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0"],
                        help="设备映射 (默认 auto)")
    args = parser.parse_args()

    export_int4(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        input_type=args.input_type,
        offline=args.offline,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
