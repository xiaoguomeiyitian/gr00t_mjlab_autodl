#!/usr/bin/env python3
"""
INT4 量化导出 — 将 GR00T fine-tune 后的 BF16 完整模型量化为 INT4 (NF4 + double-quant)

纯后训练量化 (PTQ)，不需要重新训练，8GB 显存即可完成。




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
    """将 fine-tune 后的 BF16 完整模型量化为 INT4。

    Args:
        model_dir: BF16 完整模型目录 (含 config.json + *.safetensors)
        output_dir: INT4 量化模型输出目录
        model_path: HuggingFace 模型 ID (仅远程模式需要)
        input_type: 输入类型 ("auto" | "fp16")
            - "auto": 根据 model_dir 是否含 config.json 自动判断
            - "fp16": 显式指定为 BF16 完整模型 (无需联网)
        offline: 离线模式 (TRANSFORMERS_OFFLINE=1, 禁止 HF 在线下载)
        device_map: 设备映射 ("auto" | "cpu" | "cuda:0")

    Returns:
        output_dir: 输出目录路径
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error("需要安装: pip install bitsandbytes accelerate transformers")
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

    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

    # ── 0. 自动检测输入类型 ─────────────────────────────────────────────
    if input_type == "auto":
        if (model_path_obj / "config.json").exists():
            input_type = "fp16"
        else:
            logger.error("无法自动判断输入类型: %s", model_dir)
            logger.error("  请用 --input-type fp16 显式指定")
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
        # ── 场景 A: 本地 BF16 完整模型直接量化 ─────────────────────────
        logger.info("=" * 60)
        logger.info("输入: BF16 完整模型 (本地, 无需联网)")
        logger.info("  路径: %s", model_dir)
        logger.info("=" * 60)

        step("加载 BF16 模型并直接量化为 INT4...")
        try:
            base = AutoModel.from_pretrained(
                str(model_path_obj),
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info("✅ BF16 模型加载并量化完成")
        except Exception as e:
            logger.error("BF16 模型加载失败: %s", e)
            sys.exit(1)

            model = base
        processor_source = str(model_path_obj)

    else:
        logger.error("不支持的 input_type: %s", input_type)
        sys.exit(1)
    # ── 2. 保存 INT4 模型 ───────────────────────────────────────────────
    step("保存 INT4 量化模型...")
    model.save_pretrained(str(output_path))

    # 保存 processor
    try:
        proc = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            local_files_only=offline,
        )
        proc.save_pretrained(str(output_path))
        logger.info("✅ Processor 保存完成 (源: %s)", processor_source)
    except Exception as e:
        logger.warning("Processor 保存失败: %s (不影响推理, 但 Gr00tPolicy 加载时需手动指定 processor_dir)", e)

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
  # 本地 BF16 完整模型 → INT4 (无需联网)
  python export_int4.py --input-type fp16 \\
      --model-dir models/g1_gr00t \\
      --output-dir models/g1_gr00t_int4

  # 离线模式
  python export_int4.py --input-type fp16 --offline \\
      --model-dir models/g1_gr00t \\
      --output-dir models/g1_gr00t_int4
        """,
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="输入目录 (BF16 完整模型, 含 config.json + *.safetensors)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="INT4 模型输出目录")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.7-3B",
                        help="基础模型 HuggingFace ID (仅 --input-type lora 有效)")
    parser.add_argument("--input-type", type=str, default="auto",
                        choices=["auto", "fp16"],
                        help="输入类型 (默认 auto 自动检测, 仅支持 fp16 完整模型)")
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
