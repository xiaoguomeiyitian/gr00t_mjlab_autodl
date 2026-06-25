#!/usr/bin/env python3
"""
INT4 量化导出 (离线版) — 跳过 backbone 下载，直接量化 GR00T 模型。

与 export_int4.py 不同，此脚本:
  1. 不下载 VLM backbone (nvidia/Cosmos-Reason2-2B)
  2. 直接加载本地 BF16 权重 (GR00T 的 action head + projector + backbone 权重已在本地)
  3. 量化为 INT4 (NF4 + double-quant)

前提:
  - GR00T 完整模型已下载到本地 (含 config.json + *.safetensors)
  - Isaac-GR00T 仓库在 PYTHONPATH 中 (用于注册 Gr00tN1d7 架构)

用法:
  python export_int4_offline.py \
      --model-dir /path/to/GR00T-N1.7-3B \
      --output-dir /path/to/GR00T-N1.7-3B-int4 \
      --device-map auto
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def export_int4_offline(
    model_dir: str,
    output_dir: str,
    device_map: str = "auto",
) -> str:
    """将本地 BF16 GR00T 模型量化为 INT4，不下载 backbone。"""
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.error("需要安装: pip install bitsandbytes accelerate transformers")
        sys.exit(1)

    # 注册 Isaac-GR00T 自定义模型架构 (Gr00tN1d7)
    try:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
        logger.info("已注册 Isaac-GR00T 模型架构 (Gr00tN1d7)")
    except ImportError as e:
        logger.warning(f"无法导入 Gr00tN1d7: {e}")
        logger.warning("请确保 Isaac-GR00T 在 PYTHONPATH 中")
        sys.exit(1)

    model_path_obj = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 0. 修改 config.json: 将 model_name 改为本地路径 ──────────────
    config_path = model_path_obj / "config.json"
    if not config_path.exists():
        logger.error("config.json 不存在: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        original_config = json.load(f)

    original_model_name = original_config.get("model_name", "")
    logger.info("原始 model_name: %s", original_model_name)

    # 将 model_name 改为本地目录 (让 backbone 从本地加载)
    # GR00T 的 backbone (Cosmos-Reason2-2B) 权重已经包含在 safetensors 中
    # 但 config 中的 model_name 指向 HF repo，会导致尝试下载
    # 改为本地路径后，模型会从 model_dir 加载 backbone
    modified_config = original_config.copy()
    modified_config["model_name"] = str(model_path_obj)

    # 同时确保 output_path 不在 model_path 中 (避免递归)
    # 注意: 需要检查路径分隔符，避免 "model-3B-INT4" 被误判为在 "model-3B" 内部
    model_resolved = model_path_obj.resolve()
    output_resolved = output_path.resolve()
    if output_resolved.is_relative_to(model_resolved):
        logger.error("输出目录不能在输入目录内部!")
        sys.exit(1)

    # 临时写回修改后的 config
    # 注意: 我们在 model_dir 下临时修改，加载完成后恢复
    backup_config = json.dumps(original_config, indent=2)
    with open(config_path, "w") as f:
        json.dump(modified_config, f, indent=2)
    logger.info("已临时修改 config.json 的 model_name 为本地路径")

    try:
        # ── 1. 设置环境变量 ────────────────────────────────────────────
        # 禁止 HuggingFace 在线下载
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        # ── 2. 加载模型 (INT4 量化) ───────────────────────────────────
        from transformers import AutoModel, AutoProcessor

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info("=" * 60)
        logger.info("INT4 量化导出 (离线模式, 跳过 backbone 下载)")
        logger.info("  输入: %s", model_dir)
        logger.info("  输出: %s", output_dir)
        logger.info("=" * 60)

        step("加载 BF16 模型并直接量化为 INT4...")
        try:
            model = AutoModel.from_pretrained(
                str(model_path_obj),
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            logger.info("✅ BF16 模型加载并量化完成")
        except Exception as e:
            logger.error("BF16 模型加载失败: %s", e)
            sys.exit(1)

        # ── 3. 保存 INT4 模型 ───────────────────────────────────────────
        step("保存 INT4 量化模型...")
        model.save_pretrained(str(output_path))

        # 保存 processor
        try:
            proc = AutoProcessor.from_pretrained(
                str(model_path_obj),
                trust_remote_code=True,
                local_files_only=True,
            )
            proc.save_pretrained(str(output_path))
            logger.info("✅ Processor 保存完成")
        except Exception as e:
            logger.warning("Processor 保存失败: %s (不影响推理)", e)

        # ── 4. 统计大小 ─────────────────────────────────────────────────
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

    finally:
        # ── 5. 恢复原始 config.json ─────────────────────────────────────
        with open(config_path, "w") as f:
            f.write(backup_config)
        logger.info("已恢复原始 config.json (model_name: %s)", original_model_name)


def step(msg: str):
    logger.info("[→] %s", msg)


def main():
    parser = argparse.ArgumentParser(
        description="GR00T INT4 量化导出 (离线版, 跳过 backbone 下载)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python export_int4_offline.py \\
      --model-dir /home/kxy/work/unitree/models/GR00T-N1.7-3B \\
      --output-dir /home/kxy/work/unitree/models/GR00T-N1.7-3B-int4

  # CPU 模式 (显存不足时)
  python export_int4_offline.py \\
      --model-dir models/GR00T-N1.7-3B \\
      --output-dir models/GR00T-N1.7-3B-int4 \\
      --device-map cpu
        """,
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="输入目录 (BF16 完整模型, 含 config.json + *.safetensors)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="INT4 模型输出目录")
    parser.add_argument("--device-map", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "cuda:0"],
                        help="设备映射 (默认 auto)")
    args = parser.parse_args()

    export_int4_offline(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
