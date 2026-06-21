#!/usr/bin/env bash
# ============================================================================
# 容器内: Isaac-GR00T 训练入口
# ============================================================================
set -euo pipefail

cd /root/Isaac-GR00T

# 加载 GR00T 的 Python 3.10 虚拟环境
source /opt/gr00t-venv/bin/activate

DATA_DIR="${DATA_DIR:-/root/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/models/${ROBOT:-g1}_gr00t}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/root/models/GR00T-N1-${MODEL_SIZE:-1.7-3B}}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ROBOT="${ROBOT:-g1}"
MODEL_SIZE="${MODEL_SIZE:-1.7-3B}"

# 默认数据路径 (与 collect 输出一致)
LEROBOT_DATA_DIR="${DATA_DIR}/${ROBOT}_lerobot"

echo "=========================================="
echo " [2] GR00T 训练 (Isaac-GR00T)"
echo "=========================================="
echo "机器人:     ${ROBOT}"
echo "数据:       ${LEROBOT_DATA_DIR}"
echo "输出:       ${OUTPUT_DIR}"
echo "Epochs:     ${NUM_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE} x ${GRAD_ACCUM}"
echo "LR:         ${LEARNING_RATE}"
echo ""

# ── 检查数据 ────────────────────────────────────────────────────────────
if [ ! -d "${LEROBOT_DATA_DIR}" ]; then
    echo "ERROR: 数据目录不存在: ${LEROBOT_DATA_DIR}"
    echo "请先上传训练包并解压, 或在挂载的 data/ 目录中准备数据"
    exit 1
fi

# ── 检查基础模型 ────────────────────────────────────────────────────────
if [ ! -d "${BASE_MODEL_DIR}" ]; then
    echo "下载基础模型 ${BASE_MODEL_DIR}..."
    mkdir -p "${BASE_MODEL_DIR}"
    hf download "nvidia/GR00T-N1.${MODEL_SIZE}" --local-dir "${BASE_MODEL_DIR}"
fi

# ── 启动 fine-tune (使用 LoRA) ──────────────────────────────────────────
python3 gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL_DIR}" \
    --dataset-path "${LEROBOT_DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --learning-rate "${LEARNING_RATE}" \
    --embodiment-tag "NEW_EMBODIMENT" \
    --use-lora

echo ""
echo "✅ 训练完成: ${OUTPUT_DIR}"
