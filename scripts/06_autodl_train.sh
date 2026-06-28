#!/bin/bash
# ─── 云端微调训练 ───
# 在 AutoDL 服务器上执行（通过 SSH 远程命令运行）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/config/ssh_config.sh"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
REMOTE_DATA="/root/training_data/${ROBOT}_lerobot"
REMOTE_MODALITY_CONFIG="/root/training_data/${ROBOT}_modality_config.py"
REMOTE_OUTPUT="/root/checkpoints/${ROBOT}_finetune"
BASE_MODEL="${2:-nvidia/GR00T-N1.7-3B}"
MAX_STEPS="${3:-2000}"
NUM_GPUS="${4:-1}"

echo "🚀 云端微调训练"
echo "   模型: $BASE_MODEL"
echo "   数据: $REMOTE_DATA"
echo "   输出: $REMOTE_OUTPUT"
echo "   最大步数: $MAX_STEPS"
echo "   GPU 数: $NUM_GPUS"
echo ""

# 远端执行训练
ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} << REMOTE_SCRIPT
    set -e

    echo "📂 检查训练数据..."
    if [ ! -d "$REMOTE_DATA" ]; then
        echo "❌ 未找到训练数据: $REMOTE_DATA"
        exit 1
    fi
    ls -la "$REMOTE_DATA/meta/"
    echo ""

    echo "📂 检查 modality_config..."
    if [ -f "$REMOTE_MODALITY_CONFIG" ]; then
        cat "$REMOTE_MODALITY_CONFIG"
    else
        echo "⚠️  未找到 modality_config: $REMOTE_MODALITY_CONFIG"
        echo "   将使用默认配置"
    fi
    echo ""

    echo "🚀 开始训练..."
    cd /root/Isaac-GR00T

    # 检查 uv 是否可用
    if command -v uv &> /dev/null; then
        PYTHON_CMD="uv run python"
    else
        PYTHON_CMD="python3"
    fi

    CUDA_VISIBLE_DEVICES=0 \$PYTHON_CMD \\
        gr00t/experiment/launch_finetune.py \\
        --base-model-path $BASE_MODEL \\
        --dataset-path $REMOTE_DATA \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path $REMOTE_MODALITY_CONFIG \\
        --num-gpus $NUM_GPUS \\
        --output-dir $REMOTE_OUTPUT \\
        --max-steps $MAX_STEPS \\
        --save-steps $((MAX_STEPS / 4)) \\
        --save-total-limit 5 \\
        --global-batch-size 32 \\
        --dataloader-num-workers 4

    echo ""
    echo "✅ 训练完成！"
    echo "📂 检查点目录:"
    ls -la "$REMOTE_OUTPUT/"
REMOTE_SCRIPT

echo ""
echo "✅ 云端训练完成！"
echo "   模型路径: ${SSH_HOST}:${REMOTE_OUTPUT}"
echo "   下一步: bash scripts/07_download_model.sh $ROBOT"
