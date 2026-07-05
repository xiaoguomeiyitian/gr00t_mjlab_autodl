#!/bin/bash
# ─── 云端微调训练 — 在 AutoDL 服务器上执行 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/config/ssh_config.sh"

ROBOT="${1:-g1}"
REMOTE_DATA="/root/gr00t_mjlab_autodl/${ROBOT}_lerobot"
REMOTE_MODALITY_CONFIG="/root/gr00t_mjlab_autodl/${ROBOT}_modality_config.py"
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
        echo "⚠️  未找到 modality_config，将使用默认配置"
    fi
    echo ""

    echo "🚀 开始训练..."
    # Isaac-GR00T 路径：优先 00_autodl_init.sh 的默认克隆位置（项目父目录），
    # 回退 AutoDL 数据盘 /autodl-fs/data/Isaac-GR00T
    ISAAC_DIR=""
    for cand in \"\$HOME/Isaac-GR00T\" \"/root/Isaac-GR00T\" \"/autodl-fs/data/Isaac-GR00T\"; do
        if [ -d \"\$cand\" ]; then
            ISAAC_DIR=\"\$cand\"
            break
        fi
    done
    if [ -z \"\$ISAAC_DIR\" ]; then
        echo \"❌ 未找到 Isaac-GR00T，请先执行 00_autodl_init.sh\"
        exit 1
    fi
    echo \"   Isaac-GR00T: \$ISAAC_DIR\"
    cd \"\$ISAAC_DIR\"

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

    echo "✅ 训练完成！"
    ls -la "$REMOTE_OUTPUT/"
REMOTE_SCRIPT

echo "✅ 云端训练完成！模型路径: ${SSH_HOST}:${REMOTE_OUTPUT}"
echo "下一步: bash scripts/07_download_model.sh $ROBOT"
