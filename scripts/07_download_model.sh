#!/bin/bash
# ─── 从 AutoDL 下载微调后的模型 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/config/ssh_config.sh"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
REMOTE_MODEL="${2:-/autodl-fs/data/checkpoints/${ROBOT}_finetune/checkpoint-${3:-2000}}"
LOCAL_DIR="${4:-$SCRIPT_DIR/../checkpoints/${ROBOT}_finetune}"

mkdir -p "$LOCAL_DIR"

echo "📥 下载模型"
echo "   远端: ${SSH_HOST}:${REMOTE_MODEL}"
echo "   本地: $LOCAL_DIR"
echo ""

# 检查远端目录
ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} "ls -la ${REMOTE_MODEL}/" || {
    echo "❌ 远端模型目录不存在: ${REMOTE_MODEL}"
    echo "   可用 checkpoint:"
    ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} "ls /autodl-fs/data/checkpoints/${ROBOT}_finetune/ 2>/dev/null || echo '   无'"
    exit 1
}

# 下载
scp -P ${SSH_PORT} -r ${SSH_USER}@${SSH_HOST}:"${REMOTE_MODEL}"/* "$LOCAL_DIR/"

echo ""
echo "✅ 下载完成"
ls -lh "$LOCAL_DIR/"
echo ""
echo "   文件大小: $(du -sh "$LOCAL_DIR" | cut -f1)"
