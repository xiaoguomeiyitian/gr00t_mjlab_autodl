#!/bin/bash
# ─── 上传训练数据到 AutoDL ───
# 本地运行：转换格式 → SCP 上传
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/config/ssh_config.sh"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
RAW_DIR="${2:-$SCRIPT_DIR/../${ROBOT}_raw}"
LEROBOT_DIR="${3:-$SCRIPT_DIR/../${ROBOT}_lerobot}"
REMOTE_DIR="${4:-/root/training_data}"
MODALITY_CONFIG="${5:-$SCRIPT_DIR/src/configs/${ROBOT}_modality_config.py}"

echo "📦 上传训练数据到 AutoDL"
echo ""

# ─── Step 1: 格式转换 ───
echo "🔄 Step 1: 格式转换 (npz+mp4 → LeRobot v2)"
if [ ! -d "$RAW_DIR" ]; then
    echo "❌ 未找到原始数据: $RAW_DIR"
    echo "   请先运行 scripts/04_local_collect.sh"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 -m src.convert_to_lerobot \
    --input-dir "$RAW_DIR" \
    --output-dir "$LEROBOT_DIR" \
    --robot "$ROBOT"

echo ""

# ─── Step 2: 上传 ───
echo "📤 Step 2: SCP 上传到 AutoDL"
echo "   本地: $LEROBOT_DIR"
echo "   远端: ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

# 创建远端目录
ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} "mkdir -p ${REMOTE_DIR}"

# 上传数据集
scp -P ${SSH_PORT} -r "$LEROBOT_DIR" ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/

# 上传 modality_config
if [ -f "$MODALITY_CONFIG" ]; then
    scp -P ${SSH_PORT} "$MODALITY_CONFIG" ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/
    echo "   ✅ 上传 modality_config: $(basename $MODALITY_CONFIG)"
else
    echo "   ⚠️  未找到 modality_config: $MODALITY_CONFIG"
fi

# 上传采集 metadata
if [ -f "$RAW_DIR/collection_meta.json" ]; then
    scp -P ${SSH_PORT} "$RAW_DIR/collection_meta.json" ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/
fi

# ─── Step 3: 远端校验 ───
echo ""
echo "🔍 Step 3: 远端校验"
ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} "
    echo '📁 远端目录:' && ls -la ${REMOTE_DIR}/
    echo ''
    if [ -d '${REMOTE_DIR}/${ROBOT}_lerobot/meta' ]; then
        echo '📋 modality.json:' && cat ${REMOTE_DIR}/${ROBOT}_lerobot/meta/modality.json
        echo ''
        echo '📋 info.json:' && cat ${REMOTE_DIR}/${ROBOT}_lerobot/meta/info.json
    fi
"

echo ""
echo "✅ 上传完成！"
echo "   下一步: bash scripts/06_autodl_train.sh $ROBOT"
