#!/bin/bash
# ─── 上传训练数据到 AutoDL ───
# 本地运行：SCP 上传（格式转换已在采集步骤中完成）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 自动检测 Python（优先 .venv）───
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi
source "$SCRIPT_DIR/config/ssh_config.sh"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
LEROBOT_DIR="${2:-$SCRIPT_DIR/output/${ROBOT}_lerobot}"
REMOTE_DIR="${3:-/root/gr00t_mjlab_autodl}"
MODALITY_CONFIG="${4:-$SCRIPT_DIR/src/configs/${ROBOT}_modality_config.py}"

echo "📦 上传训练数据到 AutoDL"
echo ""

# ─── 检查数据集目录 ───
if [ ! -d "$LEROBOT_DIR" ]; then
    echo "❌ 未找到 LeRobot 数据集: $LEROBOT_DIR"
    echo "   请先运行数据采集 (选项 5) 或 retarget (选项 12/13)"
    exit 1
fi

# ─── Step 1: 上传 ───
echo "📤 Step 1: SCP 上传到 AutoDL"
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

# ─── Step 2: 远端校验 ───
echo ""
echo "🔍 Step 2: 远端校验"
ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST} "
    echo '📁 远端目录:' && ls -la ${REMOTE_DIR}/
    echo ''
    if [ -d '${REMOTE_DIR}/$(basename $LEROBOT_DIR)/meta' ]; then
        echo '📋 modality.json:' && cat ${REMOTE_DIR}/$(basename $LEROBOT_DIR)/meta/modality.json
        echo ''
        echo '📋 info.json:' && cat ${REMOTE_DIR}/$(basename $LEROBOT_DIR)/meta/info.json
    fi
"

echo ""
echo "✅ 上传完成！"
echo "   下一步: bash scripts/06_autodl_train.sh $ROBOT"
