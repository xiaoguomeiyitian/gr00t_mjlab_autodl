#!/usr/bin/env bash
# ============================================================================
# 容器内: 数据采集入口
# 默认: 收集 100 episodes + 转换为 LeRobot v2 + 打包
# ============================================================================
set -euo pipefail

cd "${GR00T_MJLAB_ROOT:-/root/gr00t_mjlab_autodl}"

ROBOT="${ROBOT:-g1}"
NUM_EPISODES="${NUM_EPISODES:-100}"
INSTRUCTION="${INSTRUCTION:-walk forward}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"
DATA_DIR="${DATA_DIR:-/root/data/${ROBOT}_raw}"
LEROBOT_DIR="${LEROBOT_DIR:-/root/data/${ROBOT}_lerobot}"
TASK_ID="${TASK_ID:-}"

# 自动选择 task
if [ -z "$TASK_ID" ]; then
    case "$ROBOT" in
        g1)  TASK_ID="Unitree-G1-Flat" ;;
        go2) TASK_ID="Unitree-Go2-Flat" ;;
        *)   echo "未知机器人: $ROBOT (支持 g1/go2)"; exit 1 ;;
    esac
fi

echo "=========================================="
echo " [1] 数据采集: ${ROBOT} (${TASK_ID})"
echo "=========================================="
echo "Episodes:  ${NUM_EPISODES}"
echo "Length:    ${EPISODE_LENGTH} steps"
echo "指令:      ${INSTRUCTION}"
echo "输出:      ${DATA_DIR}"
echo ""

# ── 步骤 1: mjlab 仿真收集 ──────────────────────────────────────────────
python3 src/collect_data.py \
    --task "${TASK_ID}" \
    --robot "${ROBOT}" \
    --num-episodes "${NUM_EPISODES}" \
    --episode-length "${EPISODE_LENGTH}" \
    --instruction "${INSTRUCTION}" \
    --output-dir "${DATA_DIR}"

# ── 步骤 2: 转换为 LeRobot v2 ───────────────────────────────────────────
echo ""
echo "=========================================="
echo " [2] 格式转换: npz → LeRobot v2"
echo "=========================================="
echo "输出:  ${LEROBOT_DIR}"
echo ""

python3 src/convert_to_lerobot.py \
    --robot "${ROBOT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${LEROBOT_DIR}"

# ── 步骤 3: 打包为训练包 ────────────────────────────────────────────────
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
PACK_PATH="/root/${PACK_NAME}"

echo ""
echo "=========================================="
echo " [3] 打包"
echo "=========================================="
cd /root
tar -czf "${PACK_NAME}" \
    -C "$(dirname "${LEROBOT_DIR}")" \
    "$(basename "${LEROBOT_DIR}")"
echo "训练包: ${PACK_PATH}"
ls -lh "${PACK_PATH}"
