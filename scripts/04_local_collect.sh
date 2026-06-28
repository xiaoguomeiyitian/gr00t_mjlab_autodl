#!/bin/bash
# ─── 本地数据采集 ───
# 在本地电脑上运行 MJLab 仿真环境采集演示数据
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
NUM_EPISODES="${2:-50}"
EPISODE_LENGTH="${3:-300}"
ACTION_MODE="${4:-delta}"
OUTPUT_DIR="${5:-$SCRIPT_DIR/../output/${ROBOT}_raw}"

echo "🤖 数据采集"
echo "   机器人: $ROBOT"
echo "   Episodes: $NUM_EPISODES"
echo "   每 episode 步数: $EPISODE_LENGTH"
echo "   动作模式: $ACTION_MODE"
echo "   输出: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

python3 -m src.collect_data \
    --robot "$ROBOT" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    --action-mode "$ACTION_MODE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ 采集完成: $OUTPUT_DIR"
