#!/bin/bash
# ─── 本地数据采集 — 运行 MJLab 仿真环境采集演示数据 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

ROBOT="${1:-g1}"
NUM_EPISODES="${2:-50}"
EPISODE_LENGTH="${3:-300}"
ACTION_MODE="${4:-delta}"
OUTPUT_DIR="${5:-$SCRIPT_DIR/output/${ROBOT}_raw}"

echo "🤖 数据采集"
echo "   机器人: $ROBOT"
echo "   Episodes: $NUM_EPISODES"
echo "   输出: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

$PYTHON -m src.collect_data \
    --robot "$ROBOT" --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" --action-mode "$ACTION_MODE" \
    --output-dir "$OUTPUT_DIR"

echo "✅ 采集完成: $OUTPUT_DIR"
