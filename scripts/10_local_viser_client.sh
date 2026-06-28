#!/bin/bash
# ─── Viser 浏览器可视化 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ROBOT="${1:-g1}"
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"
PORT="${3:-20006}"

echo "🌐 Viser 浏览器可视化"
echo "   机器人: $ROBOT"
echo "   模型: $MODEL_PATH"
echo "   端口: $PORT"
echo ""
echo "   浏览器打开: http://localhost:${PORT}"
echo ""

cd "$SCRIPT_DIR"

python3 -m src.viz.viser_viewer \
    --robot "$ROBOT" \
    --model-path "$MODEL_PATH" \
    --port "$PORT"
