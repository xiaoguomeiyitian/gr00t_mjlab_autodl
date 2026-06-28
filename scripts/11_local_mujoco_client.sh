#!/bin/bash
# ─── MuJoCo 原生可视化 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ROBOT="${1:-g1}"
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"
STEPS="${3:-300}"
FPS="${4:-30}"

echo "🖥️  MuJoCo 原生可视化"
echo "   机器人: $ROBOT"
echo "   模型: $MODEL_PATH"
echo "   步数: $STEPS"
echo "   帧率: $FPS"
echo ""
echo "   按键: 空格=暂停 R=重播 Esc=退出"
echo ""

cd "$SCRIPT_DIR"

python3 -m src.viz.mujoco_viewer \
    --robot "$ROBOT" \
    --model-path "$MODEL_PATH" \
    --steps "$STEPS" \
    --fps "$FPS"
