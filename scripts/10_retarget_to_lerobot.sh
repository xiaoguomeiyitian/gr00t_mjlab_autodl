#!/bin/bash
# ─── 将 robot_retargeter 的运动数据转换为 LeRobot v2 格式 — 在本地电脑上运行 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

ROBOT="${1:-g1}"
MOTION_FILE="${2:-}"
OUTPUT_DIR="${3:-$SCRIPT_DIR/output/${ROBOT}_from_retarget}"
EPISODE_LENGTH="${4:-}"
OVERLAP="${5:-0.5}"
FPS="${6:-}"
TASK="${7:-}"
MJCF="${8:-}"
NO_VIDEO="${9:-}"

echo "📦 转换 robot_retargeter 运动数据 → LeRobot v2"
echo "   机器人: $ROBOT"
echo "   输出: $OUTPUT_DIR"
echo ""

if [ -z "$MOTION_FILE" ]; then
    echo "❌ 未指定动作文件"
    echo "用法: bash scripts/10_retarget_to_lerobot.sh <robot> <motion_file> [output_dir] ..."
    echo "示例:"
    echo "   bash scripts/10_retarget_to_lerobot.sh g1 ../robot_retargeter/output_data/robot_motion/Form_1_stageii_g1.csv"
    ls -la ../robot_retargeter/output_data/robot_motion/*.csv 2>/dev/null | head -5 || echo "   (无)"
    exit 1
fi

if [ ! -f "$MOTION_FILE" ]; then
    echo "❌ 动作文件不存在: $MOTION_FILE"
    exit 1
fi

CMD="$PYTHON -m src.retarget_to_lerobot"
CMD="$CMD --robot $ROBOT --output $OUTPUT_DIR"
[ -n "$EPISODE_LENGTH" ] && CMD="$CMD --episode-length $EPISODE_LENGTH"
CMD="$CMD --overlap $OVERLAP"

if [[ "$MOTION_FILE" == *.csv ]]; then
    CMD="$CMD --csv $MOTION_FILE"
elif [[ "$MOTION_FILE" == *.npz ]]; then
    CMD="$CMD --npz $MOTION_FILE"
fi

[ -n "$FPS" ] && CMD="$CMD --fps $FPS"
[ -n "$TASK" ] && CMD="$CMD --task \"$TASK\""
[ -n "$MJCF" ] && CMD="$CMD --mjcf $MJCF"
[ -n "$NO_VIDEO" ] && CMD="$CMD --no-video"

echo "执行: $CMD"
eval $CMD

echo "✅ 转换完成！输出: $OUTPUT_DIR"
echo "下一步: bash scripts/05_upload_to_autodl.sh $ROBOT $OUTPUT_DIR"
