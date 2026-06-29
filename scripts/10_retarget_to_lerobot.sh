#!/bin/bash
# ─── 将 robot_retargeter 的运动数据转换为 LeRobot v2 格式 ───
# 在本地电脑上运行
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 自动检测 Python（优先 .venv）───
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# ─── 默认参数 ───
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

# 检查输入
if [ -z "$MOTION_FILE" ]; then
    echo "❌ 未指定动作文件"
    echo ""
    echo "用法: bash scripts/10_retarget_to_lerobot.sh <robot> <motion_file> [output_dir] [episode_length] [overlap] [fps] [task] [mjcf]"
    echo ""
    echo "示例:"
    echo "   bash scripts/10_retarget_to_lerobot.sh g1 ../robot_retargeter/output_data/robot_motion/Form_1_stageii_g1.csv"
    echo "   bash scripts/10_retarget_to_lerobot.sh g1 ../robot_retargeter/output_data/npz/xxx.npz output/g1_dance"
    echo ""
    echo "可用的动作文件:"
    ls -la ../robot_retargeter/output_data/robot_motion/*.csv 2>/dev/null | head -10 || echo "   (无)"
    ls -la ../robot_retargeter/output_data/npz/*.npz 2>/dev/null | head -10 || echo "   (无)"
    exit 1
fi

if [ ! -f "$MOTION_FILE" ]; then
    echo "❌ 动作文件不存在: $MOTION_FILE"
    exit 1
fi

# 构建命令
CMD="$PYTHON -m src.retarget_to_lerobot"
CMD="$CMD --robot $ROBOT"
CMD="$CMD --output $OUTPUT_DIR"
if [ -n "$EPISODE_LENGTH" ]; then
    CMD="$CMD --episode-length $EPISODE_LENGTH"
fi
CMD="$CMD --overlap $OVERLAP"

# 判断文件类型
if [[ "$MOTION_FILE" == *.csv ]]; then
    CMD="$CMD --csv $MOTION_FILE"
elif [[ "$MOTION_FILE" == *.npz ]]; then
    CMD="$CMD --npz $MOTION_FILE"
fi

# 可选参数
if [ -n "$FPS" ]; then
    CMD="$CMD --fps $FPS"
fi
if [ -n "$TASK" ]; then
    CMD="$CMD --task \"$TASK\""
fi
if [ -n "$MJCF" ]; then
    CMD="$CMD --mjcf $MJCF"
fi
if [ -n "$NO_VIDEO" ]; then
    CMD="$CMD --no-video"
fi

# 执行
echo "执行: $CMD"
echo ""
eval $CMD

echo ""
echo "✅ 转换完成！"
echo "   输出: $OUTPUT_DIR"
echo ""
echo "   下一步:"
echo "   bash scripts/05_upload_to_autodl.sh $ROBOT $OUTPUT_DIR"
