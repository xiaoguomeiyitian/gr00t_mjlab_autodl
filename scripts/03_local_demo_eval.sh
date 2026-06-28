#!/bin/bash
# ─── 本地 Demo 推理 ───
# 在本地电脑上执行（需要先建立 SSH 隧道）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 默认参数 ───
DATASET_PATH="${1:-$SCRIPT_DIR/../Isaac-GR00T/demo_data/droid_sample}"
EMBODIMENT_TAG="${2:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
HOST="${3:-127.0.0.1}"
PORT="${4:-5555}"
OUTPUT_DIR="${5:-$SCRIPT_DIR/../output}"
TRAJ_IDS="${6:-1 2}"
ACTION_HORIZON="${7:-8}"

echo "🚀 GR00T Demo 推理"
echo "   数据集: $DATASET_PATH"
echo "   具身: $EMBODIMENT_TAG"
echo "   服务器: ${HOST}:${PORT}"
echo "   输出: $OUTPUT_DIR"
echo "   轨迹: $TRAJ_IDS"
echo "   Action Horizon: $ACTION_HORIZON"
echo ""

# 检查 Isaac-GR00T 是否存在
ISAAC_DIR="$SCRIPT_DIR/../Isaac-GR00T"
if [ ! -d "$ISAAC_DIR" ]; then
    echo "❌ 未找到 Isaac-GR00T: $ISAAC_DIR"
    echo "   请确保 Isaac-GR00T 与本项目在同一父目录下"
    exit 1
fi

# 检查数据集
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 未找到数据集: $DATASET_PATH"
    echo "   请确保 Isaac-GR00T 已正确克隆（含 demo_data）"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/../Isaac-GR00T:$SCRIPT_DIR/..:$PYTHONPATH"

# 执行推理
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/..')

from src.demo_eval import run_demo_eval

run_demo_eval(
    dataset_path='$DATASET_PATH',
    embodiment_tag='$EMBODIMENT_TAG',
    host='$HOST',
    port=int('$PORT'),
    output_dir='$OUTPUT_DIR',
    traj_ids=[int(x) for x in '$TRAJ_IDS'.split()],
    action_horizon=int('$ACTION_HORIZON'),
)
"

echo ""
echo "✅ 推理完成！"
echo "   结果保存在: $OUTPUT_DIR"
echo ""
echo "   查看图片:"
for traj_id in $TRAJ_IDS; do
    if [ -f "$OUTPUT_DIR/traj_${traj_id}.jpeg" ]; then
        echo "   - $OUTPUT_DIR/traj_${traj_id}.jpeg"
    fi
done
