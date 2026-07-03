#!/bin/bash
# ─── 本地推理验证 ───
# 加载量化模型，在本地推理验证（Viser 或 MuJoCo 可视化）
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
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"
DATASET_PATH="${3:-$SCRIPT_DIR/output/${ROBOT}_lerobot}"
OUTPUT_DIR="${4:-$SCRIPT_DIR/../output/verify}"
VIS_MODE="${5:-viser}"  # viser / mujoco

echo "🔍 本地推理验证"
echo "   模型: $MODEL_PATH"
echo "   数据: $DATASET_PATH"
echo "   输出: $OUTPUT_DIR"
echo "   可视化: $VIS_MODE"
echo ""

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

case "$VIS_MODE" in
    viser)
        echo "🌐 Viser + Policy Server 推理可视化"
        $PYTHON -m src.viz.viser_infer \
            --robot "$ROBOT" \
            --host 127.0.0.1 \
            --port 5555 \
            --viser-port 20006 \
            --dataset "$DATASET_PATH" \
            --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
        ;;
    mujoco)
        echo "🖥️  MuJoCo + Policy Server 推理可视化"
        $PYTHON -m src.viz.mujoco_infer \
            --robot "$ROBOT" \
            --host 127.0.0.1 \
            --port 5555 \
            --dataset "$DATASET_PATH" \
            --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
        ;;
esac

echo ""
echo "✅ 验证完成"
