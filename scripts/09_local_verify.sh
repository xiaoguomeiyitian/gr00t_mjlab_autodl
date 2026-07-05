#!/bin/bash
# ─── 本地推理验证 — 加载量化模型，在本地推理验证 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

ROBOT="${1:-g1}"
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"
DATASET_PATH="${3:-$SCRIPT_DIR/output/${ROBOT}_lerobot}"
OUTPUT_DIR="${4:-$SCRIPT_DIR/../output/verify}"
VIS_MODE="${5:-viser}"
# 微调后的模型用 NEW_EMBODIMENT（与 06_autodl_train.sh 一致）；
# 预训练模型用 OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
EMBODIMENT_TAG="${6:-NEW_EMBODIMENT}"

echo "🔍 本地推理验证"
echo "   模型: $MODEL_PATH"
echo "   数据: $DATASET_PATH"
echo "   可视化: $VIS_MODE"
echo ""

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

case "$VIS_MODE" in
    viser)
        echo "🌐 Viser + Policy Server 推理可视化"
        $PYTHON -m src.viz.viser_infer \
            --robot "$ROBOT" --host 127.0.0.1 --port 5555 \
            --viser-port 20006 --dataset "$DATASET_PATH" \
            --embodiment-tag "$EMBODIMENT_TAG"
        ;;
    mujoco)
        echo "🖥️  MuJoCo + Policy Server 推理可视化"
        $PYTHON -m src.viz.mujoco_infer \
            --robot "$ROBOT" --host 127.0.0.1 --port 5555 \
            --dataset "$DATASET_PATH" \
            --embodiment-tag "$EMBODIMENT_TAG"
        ;;
esac

echo "✅ 验证完成"
