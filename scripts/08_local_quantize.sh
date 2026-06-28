#!/bin/bash
# ─── 本地 INT4 量化 ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_finetune}"
OUTPUT_DIR="${3:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"

echo "⚙️  INT4 量化"
echo "   输入: $MODEL_PATH"
echo "   输出: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

python3 -m src.export_int4 \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device auto

echo ""
echo "✅ 量化完成: $OUTPUT_DIR"
echo "   下一步: bash scripts/09_local_verify.sh $ROBOT"
