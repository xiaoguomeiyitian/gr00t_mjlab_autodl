#!/bin/bash
# ─── 批量转换 robot_retargeter 的运动数据 ───
# 扫描指定目录下所有 CSV/NPZ 文件，逐个转换为 LeRobot v2 格式
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

ROBOT="${1:-g1}"
INPUT_DIR="${2:-$SCRIPT_DIR/../robot_retargeter/output_data/robot_motion}"
OUTPUT_BASE="${3:-$SCRIPT_DIR/output/${ROBOT}_all_retarget}"
EPISODE_LENGTH="${4:-300}"
OVERLAP="${5:-0.5}"

echo "📦 批量转换 robot_retargeter 运动数据"
echo "   机器人: $ROBOT"
echo "   输入目录: $INPUT_DIR"
echo "   输出目录: $OUTPUT_BASE"
echo ""

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    echo "用法: bash scripts/11_batch_retarget.sh <robot> [input_dir] [output_dir] [episode_length] [overlap]"
    exit 1
fi

FILES=()
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "$INPUT_DIR" -maxdepth 1 \( -name "*_${ROBOT}.csv" -o -name "*_${ROBOT}.npz" \) -print0 2>/dev/null)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "❌ 未找到 ${ROBOT} 的动作文件"
    echo "   搜索路径: $INPUT_DIR/*_${ROBOT}.csv, $INPUT_DIR/*_${ROBOT}.npz"
    exit 1
fi

echo "找到 ${#FILES[@]} 个动作文件："
for f in "${FILES[@]}"; do
    echo "   $(basename "$f")"
done
echo ""

MERGED_DIR="$OUTPUT_BASE"
mkdir -p "$MERGED_DIR"

SUCCESS=0
FAILED=0
for f in "${FILES[@]}"; do
    FILENAME=$(basename "$f")
    MOTION_NAME="${FILENAME%_${ROBOT}.csv}"
    MOTION_NAME="${MOTION_NAME%_${ROBOT}.npz}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 转换: $FILENAME → $MOTION_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$f" == *.csv ]]; then
        SOURCE_ARG="--csv $f"
    else
        SOURCE_ARG="--npz $f"
    fi

    cd "$SCRIPT_DIR"
    if $PYTHON -m src.retarget_to_lerobot \
        $SOURCE_ARG --robot "$ROBOT" --output "$MERGED_DIR" \
        --episode-length "$EPISODE_LENGTH" --overlap "$OVERLAP" --no-video; then
        SUCCESS=$((SUCCESS + 1))
        echo "✅ 成功: $FILENAME"
    else
        FAILED=$((FAILED + 1))
        echo "❌ 失败: $FILENAME"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 批量转换完成"
echo "   成功: $SUCCESS"
echo "   失败: $FAILED"
echo "   输出: $MERGED_DIR"
echo "下一步: bash scripts/05_upload_to_autodl.sh $ROBOT $MERGED_DIR"
