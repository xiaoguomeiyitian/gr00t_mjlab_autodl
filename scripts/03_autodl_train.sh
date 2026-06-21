#!/usr/bin/env bash
# ============================================================================
# 第 3 步: AutoDL 云端 Fine-tune 训练 (解压数据 + 训练 + 打包模型)
#
# 运行环境: AutoDL 云端实例 (RTX 5090 32GB / A100 推荐)
# 作用: 解压 [2] 上传的训练包 → Fine-tune GR00T → INT4 量化 → 打包模型
#
# 前置条件 (一次性):
#   - 已运行 00_autodl_init.sh 初始化云端环境
#     - Isaac-GR00T 已克隆到 /root/Isaac-GR00T
#     - Python 3.10 venv 在 /root/Isaac-GR00T/.venv
#     - 基础模型已在 /root/models/GR00T-N1-{size}/
#   - 已通过 [2] ./02_upload_to_autodl.sh 上传训练包到 /root/workspace/
#
# 用法 (在云端运行):
#   bash 03_autodl_train.sh --robot g1
#   bash 03_autodl_train.sh --robot g1 --epochs 20 --batch-size 2
#   bash 03_autodl_train.sh --robot g1 --skip-fp16      # 云端只导出 INT4
#   bash 03_autodl_train.sh --robot g1 --no-export-int4 # 完全跳过 INT4 (本地量化)
# ============================================================================
set -euo pipefail

# ── 颜色 ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
log()   { echo -e "${CYAN}[·]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ───────────────────────────────────────────────────────────────────
ROBOT="g1"
NUM_EPOCHS=10
BATCH_SIZE=2
GRAD_ACCUM=2
LEARNING_RATE=1e-4
SKIP_FP16=false            # 跳过 FP16 全量导出 (默认生成)
SKIP_INT4=false            # 跳过 INT4 量化 (用 --no-export-int4)

# 云端路径 (与 00_autodl_init.sh 保持一致)
GR00T_REPO="/root/Isaac-GR00T"
DATA_DIR="/root/data"
MODELS_DIR="/root/models"
BASE_MODEL_DIR=""          # 自动探测
OUTPUT_DIR=""              # 默认 $MODELS_DIR/${ROBOT}_gr00t
FULL_FP16_DIR=""           # 默认 $MODELS_DIR/${ROBOT}_gr00t_full_fp16
INT4_DIR=""                # 默认 $MODELS_DIR/${ROBOT}_gr00t_int4
PACK_NAME=""               # 训练包 (自动探测)
MODEL_SIZE="1.7-3B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";          shift 2 ;;
        --epochs)       NUM_EPOCHS="$2";     shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";     shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2";     shift 2 ;;
        --lr)           LEARNING_RATE="$2";  shift 2 ;;
        --data-dir)     DATA_DIR="$2";       shift 2 ;;
        --models-dir)   MODELS_DIR="$2";     shift 2 ;;
        --gr00t-repo)   GR00T_REPO="$2";     shift 2 ;;
        --model-size)   MODEL_SIZE="$2";     shift 2 ;;
        --skip-fp16)    SKIP_FP16=true;      shift   ;;
        --no-export-int4) SKIP_INT4=true;    shift   ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── 路径归一化 ─────────────────────────────────────────────────────────────
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
LEROBOT_DATA_DIR="$DATA_DIR/${ROBOT}_lerobot"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_DIR/${ROBOT}_gr00t}"
FULL_FP16_DIR="${FULL_FP16_DIR:-$MODELS_DIR/${ROBOT}_gr00t_full_fp16}"
INT4_DIR="${INT4_DIR:-$MODELS_DIR/${ROBOT}_gr00t_int4}"
FULL_FP16_PACK_NAME="${ROBOT}_gr00t_full_fp16.tar.gz"
INT4_PACK_NAME="${ROBOT}_gr00t_int4_model.tar.gz"
MODEL_PACK_NAME="${ROBOT}_gr00t_model.tar.gz"   # FP16 别名 (兼容旧脚本)

# 自动探测基础模型
if [ -z "$BASE_MODEL_DIR" ]; then
    for cand in "$MODELS_DIR/GR00T-N1-${MODEL_SIZE}" "$MODELS_DIR/GR00T-N1.7-3B" "$MODELS_DIR/GR00T-N1-3B"; do
        if [ -d "$cand" ] && [ "$(ls -A "$cand" 2>/dev/null)" ]; then
            BASE_MODEL_DIR="$cand"
            break
        fi
    done
fi
# 反推 model-size
if [ -n "$BASE_MODEL_DIR" ]; then
    DETECTED_SIZE=$(basename "$BASE_MODEL_DIR" | sed -n 's/^GR00T-N1-\(.*\)$/\1/p')
    [ -n "$DETECTED_SIZE" ] && MODEL_SIZE="$DETECTED_SIZE"
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 第 3 步: 云端训练          ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "机器人:        $ROBOT"
info "模型:          GR00T-N1-$MODEL_SIZE"
info "Epochs:        $NUM_EPOCHS"
info "Batch Size:    $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
info "Learning Rate: $LEARNING_RATE"
info "GR00T 仓库:    $GR00T_REPO"
info "数据目录:      $LEROBOT_DATA_DIR"
info "输出 LoRA:     $OUTPUT_DIR"
info "FP16 全量:     $([ "$SKIP_FP16" = true ] && echo '跳过' || echo "$FULL_FP16_DIR")"
info "INT4 量化:     $([ "$SKIP_INT4" = true ] && echo '跳过' || echo "$INT4_DIR")"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 0: 前置检查
# ════════════════════════════════════════════════════════════════════════════
step "Phase 0: 环境检查..."

# GPU
if ! command -v nvidia-smi &>/dev/null; then
    fail "未检测到 NVIDIA GPU, 请确认 PyTorch 2.x + CUDA 12.x 镜像"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
info "GPU: $GPU_NAME ($GPU_MEM)"

# GR00T 仓库
if [ ! -d "$GR00T_REPO" ]; then
    fail "Isaac-GR00T 不存在: $GR00T_REPO
请先运行一次性初始化: bash scripts/00_autodl_init.sh"
fi
info "GR00T 仓库: ✓"

# venv
if [ ! -f "$GR00T_REPO/.venv/bin/activate" ]; then
    fail "Python venv 不存在: $GR00T_REPO/.venv
请先运行一次性初始化: bash scripts/00_autodl_init.sh"
fi
# shellcheck disable=SC1091
source "$GR00T_REPO/.venv/bin/activate"
info "venv: ✓ (Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'))"

# 基础模型
if [ -z "$BASE_MODEL_DIR" ] || [ ! -d "$BASE_MODEL_DIR" ]; then
    fail "未找到 GR00T 基础模型, 预期在 $MODELS_DIR/GR00T-N1-*/"
fi
info "基础模型: ✓ $BASE_MODEL_DIR"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: 解压训练数据
# ════════════════════════════════════════════════════════════════════════════
step "Phase 1/4: 解压训练数据..."

if [ -d "$LEROBOT_DATA_DIR" ] && [ "$(ls -A "$LEROBOT_DATA_DIR" 2>/dev/null)" ]; then
    info "训练数据已存在: $LEROBOT_DATA_DIR (跳过解压)"
else
    # 查找上传的 tar.gz
    REMOTE_PACK=""
    for base in /root/workspace /workspace /root /root/data; do
        p="$base/$PACK_NAME"
        if [ -f "$p" ]; then
            REMOTE_PACK="$p"
            break
        fi
    done
    if [ -z "$REMOTE_PACK" ]; then
        fail "未找到训练包: $PACK_NAME
请先通过 [2] ./02_upload_to_autodl.sh 上传
常见路径: /root/workspace/  /root/  /workspace/"
    fi
    info "找到训练包: $REMOTE_PACK"
    mkdir -p "$DATA_DIR"
    tar -xzf "$REMOTE_PACK" -C "$DATA_DIR"
    if [ ! -d "$LEROBOT_DATA_DIR" ]; then
        fail "解压后仍未找到: $LEROBOT_DATA_DIR"
    fi
    info "训练数据已解压: $LEROBOT_DATA_DIR"
fi
EPISODE_COUNT=$(find "$LEROBOT_DATA_DIR/data" -name "*.parquet" 2>/dev/null | wc -l)
info "Episode 数: $EPISODE_COUNT"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: Fine-tune 训练 (LoRA)
# ════════════════════════════════════════════════════════════════════════════
step "Phase 2/4: Fine-tune 训练 (LoRA)..."
cd "$GR00T_REPO"

mkdir -p "$OUTPUT_DIR"
python3 gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_DIR" \
    --dataset-path "$LEROBOT_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --embodiment-tag "NEW_EMBODIMENT" \
    --use-lora

info "✅ Fine-tune 训练完成"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: 合并 LoRA → FP16 全量 (可选)
# ════════════════════════════════════════════════════════════════════════════
if $SKIP_FP16; then
    warn "Phase 3: 跳过 FP16 全量导出 (--skip-fp16)"
    FULL_FP16_DIR=""
else
    step "Phase 3/4: 合并 LoRA → FP16 全量模型..."

    MERGE_SCRIPT="/root/gr00t_mjlab_autodl/src/merge_lora.py"
    if [ ! -f "$MERGE_SCRIPT" ]; then
        # fallback: 随训练脚本同步上传的旧路径
        MERGE_SCRIPT="/root/workspace/gr00t_mjlab_autodl/src/merge_lora.py"
    fi
    if [ ! -f "$MERGE_SCRIPT" ]; then
        warn "未找到 merge_lora.py, 跳过 FP16 全量生成"
        FULL_FP16_DIR=""
    else
        python3 "$MERGE_SCRIPT" \
            --base-model "$BASE_MODEL_DIR" \
            --lora-path "$OUTPUT_DIR" \
            --output-dir "$FULL_FP16_DIR"
        info "✅ FP16 全量模型: $FULL_FP16_DIR"
    fi
fi
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 3.5: INT4 量化导出 (可选)
# ════════════════════════════════════════════════════════════════════════════
if $SKIP_INT4; then
    warn "Phase 3.5: 跳过 INT4 量化 (--no-export-int4)"
    INT4_DIR=""
else
    step "Phase 3.5/4: INT4 量化导出..."

    EXPORT_SCRIPT="/root/gr00t_mjlab_autodl/src/export_int4.py"
    if [ ! -f "$EXPORT_SCRIPT" ]; then
        EXPORT_SCRIPT="/root/workspace/gr00t_mjlab_autodl/src/export_int4.py"
    fi

    # 量化输入: 优先 FP16 全量 (质量更好), 否则 LoRA 合并产物
    QUANT_INPUT="$FULL_FP16_DIR"
    if [ -z "$QUANT_INPUT" ] || [ ! -d "$QUANT_INPUT" ]; then
        QUANT_INPUT="$OUTPUT_DIR"   # LoRA 也能直接量化
        warn "未找到 FP16 全量, 用 LoRA 产物 ($QUANT_INPUT) 做 INT4 量化"
    fi

    if [ ! -f "$EXPORT_SCRIPT" ]; then
        warn "未找到 export_int4.py, 跳过 INT4"
        INT4_DIR=""
    else
        python3 "$EXPORT_SCRIPT" \
            --input-type fp16 \
            --model-dir "$QUANT_INPUT" \
            --output-dir "$INT4_DIR" \
            --model-path "$BASE_MODEL_DIR"
        info "✅ INT4 量化模型: $INT4_DIR"
    fi
fi
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 4: 打包模型
# ════════════════════════════════════════════════════════════════════════════
step "Phase 4/4: 打包模型..."

PACK_BASE_DIR="/root/workspace"
mkdir -p "$PACK_BASE_DIR"
cd "$MODELS_DIR"

PACK_COUNT=0

# ── FP16 全量包 ────────────────────────────────────────────────────────────
if [ -n "$FULL_FP16_DIR" ] && [ -d "$FULL_FP16_DIR" ]; then
    tar -czf "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" \
        -C "$(dirname "$FULL_FP16_DIR")" "$(basename "$FULL_FP16_DIR")"
    FP16_SIZE=$(du -h "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" | cut -f1)
    info "✅ FP16 全量包: $FULL_FP16_PACK_NAME ($FP16_SIZE) → $PACK_BASE_DIR/"
    # 别名 (兼容)
    cp "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" "$PACK_BASE_DIR/$MODEL_PACK_NAME"
    PACK_COUNT=$((PACK_COUNT + 1))
else
    # 仅打包 LoRA
    tar -czf "$PACK_BASE_DIR/$MODEL_PACK_NAME" \
        -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    LORA_SIZE=$(du -h "$PACK_BASE_DIR/$MODEL_PACK_NAME" | cut -f1)
    warn "LoRA adapter 包: $MODEL_PACK_NAME ($LORA_SIZE)"
    PACK_COUNT=$((PACK_COUNT + 1))
fi

# ── INT4 量化包 ────────────────────────────────────────────────────────────
if [ -n "$INT4_DIR" ] && [ -d "$INT4_DIR" ]; then
    tar -czf "$PACK_BASE_DIR/$INT4_PACK_NAME" \
        -C "$(dirname "$INT4_DIR")" "$(basename "$INT4_DIR")"
    INT4_SIZE=$(du -h "$PACK_BASE_DIR/$INT4_PACK_NAME" | cut -f1)
    info "✅ INT4 量化包: $INT4_PACK_NAME ($INT4_SIZE) → $PACK_BASE_DIR/"
    PACK_COUNT=$((PACK_COUNT + 1))
fi

# ════════════════════════════════════════════════════════════════════════════
# 完成
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🎉 第 3 步完成!${NC}"
echo ""
echo "云端生成的模型包 ($PACK_COUNT 个):"
[ -f "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$FULL_FP16_PACK_NAME  (FP16 全量, ~7GB, 高显存 GPU)"
[ -f "$PACK_BASE_DIR/$MODEL_PACK_NAME" ] && [ "$MODEL_PACK_NAME" != "$FULL_FP16_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$MODEL_PACK_NAME       (FP16 别名, 兼容)"
[ -f "$PACK_BASE_DIR/$INT4_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$INT4_PACK_NAME         (INT4 量化, ~1.5GB, 低显存 GPU)"
echo ""
echo -e "${BOLD}进入第 4 步: 本地下载${NC}"
echo ""
echo "回到本地, 运行:"
echo "  ./scripts/04_download_model.sh --robot $ROBOT"
echo ""
