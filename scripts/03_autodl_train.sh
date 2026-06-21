#!/usr/bin/env bash
# ============================================================================
# 第 3 步: AutoDL 云端 Fine-tune 训练 (解压数据 + 训练 + 打包模型)
#
# 运行环境 (基于 Isaac-GR00T 官方 hardware_recommendation.md):
#   ⭐ 推荐: A100-40GB+ / L40-48GB / H100-40GB+ (官方背书, peak < 35GB)
#   ⚠️ 边缘: V100-32GB / RTX 5090-32GB (低于官方最低 40GB, 需小 batch + gradient checkpointing)
#   ❌ 不支持: < 24GB 显存 (会 OOM)
#
# 训练策略 (GR00T 官方默认, 见 gr00t/experiment/launch_finetune.py):
#   - 默认: 只微调 projector + diffusion action head (peak VRAM < 35GB)
#   - 启用 --tune-llm 或 --tune-visual: 需要 80GB+ VRAM
#   - 注: Isaac-GR00T 官方训练**不支持 LoRA** (peft 是依赖但未实际使用)
#         训练输出 OUTPUT_DIR 本身就是完整可推理模型, 无需 merge
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
#   bash 03_autodl_train.sh --robot g1 --epochs 10 --batch-size 2
#   bash 03_autodl_train.sh --robot g1 --no-export-int4  # 完全跳过 INT4 (本地量化)
#   bash 03_autodl_train.sh --robot g1 --max-steps 5000  # 直接指定步数 (官方参数)
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
MAX_STEPS=""               # 留空: 由 (episodes × epochs) 自动估算; 显式传值则覆盖
SKIP_INT4=false            # 跳过 INT4 量化 (用 --no-export-int4)
TUNE_LLM=false             # 官方默认 off (启用需 80GB+ VRAM)
TUNE_VISUAL=false          # 官方默认 off (启用需 80GB+ VRAM)
SAVE_ONLY_MODEL=true       # 只存模型权重 (省空间, 不可恢复训练)
ACTION_MODE=""             # 动作空间: absolute | delta | relative_eef (留空: 从 data_dir/info.json 读)

# 云端路径 (与 00_autodl_init.sh 保持一致)
GR00T_REPO="/root/Isaac-GR00T"
DATA_DIR="/root/data"
MODELS_DIR="/root/models"
BASE_MODEL_DIR=""          # 自动探测
OUTPUT_DIR=""              # 默认 $MODELS_DIR/${ROBOT}_gr00t  (训练输出 = final model)
INT4_DIR=""                # 默认 $MODELS_DIR/${ROBOT}_gr00t_int4
PACK_NAME=""               # 训练包 (自动探测)
MODEL_SIZE="1.7-3B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)            ROBOT="$2";            shift 2 ;;
        --epochs)           NUM_EPOCHS="$2";       shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        --grad-accum)       GRAD_ACCUM="$2";       shift 2 ;;
        --max-steps)        MAX_STEPS="$2";        shift 2 ;;
        --lr)               LEARNING_RATE="$2";    shift 2 ;;
        --data-dir)         DATA_DIR="$2";         shift 2 ;;
        --models-dir)       MODELS_DIR="$2";       shift 2 ;;
        --gr00t-repo)       GR00T_REPO="$2";       shift 2 ;;
        --model-size)       MODEL_SIZE="$2";       shift 2 ;;
        --no-export-int4)   SKIP_INT4=true;        shift   ;;
        --tune-llm)         TUNE_LLM=true;         shift   ;;
        --tune-visual)      TUNE_VISUAL=true;      shift   ;;
        --save-only-model)  SAVE_ONLY_MODEL=true;  shift   ;;
        --no-save-only-model) SAVE_ONLY_MODEL=false; shift ;;
        --action-mode)      ACTION_MODE="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── 路径归一化 ─────────────────────────────────────────────────────────────
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
LEROBOT_DATA_DIR="$DATA_DIR/${ROBOT}_lerobot"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_DIR/${ROBOT}_gr00t}"     # 训练输出 (final model, 可直接推理)
INT4_DIR="${INT4_DIR:-$MODELS_DIR/${ROBOT}_gr00t_int4}"
MODEL_PACK_NAME="${ROBOT}_gr00t_model.tar.gz"               # 最终 BF16 模型包
INT4_PACK_NAME="${ROBOT}_gr00t_int4_model.tar.gz"           # INT4 量化包
# 兼容旧名 (旧文档/04_download_model.sh 可能还在引用)
FULL_FP16_PACK_NAME="${ROBOT}_gr00t_full_fp16.tar.gz"

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
info "Tune LLM:      $TUNE_LLM  (官方默认 off; 启用需 80GB+ VRAM)"
info "Tune Visual:   $TUNE_VISUAL  (官方默认 off; 启用需 80GB+ VRAM)"
info "Save Only:     $SAVE_ONLY_MODEL  (true=省空间, false=可恢复训练)"
info "GR00T 仓库:    $GR00T_REPO"
info "数据目录:      $LEROBOT_DATA_DIR"
info "输出 (final):  $OUTPUT_DIR  (官方训练输出, 直接可推理)"
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

# ── 0.5: 读 modality.json + info.json, 检查 action_mode 一致性 ─────
if [ -f "$LEROBOT_DATA_DIR/meta/modality.json" ]; then
    info "modality.json: ✓"
    if [ -z "$ACTION_MODE" ] && command -v python3 &>/dev/null; then
        ACTION_MODE=$(python3 -c "
import json
m = json.load(open('$LEROBOT_DATA_DIR/meta/modality.json'))
a = m.get('action', {})
if 'joint_position_delta' in a: print('delta')
elif 'ee_pose_delta' in a: print('relative_eef')
elif 'joint_position_target' in a: print('absolute')
else: print('absolute')
" 2>/dev/null || echo "absolute")
        info "从 modality.json 自动探测 action_mode = $ACTION_MODE"
    fi
fi
[ -z "$ACTION_MODE" ] && ACTION_MODE="absolute"
info "训练动作空间: $ACTION_MODE  (与采集时一致很重要!)"

# 列出实际视频数量 (如有)
VIDEO_COUNT=0
if [ -d "$LEROBOT_DATA_DIR/videos/chunk-000" ]; then
    VIDEO_COUNT=$(find "$LEROBOT_DATA_DIR/videos/chunk-000" -name "*.mp4" 2>/dev/null | wc -l)
fi
info "视频文件数: $VIDEO_COUNT"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: Fine-tune 训练 (官方默认: projector + diffusion, 关闭 LLM/visual)
# ════════════════════════════════════════════════════════════════════════════
step "Phase 2/4: Fine-tune 训练..."
cd "$GR00T_REPO"
mkdir -p "$OUTPUT_DIR"

# 估算 max_steps (如果用户没显式指定)
# 公式: steps = (episodes × assumed_frames × epochs) / (batch × grad_accum)
# 假设平均每 episode ~50 帧 (GR00T 官方 LeRobotDataConfig 常见)
if [ -z "$MAX_STEPS" ]; then
    ASSUMED_FRAMES_PER_EP=50
    EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
    MAX_STEPS=$(( (EPISODE_COUNT * ASSUMED_FRAMES_PER_EP * NUM_EPOCHS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
    [ "$MAX_STEPS" -lt 100 ] && MAX_STEPS=100   # 下限保护
    info "自动估算 max_steps=$MAX_STEPS  (${EPISODE_COUNT} episodes × ${NUM_EPOCHS} epochs × ~${ASSUMED_FRAMES_PER_EP}f / (${BATCH_SIZE}×${GRAD_ACCUM}))"
fi

# 构造条件参数 (tyro 不接受空字符串, 只在启用时才传)
EXTRA_ARGS=()
[ "$TUNE_LLM" = true ]    && EXTRA_ARGS+=(--tune-llm)
[ "$TUNE_VISUAL" = true ] && EXTRA_ARGS+=(--tune-visual)
[ "$SAVE_ONLY_MODEL" = true ] && EXTRA_ARGS+=(--save-only-model)

python3 gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL_DIR" \
    --dataset-path "$LEROBOT_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --embodiment-tag "NEW_EMBODIMENT" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --num-gpus 1 \
    "${EXTRA_ARGS[@]}"

info "✅ Fine-tune 训练完成 (输出: $OUTPUT_DIR)"
echo ""

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: INT4 量化导出 (可选)
#   注: GR00T 官方训练不支持 LoRA, 故无 "LoRA → FP16" 合并步骤
#       训练输出 (OUTPUT_DIR) 本身就是完整可推理模型, 直接做 INT4 PTQ
# ════════════════════════════════════════════════════════════════════════════
if $SKIP_INT4; then
    warn "Phase 3: 跳过 INT4 量化 (--no-export-int4)"
    INT4_DIR=""
else
    step "Phase 3/4: INT4 量化导出..."

    EXPORT_SCRIPT="/root/gr00t_mjlab_autodl/src/export_int4.py"
    if [ ! -f "$EXPORT_SCRIPT" ]; then
        EXPORT_SCRIPT="/root/workspace/gr00t_mjlab_autodl/src/export_int4.py"
    fi

    if [ ! -f "$EXPORT_SCRIPT" ]; then
        warn "未找到 export_int4.py, 跳过 INT4"
        INT4_DIR=""
    else
        python3 "$EXPORT_SCRIPT" \
            --input-type fp16 \
            --model-dir "$OUTPUT_DIR" \
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

# ── Final model 包 (BF16 完整模型, 用于 40GB+ 显存推理) ─────────────
if [ -d "$OUTPUT_DIR" ]; then
    tar -czf "$PACK_BASE_DIR/$MODEL_PACK_NAME" \
        -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    MODEL_SIZE_H=$(du -h "$PACK_BASE_DIR/$MODEL_PACK_NAME" | cut -f1)
    info "✅ 最终模型包: $MODEL_PACK_NAME ($MODEL_SIZE_H) → $PACK_BASE_DIR/"
    PACK_COUNT=$((PACK_COUNT + 1))

    # 旧名兼容 (symlink, 不复制以省空间)
    if [ "$MODEL_PACK_NAME" != "$FULL_FP16_PACK_NAME" ]; then
        ln -sf "$MODEL_PACK_NAME" "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" 2>/dev/null || true
    fi
fi

# ── INT4 量化包 (~1.5GB, 用于 16GB+ 显存推理) ───────────────────────
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
[ -f "$PACK_BASE_DIR/$MODEL_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$MODEL_PACK_NAME  (~7GB, BF16 完整模型, 用于 40GB+ 显存)"
[ -f "$PACK_BASE_DIR/$FULL_FP16_PACK_NAME" ] && [ "$MODEL_PACK_NAME" != "$FULL_FP16_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$FULL_FP16_PACK_NAME  (→ symlink, 旧名兼容)"
[ -f "$PACK_BASE_DIR/$INT4_PACK_NAME" ] && \
    echo "  📦 $PACK_BASE_DIR/$INT4_PACK_NAME    (~1.5GB, INT4 量化, 用于 16GB+ 显存)"
echo ""
echo -e "${BOLD}进入第 4 步: 本地下载${NC}"
echo ""
echo "回到本地, 运行:"
echo "  ./scripts/04_download_model.sh --robot $ROBOT"
echo ""
