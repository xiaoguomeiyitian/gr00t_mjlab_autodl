#!/usr/bin/env bash
# ============================================================================
# ⚠️  DEPRECATED — 此脚本为旧版"all-in-one"训练脚本, 已被新 6 步流程替代
#
# 新流程 (见 start.sh + scripts/):
#   [0] 00_autodl_init.sh    一次性环境初始化 (克隆 GR00T + venv + 模型)
#   [1] 01_local_collect.sh  本地 mjlab 收集数据
#   [2] 02_upload_to_autodl.sh  上传训练包到云端
#   [3] 03_autodl_train.sh   云端训练 (官方 launch_finetune.py, 无 LoRA bug)
#   [4] 04_download_model.sh 下载模型到本地
#   [5] 05_local_quantize.sh 本地 INT4 量化 (可选)
#   [6] 06_local_verify.sh   本地推理验证
#
# 旧版问题:
#   1. 假设 Isaac-GR00T 训练使用 LoRA (--use-lora), 但官方实际不支持 LoRA
#      (peft 是依赖但未调用), 该参数会让 tyro.cli 报错
#   2. 使用 --num-epochs, 但官方 FinetuneConfig 只有 --max-steps
#   3. 路径硬编码 /workspace/ (非 AutoDL 默认 /root/), 兼容性差
#
# 此脚本仅保留以防需要回退, 强烈建议改用 start.sh 统一入口
# ============================================================================
# 原内容 (保留供回退参考):
#
# 第 2 步: AutoDL 云端环境搭建 + Fine-tune 训练
#
# 运行环境: AutoDL 云端实例 (A100 40GB+ / L40 48GB / H100 推荐)
#          V100-32GB / RTX 5090-32GB 边缘可行 (需小 batch)
# 作用: 克隆 GR00T → 安装依赖 → 下载模型 → 解压数据 → Fine-tune → INT4 量化 → 打包
#
# 前置条件:
#   - 已上传 {robot}_gr00t_training.tar.gz 到 /root/workspace/
#   - 实例镜像选择 PyTorch 2.7+ + CUDA 12.6+
#
# 用法 (DEPRECATED):
#   bash 02_autodl_train.sh --robot g1
#   bash 02_autodl_train.sh --robot g1 --epochs 20 --batch-size 2
#   bash 02_autodl_train.sh --robot g1 --skip-setup  # 跳过环境搭建
# ============================================================================
set -euo pipefail

# 一进来就警告
echo -e "\033[1;33m[⚠️  DEPRECATED]\033[0m 02_autodl_train.sh 是旧版 all-in-one 脚本"
echo -e "   建议改用新流程:  \033[0;36m./start.sh 3\033[0m  (云端训练)"
echo -e "   完整流程入口:    \033[0;36m./start.sh\033[0m  (交互式菜单)"
echo -e "   或单独运行:      \033[0;36mbash scripts/03_autodl_train.sh --robot g1\033[0m"
echo ""

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
log()   { echo -e "${CYAN}[·]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ────────────────────────────────────────────────────────────────────
ROBOT="g1"
NUM_EPOCHS=10
BATCH_SIZE=2
GRAD_ACCUM=2
LEARNING_RATE=1e-4
SKIP_SETUP=false
GR00T_REPO="/workspace/Isaac-GR00T"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/models/${ROBOT}_gr00t"
FULL_FP16_DIR="/workspace/models/${ROBOT}_gr00t_full_fp16"
INT4_DIR="/workspace/models/${ROBOT}_gr00t_int4"
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
MODEL_PACK_NAME="${ROBOT}_gr00t_model.tar.gz"          # 主包: 完整 FP16 全量模型
FULL_FP16_PACK_NAME="${ROBOT}_gr00t_full_fp16.tar.gz"  # 别名: 同上, 方便识别
INT4_PACK_NAME="${ROBOT}_gr00t_int4_model.tar.gz"      # INT4 量化包
MODEL_SIZE="1.7-3B"
EXPORT_INT4=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";          shift 2 ;;
        --epochs)       NUM_EPOCHS="$2";     shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";     shift 2 ;;
        --grad-accum)   GRAD_ACCUM="$2";     shift 2 ;;
        --lr)           LEARNING_RATE="$2";  shift 2 ;;
        --skip-setup)   SKIP_SETUP=true;     shift   ;;
        --data-dir)     DATA_DIR="$2";       shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --model-size)   MODEL_SIZE="$2";     shift 2 ;;
        --no-export-int4) EXPORT_INT4=false; shift   ;;
        -h|--help)
            echo "用法: $0 [--robot g1|go2] [--epochs N] [--batch-size N]"
            echo "      [--model-size 2B|1.7-3B] [--no-export-int4] [--skip-setup]"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

LEROBOT_DATA_DIR="$DATA_DIR/${ROBOT}_lerobot"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 第 2 步: 云端训练          ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "机器人:        $ROBOT"
info "模型:          GR00T-N1-$MODEL_SIZE"
info "Epochs:        $NUM_EPOCHS"
info "Batch Size:    $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
info "Learning Rate: $LEARNING_RATE"
info "导出 INT4:     $EXPORT_INT4"
echo ""

# ── GPU 信息 ────────────────────────────────────────────────────────────────
step "检测 GPU 环境..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    info "GPU: $GPU_NAME ($GPU_MEM)"
else
    fail "未检测到 NVIDIA GPU"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: 环境搭建 (可跳过)
# ════════════════════════════════════════════════════════════════════════════
if ! $SKIP_SETUP; then
    step "Phase 1: 环境搭建"

    # ── 克隆 GR00T ──────────────────────────────────────────────────────
    if [ ! -d "$GR00T_REPO" ]; then
        step "1.1/4: 克隆 Isaac-GR00T..."
        cd "$(dirname "$GR00T_REPO")"
        git clone https://github.com/NVIDIA/Isaac-GR00T.git
        cd "$GR00T_REPO"
        info "Isaac-GR00T 已克隆到 $GR00T_REPO"
    else
        info "Isaac-GR00T 已存在: $GR00T_REPO"
        cd "$GR00T_REPO"
        git pull --ff-only || warn "git pull 失败, 继续使用现有代码"
    fi

    # ── 安装依赖 ────────────────────────────────────────────────────────
    step "1.2/4: 安装 Python 依赖..."
    if command -v uv &>/dev/null; then
        uv sync --python 3.10
        source .venv/bin/activate
    else
        pip install -e .
        pip install bitsandbytes accelerate peft
    fi
    info "依赖安装完成"

    # ── 下载基础模型 ────────────────────────────────────────────────────
    step "1.3/4: 下载 GR00T 基础模型..."
    BASE_MODEL_DIR="/workspace/models/GR00T-N1-${MODEL_SIZE}"
    if [ ! -d "$BASE_MODEL_DIR" ]; then
        mkdir -p "$BASE_MODEL_DIR"
        # 使用 huggingface-cli 下载
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download nvidia/GR00T-N1.${MODEL_SIZE} \
                --local-dir "$BASE_MODEL_DIR"
        else
            pip install -U huggingface_hub
            huggingface-cli download nvidia/GR00T-N1.${MODEL_SIZE} \
                --local-dir "$BASE_MODEL_DIR"
        fi
        info "基础模型已下载到 $BASE_MODEL_DIR"
    else
        info "基础模型已存在: $BASE_MODEL_DIR"
    fi

    # ── 解压训练数据 ────────────────────────────────────────────────────
    step "1.4/4: 解压训练数据..."
    if [ ! -d "$LEROBOT_DATA_DIR" ]; then
        # 查找上传的 tar.gz
        for p in "/root/workspace/$PACK_NAME" "/workspace/$PACK_NAME" "$HOME/$PACK_NAME"; do
            if [ -f "$p" ]; then
                log "找到训练包: $p"
                cd /workspace
                tar -xzf "$p"
                info "训练数据已解压到 $LEROBOT_DATA_DIR"
                break
            fi
        done
        if [ ! -d "$LEROBOT_DATA_DIR" ]; then
            fail "未找到训练数据, 请先上传: $PACK_NAME"
        fi
    else
        info "训练数据已存在: $LEROBOT_DATA_DIR"
    fi
else
    info "跳过环境搭建 (--skip-setup)"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: Fine-tune 训练
# ════════════════════════════════════════════════════════════════════════════
step "Phase 2: Fine-tune 训练..."

cd "$GR00T_REPO"
source .venv/bin/activate 2>/dev/null || true

# 估算 max_steps (官方训练只接 max-steps, 不接 num-epochs)
EPISODE_COUNT=$(find "$LEROBOT_DATA_DIR/data" -name "*.parquet" 2>/dev/null | wc -l)
ASSUMED_FRAMES_PER_EP=50
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
MAX_STEPS=$(( (EPISODE_COUNT * ASSUMED_FRAMES_PER_EP * NUM_EPOCHS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
[ "$MAX_STEPS" -lt 100 ] && MAX_STEPS=100
log "max_steps=$MAX_STEPS  (${EPISODE_COUNT} episodes × ${NUM_EPOCHS} epochs × ~${ASSUMED_FRAMES_PER_EP}f / (${BATCH_SIZE}×${GRAD_ACCUM}))"

python3 gr00t/experiment/launch_finetune.py \
    --base-model-path "/workspace/models/GR00T-N1-${MODEL_SIZE}" \
    --dataset-path "$LEROBOT_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --embodiment-tag "NEW_EMBODIMENT" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --num-gpus 1 \
    --save-only-model

info "✅ Fine-tune 训练完成"

# ════════════════════════════════════════════════════════════════════════════
# Phase 2.5: 合并 LoRA → 完整 FP16 模型 (供高显存 GPU 推理)
# ════════════════════════════════════════════════════════════════════════════
step "Phase 2.5: 合并 LoRA → 完整 FP16 模型..."

SCRIPT_DIR_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
MERGE_SCRIPT="$SCRIPT_DIR_LOCAL/src/merge_lora.py"
if [ ! -f "$MERGE_SCRIPT" ]; then
    MERGE_SCRIPT="/workspace/gr00t_mjlab_autodl/src/merge_lora.py"
fi

if [ -f "$MERGE_SCRIPT" ]; then
    python3 "$MERGE_SCRIPT" \
        --base-model "/workspace/models/GR00T-N1-${MODEL_SIZE}" \
        --lora-path "$OUTPUT_DIR" \
        --output-dir "$FULL_FP16_DIR"
    info "✅ 完整 FP16 模型已生成: $FULL_FP16_DIR"
else
    warn "未找到 merge_lora.py, 跳过 FP16 全量模型生成"
    FULL_FP16_DIR=""
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: INT4 量化导出
# ════════════════════════════════════════════════════════════════════════════
if $EXPORT_INT4; then
    step "Phase 3: INT4 量化导出..."
    INT4_DIR="${OUTPUT_DIR}_int4"

    # 使用本项目的 export_int4.py
    SCRIPT_DIR_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
    EXPORT_SCRIPT="$SCRIPT_DIR_LOCAL/src/export_int4.py"
    if [ ! -f "$EXPORT_SCRIPT" ]; then
        EXPORT_SCRIPT="/workspace/gr00t_mjlab_autodl/src/export_int4.py"
    fi

    if [ -f "$EXPORT_SCRIPT" ]; then
        python3 "$EXPORT_SCRIPT" \
            --model-dir "$OUTPUT_DIR" \
            --output-dir "$INT4_DIR" \
            --model-path "/workspace/models/GR00T-N1-${MODEL_SIZE}"
        info "✅ INT4 量化完成: $INT4_DIR"
    else
        warn "未找到 export_int4.py, 跳过 INT4 导出"
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 4: 打包
# ════════════════════════════════════════════════════════════════════════════
step "Phase 4: 打包模型 (FP16 全量 + INT4 量化)..."

cd /workspace

# ── 打包完整 FP16 全量模型 (适合高显存 GPU, 如 RTX 4090 24GB+) ────────
if [ -n "$FULL_FP16_DIR" ] && [ -d "$FULL_FP16_DIR" ]; then
    tar -czf "$FULL_FP16_PACK_NAME" \
        -C "$(dirname "$FULL_FP16_DIR")" "$(basename "$FULL_FP16_DIR")"
    FP16_SIZE=$(du -h "$FULL_FP16_PACK_NAME" | cut -f1)
    info "✅ FP16 全量模型包: $FULL_FP16_PACK_NAME ($FP16_SIZE)"
    info "   路径: $FULL_FP16_DIR"
    info "   用途: 高显存 GPU 推理 (RTX 4090 24GB+)"

    # 别名: 同时打包一份为 MODEL_PACK_NAME (兼容旧脚本)
    if [ "$FULL_FP16_PACK_NAME" != "$MODEL_PACK_NAME" ]; then
        cp "$FULL_FP16_PACK_NAME" "$MODEL_PACK_NAME"
        info "   别名: $MODEL_PACK_NAME (兼容旧脚本)"
    fi
else
    warn "未生成 FP16 全量模型, 仅打包 LoRA adapter"
    tar -czf "$MODEL_PACK_NAME" \
        -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    LORA_SIZE=$(du -h "$MODEL_PACK_NAME" | cut -f1)
    info "LoRA adapter 包: $MODEL_PACK_NAME ($LORA_SIZE)"
fi

# ── 打包 INT4 量化模型 (适合低显存 GPU, 如 RTX 2080 8GB) ────────────
if $EXPORT_INT4 && [ -d "$INT4_DIR" ]; then
    tar -czf "$INT4_PACK_NAME" \
        -C "$(dirname "$INT4_DIR")" "$(basename "$INT4_DIR")"
    INT4_SIZE=$(du -h "$INT4_PACK_NAME" | cut -f1)
    info "✅ INT4 量化模型包: $INT4_PACK_NAME ($INT4_SIZE)"
    info "   路径: $INT4_DIR"
    info "   用途: 低显存 GPU 推理 (RTX 2080 8GB)"
else
    warn "未生成 INT4 模型, 跳过"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
info "✅ 训练完成! 共生成 $([ -n "$FULL_FP16_DIR" ] && [ -d "$FULL_FP16_DIR" ] && echo -n "2" || echo -n "1")$([ $EXPORT_INT4 = true ] && [ -d "$INT4_DIR" ] && echo "+1" || echo "") 个模型包"
echo ""
echo "下载模型到本地 (推荐一次下载全部):"
echo "  scp -P <port> root@<host>:/workspace/$FULL_FP16_PACK_NAME ./    # FP16 全量 (高显存)"
$([ -d "$INT4_DIR" ] && echo "  scp -P <port> root@<host>:/workspace/$INT4_PACK_NAME ./         # INT4 量化 (低显存)")
echo ""
echo "或使用本项目一键下载脚本 (推荐):"
echo "  ./scripts/03_download_model.sh root@<host> -p <port> --robot $ROBOT"
echo ""
