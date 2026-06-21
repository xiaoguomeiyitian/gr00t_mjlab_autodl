#!/usr/bin/env bash
# ============================================================================
# 第 4 步: 本地验证微调后的 GR00T 模型
#
# 运行环境: 本地 (RTX 2080 8GB 起)
# 作用: 加载微调后的 GR00T 模型, 在 unitree_rl_mjlab 仿真中运行推理验证
#
# 用法:
#   ./04_local_verify.sh --robot g1
#   ./04_local_verify.sh --robot g1 --model-path models/g1_gr00t_int4
# ============================================================================
set -euo pipefail

# 项目根目录 = 脚本所在目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
# 兄弟目录: unitree_rl_mjlab 官方项目
RL_MJLAB_ROOT="$(cd "$PROJECT_ROOT/../unitree_rl_mjlab" && pwd)"

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ────────────────────────────────────────────────────────────────────
ROBOT="g1"
MODEL_PATH=""
INSTRUCTION="walk forward"
MAX_STEPS=200
SHOW_VIEWER=false
QUANTIZE="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";        shift 2 ;;
        --model-path)   MODEL_PATH="$2";   shift 2 ;;
        --instruction)  INSTRUCTION="$2";  shift 2 ;;
        --max-steps)    MAX_STEPS="$2";    shift 2 ;;
        --quantize)     QUANTIZE="$2";     shift 2 ;;
        --show-viewer)  SHOW_VIEWER=true;  shift   ;;
        -h|--help)
            echo "用法: $0 [--robot g1|go2] [--model-path PATH]"
            echo "      [--instruction 'text'] [--max-steps N]"
            echo "      [--quantize auto|none|4bit|8bit] [--show-viewer]"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── 自动查找模型 ────────────────────────────────────────────────────────────
if [ -z "$MODEL_PATH" ]; then
    # 优先查找顺序: INT4 (低显存) → FP16 全量 (高显存) → LoRA adapter → 其他
    SEARCH_DIRS=(
        "$PROJECT_ROOT/models/${ROBOT}_gr00t_int4"
        "$PROJECT_ROOT/models/${ROBOT}_gr00t_full_fp16"
        "$PROJECT_ROOT/models/${ROBOT}_gr00t"
        "$PROJECT_ROOT/models/${ROBOT}_gr00t_finetuned"
    )
    for d in "${SEARCH_DIRS[@]}"; do
        if [ -d "$d" ] && [ "$(ls -A "$d" 2>/dev/null)" ]; then
            MODEL_PATH="$d"
            break
        fi
    done
    if [ -z "$MODEL_PATH" ]; then
        fail "未找到微调后的模型, 请用 --model-path 指定"
    fi
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 第 4 步: 本地验证推理    ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "机器人:     $ROBOT"
info "模型路径:   $MODEL_PATH"
info "指令:       $INSTRUCTION"
info "最大步数:   $MAX_STEPS"
echo ""

# ── 检查环境 ────────────────────────────────────────────────────────────────
step "检查环境..."
if [ ! -d "$RL_MJLAB_ROOT" ]; then
    warn "未找到 unitree_rl_mjlab: $RL_MJLAB_ROOT"
fi
if ! python3 -c "import mjlab" 2>/dev/null; then
    warn "mjlab 未安装, 将以纯推理模式运行"
fi

# ── 验证模型 ────────────────────────────────────────────────────────────────
step "验证模型文件..."
if [ ! -d "$MODEL_PATH" ]; then
    fail "模型目录不存在: $MODEL_PATH"
fi
MODEL_FILES=$(find "$MODEL_PATH" -type f | wc -l)
if [ "$MODEL_FILES" -eq 0 ]; then
    fail "模型目录为空: $MODEL_PATH"
fi
info "模型文件数: $MODEL_FILES"
find "$MODEL_PATH" -maxdepth 1 -type f -printf "    %f (%s bytes)\n" 2>/dev/null | head -10
echo ""

# ── 自动检测量化 ────────────────────────────────────────────────────────────
if [[ "$MODEL_PATH" == *"int4"* ]] && [ "$QUANTIZE" = "auto" ]; then
    QUANTIZE="4bit"
    info "自动检测: INT4 量化模型"
elif [[ "$MODEL_PATH" == *"int8"* ]] && [ "$QUANTIZE" = "auto" ]; then
    QUANTIZE="8bit"
    info "自动检测: INT8 量化模型"
elif [ "$QUANTIZE" = "auto" ]; then
    QUANTIZE="none"
    info "自动检测: 全精度模型"
fi

# ── 运行推理 ────────────────────────────────────────────────────────────────
step "启动推理..."

INFER_ARGS=(
    "--robot" "$ROBOT"
    "--model-path" "$MODEL_PATH"
    "--instruction" "$INSTRUCTION"
    "--max-steps" "$MAX_STEPS"
    "--quantize" "$QUANTIZE"
)
if $SHOW_VIEWER; then
    INFER_ARGS+=("--show-viewer")
fi

python3 "$SRC_DIR/infer.py" "${INFER_ARGS[@]}"

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🎉 第 4 步完成!${NC}"
echo ""
echo "🎊 整个 GR00T 训练流程已完成!"
echo ""
echo "下一步建议:"
echo "  1. 调整 --instruction 测试不同任务"
echo "  2. 收集更多 episodes 重新训练 (--episodes 500+)"
echo "  3. 部署到真实机器人 (需要 cyclonedds + unitree_sdk2)"
echo ""
