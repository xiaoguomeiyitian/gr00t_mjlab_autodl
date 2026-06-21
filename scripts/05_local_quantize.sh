#!/usr/bin/env bash
# ============================================================================
# 第 5 步 (可选): 本地 INT4 量化 — 将 FP16 全量模型转换为 INT4 量化版
#
# 运行环境: 本地 (8GB+ 显存即可)
# 作用: 纯后训练量化 (PTQ), 不需要重新训练, 5-15 分钟完成
#       将 FP16 全量 (~7GB) → INT4 量化 (~1.5GB), 适合 8GB 显存推理
#
# 前置条件:
#   - 已通过 ./scripts/03_download_model.sh 下载 FP16 全量模型
#   - 已安装: pip install transformers peft bitsandbytes accelerate
#
# 用法:
#   # 默认: 从 models/{robot}_gr00t_full_fp16/ → models/{robot}_gr00t_int4/
#   ./scripts/05_local_quantize.sh --robot g1
#
#   # 显式指定路径
#   ./scripts/05_local_quantize.sh \
#       --input models/g1_gr00t_full_fp16 \
#       --output models/g1_gr00t_int4
#
#   # 离线模式 (无 HF 在线下载)
#   ./scripts/05_local_quantize.sh --robot g1 --offline
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

# ── 颜色 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ──────────────────────────────────────────────────────────────
ROBOT="g1"
INPUT_DIR=""
OUTPUT_DIR=""
OFFLINE=false
DEVICE_MAP="auto"
BASE_MODEL_PATH=""   # 可选: 指定本地基础模型路径

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";        shift 2 ;;
        --input)        INPUT_DIR="$2";    shift 2 ;;
        --output)       OUTPUT_DIR="$2";   shift 2 ;;
        --offline)      OFFLINE=true;      shift   ;;
        --device-map)   DEVICE_MAP="$2";   shift 2 ;;
        --base-model)   BASE_MODEL_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "用法: $0 [--robot g1|go2] [--input PATH] [--output PATH]"
            echo "      [--offline] [--device-map auto|cpu|cuda] [--base-model PATH]"
            echo ""
            echo "说明:"
            echo "  --input  默认: models/\${ROBOT}_gr00t_full_fp16"
            echo "  --output 默认: models/\${ROBOT}_gr00t_int4"
            echo "  --offline 禁用 HF 在线下载 (推荐, 避免触发 HF 限流)"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── 默认路径 ──────────────────────────────────────────────────────────
[ -z "$INPUT_DIR" ]  && INPUT_DIR="$PROJECT_ROOT/models/${ROBOT}_gr00t_full_fp16"
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$PROJECT_ROOT/models/${ROBOT}_gr00t_int4"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 本地 INT4 量化 (PTQ)   ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "机器人:   $ROBOT"
info "输入:     $INPUT_DIR  (FP16 全量)"
info "输出:     $OUTPUT_DIR  (INT4 量化)"
info "设备:     $DEVICE_MAP"
[ "$OFFLINE" = true ] && info "离线:     启用 (无 HF 下载)"
echo ""

# ── 检查输入 ──────────────────────────────────────────────────────────
step "检查输入模型..."
if [ ! -d "$INPUT_DIR" ]; then
    fail "FP16 模型目录不存在: $INPUT_DIR
请先运行: ./scripts/03_download_model.sh root@host -p PORT --robot $ROBOT --with-fp16"
fi
# 检查必要文件
for f in config.json tokenizer_config.json; do
    if [ ! -f "$INPUT_DIR/$f" ]; then
        # fallback: 任一 tokenizer 文件存在即可
        if ! ls "$INPUT_DIR"/tokenizer*.json "$INPUT_DIR"/*.safetensors &>/dev/null; then
            fail "输入目录缺少必要文件 ($f): $INPUT_DIR"
        fi
    fi
done
INPUT_SIZE=$(du -sh "$INPUT_DIR" 2>/dev/null | cut -f1)
info "FP16 输入: $INPUT_SIZE"
info "✓ 输入检查通过"

# ── 检查输出 ──────────────────────────────────────────────────────────
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    warn "输出目录已存在: $OUTPUT_DIR"
    read -p "  覆盖? [y/N]: " ans
    case "${ans:-N}" in
        [yY]|[yY][eE][sS])
            step "删除旧目录..."
            rm -rf "$OUTPUT_DIR"
            ;;
        *) fail "已取消" ;;
    esac
fi

# ── 检查依赖 ──────────────────────────────────────────────────────────
step "检查 Python 依赖..."
python3 -c "import torch, transformers, bitsandbytes" 2>/dev/null \
    || fail "缺少依赖, 请先安装: pip install transformers peft bitsandbytes accelerate"

# 检查 GPU
if [ "$DEVICE_MAP" != "cpu" ]; then
    if ! command -v nvidia-smi &>/dev/null; then
        warn "未检测到 nvidia-smi, 改用 CPU (会很慢)"
        DEVICE_MAP="cpu"
    else
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        info "GPU: $GPU_NAME ($GPU_MEM)"
    fi
fi
info "✓ 环境检查通过"
echo ""

# ── 执行量化 ──────────────────────────────────────────────────────────
step "开始 INT4 量化 (PTQ, 预计 5-15 分钟)..."

# 构造 export_int4.py 参数
QUANT_ARGS=(
    --input-type fp16
    --model-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --device-map "$DEVICE_MAP"
)
[ "$OFFLINE" = true ] && QUANT_ARGS+=(--offline)
[ -n "$BASE_MODEL_PATH" ] && QUANT_ARGS+=(--model-path "$BASE_MODEL_PATH")

python3 "$SRC_DIR/export_int4.py" "${QUANT_ARGS[@]}"

# ── 验证输出 ──────────────────────────────────────────────────────────
echo ""
step "验证输出..."
if [ ! -d "$OUTPUT_DIR" ]; then
    fail "量化失败: 输出目录未创建"
fi
if [ ! -f "$OUTPUT_DIR/config.json" ]; then
    fail "量化失败: 缺少 config.json"
fi
OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
info "INT4 输出: $OUTPUT_SIZE ($FILE_COUNT files)"

# ── 对比摘要 ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}${BOLD}  ✅ INT4 量化完成                                              ${NC}${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "FP16 全量: $INPUT_DIR  ($INPUT_SIZE)"
info "INT4 量化: $OUTPUT_DIR  ($OUTPUT_SIZE)"
echo ""
info "📌 下一步推理:"
echo "  ./scripts/04_local_verify.sh --robot $ROBOT \\"
echo "      --model-path models/${ROBOT}_gr00t_int4 \\"
echo "      --quantize 4bit"