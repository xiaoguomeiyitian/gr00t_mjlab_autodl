#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab Docker 启动脚本
# 选择对应镜像 + 模式启动容器
#
# 交互式: ./start.sh
# 非交互: ./start.sh collect
#          ./start.sh train --robot g1 --epochs 20
#          ./start.sh infer --robot g1 --model-path /root/models/g1_gr00t_int4
#          ./start.sh shell   # 进入 collect 镜像的 shell
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

IMAGE_PREFIX="gr00t-mjlab"

# 模式 → 镜像映射
declare -A MODE_TO_IMAGE=(
    [collect]="gr00t-mjlab-collect:latest"
    [train]="gr00t-mjlab-train:latest"
    [infer]="gr00t-mjlab-infer:latest"
    [shell]="gr00t-mjlab-collect:latest"
)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 路径解析 ─────────────────────────────────────────────────────────────
PROJECT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
GR00T_REL="$(basename "$PROJECT_ROOT")"

# ── 检查镜像 ─────────────────────────────────────────────────────────────
check_image() {
    local img="$1"
    if ! docker image inspect "$img" &>/dev/null 2>&1; then
        fail "镜像 $img 不存在, 请先运行 ./build.sh 构建"
    fi
}

# ── 检查 GPU ─────────────────────────────────────────────────────────────
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "未检测到 nvidia-smi, 容器将无法使用 GPU"
        return 1
    fi
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        warn "Docker 未配置 nvidia runtime"
        return 1
    fi
    return 0
}

# ── 启动容器 ─────────────────────────────────────────────────────────────
start_container() {
    local mode="$1"; shift
    local image="${MODE_TO_IMAGE[$mode]:-gr00t-mjlab-collect:latest}"

    check_image "$image"

    info "启动镜像: $image"
    info "模式:     $mode"
    info "挂载:     $GR00T_REL -> /root/$GR00T_REL"
    info "数据:     $GR00T_REL/data  (双向同步)"
    info "模型:     $GR00T_REL/models (双向同步)"

    # 默认 GPU 参数
    local gpu_args=()
    if check_gpu; then
        gpu_args+=(--gpus all --network host)
    fi

    # 默认环境变量 (从宿主机透传)
    local env_args=()
    for v in ROBOT NUM_EPISODES NUM_EPOCHS INSTRUCTION MODEL_PATH \
             BATCH_SIZE GRAD_ACCUM LEARNING_RATE DATA_DIR MODEL_SIZE \
             EPISODE_LENGTH EPISODES QUANTIZE TASK_ID; do
        if [ -n "${!v:-}" ]; then
            env_args+=(-e "$v=${!v}")
        fi
    done

    docker run -it --rm \
        "${gpu_args[@]}" \
        -v "$PROJECT_PARENT/$GR00T_REL:/root/$GR00T_REL" \
        -v "$PROJECT_PARENT/$GR00T_REL/data:/root/data" \
        -v "$PROJECT_PARENT/$GR00T_REL/models:/root/models" \
        "${env_args[@]}" \
        "$image" \
        "$mode" "$@"
}

# ── 交互式主流程 ─────────────────────────────────────────────────────────
run_interactive() {
    echo "═══════════════════════════════════════════════════════════════"
    echo "         gr00t-mjlab Docker 启动"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "请选择启动模式:"
    echo "  1) collect  数据采集 (镜像: gr00t-mjlab-collect)"
    echo "  2) train    GR00T 训练 (镜像: gr00t-mjlab-train)"
    echo "  3) infer    推理验证 (镜像: gr00t-mjlab-infer)"
    echo "  4) shell    进入 collect 镜像的 shell"
    echo ""
    read -p "输入选项 [1/2/3/4, 默认 1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1) start_container collect ;;
        2) start_container train ;;
        3) start_container infer ;;
        4) start_container shell ;;
        *) fail "无效选项" ;;
    esac
}

# ── 入口 ─────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    run_interactive
else
    mode="$1"; shift
    case "$mode" in
        collect|train|infer|shell) start_container "$mode" "$@" ;;
        -h|--help)
            echo "用法: $0 <mode> [args...]"
            echo ""
            echo "mode:"
            echo "  collect   数据采集 (进入 collect 镜像)"
            echo "  train     GR00T 训练 (进入 train 镜像)"
            echo "  infer     推理验证 (进入 infer 镜像)"
            echo "  shell     进入 collect 镜像的 shell"
            echo ""
            echo "环境变量 (从宿主机透传到容器):"
            echo "  ROBOT, NUM_EPISODES, NUM_EPOCHS, INSTRUCTION,"
            echo "  MODEL_PATH, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE,"
            echo "  DATA_DIR, MODEL_SIZE, EPISODE_LENGTH, EPISODES, QUANTIZE"
            ;;
        *) fail "未知模式: $mode (可选: collect|train|infer|shell)" ;;
    esac
fi
