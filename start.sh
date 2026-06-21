#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab 统一入口 — 选择并执行任意一个步骤
#
# 完整 6 步流程 (与 ./scripts/0N_*.sh 一一对应):
#   [1] 数据采集   → 启动 collect 镜像, 执行 01_local_collect.sh
#   [2] 上传云端   → 主机 shell, 执行 02_upload_to_autodl.sh
#   [3] 云端训练   → 提示用户 SSH 到云端执行 03_autodl_train.sh
#   [4] 下载模型   → 主机 shell, 执行 04_download_model.sh
#   [5] 本地量化   → 主机 shell, 执行 05_local_quantize.sh
#   [6] 推理验证   → 启动 infer 镜像, 执行 06_local_verify.sh
#
# 用法:
#   ./start.sh                       # 交互式选择步骤
#   ./start.sh 1                     # 执行步骤 1
#   ./start.sh 3                     # 执行步骤 3
#   ./start.sh 1 --robot g1          # 步骤 1 透传额外参数
#   ./start.sh collect               # 旧名: 等价于 步骤 1
#   ./start.sh infer                 # 旧名: 等价于 步骤 6
#   ./start.sh shell                 # 进入 collect 镜像的 shell
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

IMAGE_PREFIX="gr00t-mjlab"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 步骤定义 (单一事实来源) ─────────────────────────────────────────────
# 字段: 编号 | 名称 | 运行环境 | 调用的脚本 | 调用的 Docker 镜像
declare -A STEP_ENV=(
    [1]="docker"   # 容器内跑
    [2]="host"     # 主机直接跑
    [3]="manual"   # 需手动去云端
    [4]="host"     # 主机直接跑
    [5]="host"     # 主机直接跑
    [6]="docker"   # 容器内跑
)
declare -A STEP_SCRIPT=(
    [1]="01_local_collect.sh"
    [2]="02_upload_to_autodl.sh"
    [3]="03_autodl_train.sh"
    [4]="04_download_model.sh"
    [5]="05_local_quantize.sh"
    [6]="06_local_verify.sh"
)
declare -A STEP_IMAGE=(
    [1]="gr00t-mjlab-collect:latest"
    [6]="gr00t-mjlab-infer:latest"
)
declare -A STEP_NAME=(
    [1]="数据采集"
    [2]="上传云端"
    [3]="云端训练"
    [4]="下载模型"
    [5]="本地量化"
    [6]="推理验证"
)
declare -A STEP_DESC=(
    [1]="本地 mjlab 收集 episodes 并打包为训练包"
    [2]="SCP 上传训练包和训练脚本到 AutoDL"
    [3]="AutoDL 云端解压 + Fine-tune + 打包 (FP16/INT4)"
    [4]="SCP 下载训练好的模型包到本地"
    [5]="FP16 → INT4 PTQ 量化 (8GB 显存友好)"
    [6]="加载微调后的 GR00T 模型, 在 mjlab 中推理验证"
)

step_name()  { echo "${STEP_NAME[$1]:-未知步骤}"; }
step_desc()  { echo "${STEP_DESC[$1]:-}"; }
step_env()   { echo "${STEP_ENV[$1]:-}"; }
step_script(){ echo "${STEP_SCRIPT[$1]:-}"; }
step_image() { echo "${STEP_IMAGE[$1]:-}"; }

# ── 路径解析 ─────────────────────────────────────────────────────────────
PROJECT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
GR00T_REL="$(basename "$PROJECT_ROOT")"

# ── 工具函数 ─────────────────────────────────────────────────────────────
check_image() {
    local img="$1"
    if ! docker image inspect "$img" &>/dev/null 2>&1; then
        fail "镜像 $img 不存在, 请先运行: ./build.sh collect  (或 ./build.sh all)"
    fi
}

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

# ── 启动 Docker 容器执行某步骤 ──────────────────────────────────────────
# 用法: run_docker_step <step_num> [args...]
run_docker_step() {
    local step="$1"; shift
    local image="$(step_image "$step")"
    local script="$(step_script "$step")"
    local script_in_container="/root/$GR00T_REL/scripts/$script"

    check_image "$image"

    info "启动镜像:   $image"
    info "步骤:       [${step}] $(step_name "$step")"
    info "执行脚本:   $script"
    info "挂载:       $GR00T_REL -> /root/$GR00T_REL"
    info "数据:       $GR00T_REL/data  (双向同步)"
    info "模型:       $GR00T_REL/models (双向同步)"
    echo ""

    # GPU 参数
    local gpu_args=()
    if check_gpu; then
        gpu_args+=(--gpus all --network host)
    fi

    # 环境变量透传 (宿主机 → 容器)
    local env_args=()
    for v in ROBOT NUM_EPISODES NUM_EPOCHS INSTRUCTION MODEL_PATH \
             BATCH_SIZE GRAD_ACCUM LEARNING_RATE DATA_DIR MODEL_SIZE \
             EPISODE_LENGTH EPISODES QUANTIZE TASK_ID; do
        if [ -n "${!v:-}" ]; then
            env_args+=(-e "$v=${!v}")
        fi
    done

    # 在容器内执行: 先 source ssh config, 再跑脚本
    # 用 entrypoint 的 shell 模式 + 注入命令
    docker run -it --rm \
        "${gpu_args[@]}" \
        -v "$PROJECT_PARENT/$GR00T_REL:/root/$GR00T_REL" \
        -v "$PROJECT_PARENT/$GR00T_REL/data:/root/data" \
        -v "$PROJECT_PARENT/$GR00T_REL/models:/root/models" \
        "${env_args[@]}" \
        "$image" \
        bash -c "bash $script_in_container $*"
}

# ── 主机直接执行某步骤 ──────────────────────────────────────────────────
# 用法: run_host_step <step_num> [args...]
run_host_step() {
    local step="$1"; shift
    local script="$(step_script "$step")"
    local script_path="$SCRIPTS_DIR/$script"

    if [ ! -f "$script_path" ]; then
        fail "脚本不存在: $script_path"
    fi

    info "环境:       主机 (host shell)"
    info "步骤:       [${step}] $(step_name "$step")"
    info "执行脚本:   scripts/$script"
    echo ""

    bash "$script_path" "$@"
}

# ── 云端手动步骤 ────────────────────────────────────────────────────────
# 用法: run_manual_step
run_manual_step() {
    local step="$1"
    info "环境:       AutoDL 云端 (需 SSH 登录)"
    info "步骤:       [${step}] $(step_name "$step")"
    info "说明:       训练在云端进行, 本机无法直接执行"
    echo ""

    # 加载 SSH 配置 (用户已填 _ssh_config.sh)
    if [ -f "$SCRIPTS_DIR/_ssh_config.sh" ]; then
        # shellcheck disable=SC1091
        source "$SCRIPTS_DIR/_ssh_config.sh" 2>/dev/null || true
    fi

    if [ -z "${SSH_HOST:-}" ] || [ "${SSH_HOST}" = "root@your-host.com" ]; then
        echo -e "${YELLOW}⚠️  请先在 scripts/_ssh_config.sh 填写 SSH 信息${NC}"
        echo ""
        echo "  或手动登录 AutoDL 后执行:"
        echo "    cd /root/workspace"
        echo "    bash 03_autodl_train.sh --robot g1 --epochs 10"
        echo ""
        return 0
    fi

    # SSH 信息归一化
    if [[ "$SSH_HOST" != *@* ]]; then
        SSH_HOST="${SSH_USER:-root}@${SSH_HOST}"
    fi
    SSH_PORT="${SSH_PORT:-22}"

    echo "将连接到: $SSH_HOST:$SSH_PORT"
    echo ""
    echo -e "${BOLD}云端训练命令 (可在本脚本中自动执行, 或复制到 ssh 会话):${NC}"
    echo ""
    echo "  cd /root/workspace"
    echo "  bash 03_autodl_train.sh --robot \${ROBOT:-g1} --epochs 10"
    echo ""
    read -p "是否现在 SSH 登录并执行训练? [y/N, 默认 N]: " choice
    case "${choice:-N}" in
        [yY]|[yY][eE][sS])
            SSH_BASE_ARGS=(-p "$SSH_PORT" $SSH_OPTS)
            [ -n "${SSH_KEY:-}" ] && SSH_BASE_ARGS+=(-i "$SSH_KEY")
            ssh -t "${SSH_BASE_ARGS[@]}" "$SSH_HOST" \
                "cd /root/workspace && bash 03_autodl_train.sh --robot \${ROBOT:-g1} --epochs 10"
            info "✅ 训练命令已发送到云端"
            ;;
        *)
            echo ""
            echo "已跳过. 稍后可手动执行:"
            echo "  ssh -p $SSH_PORT $SSH_HOST"
            echo "  cd /root/workspace && bash 03_autodl_train.sh"
            ;;
    esac
}

# ── 调度入口 ────────────────────────────────────────────────────────────
# 用法: run_step <step_num> [args...]
run_step() {
    local step="$1"; shift
    local env="$(step_env "$step")"

    case "$env" in
        docker) run_docker_step "$step" "$@" ;;
        host)   run_host_step   "$step" "$@" ;;
        manual) run_manual_step "$step" ;;
        *)      fail "步骤 [$step] 的环境配置错误: '$env'" ;;
    esac
}

# ── 打印步骤列表 (用于交互菜单 / help) ──────────────────────────────────
print_step_list() {
    for n in 1 2 3 4 5 6; do
        local env_label
        case "$(step_env "$n")" in
            docker) env_label="🐳 docker  ";;
            host)   env_label="💻 host    ";;
            manual) env_label="☁️  cloud   ";;
        esac
        printf "  ${BOLD}[%s]${NC} ${CYAN}%-8s${NC} %-12s  %s\n" "$n" "$(step_name "$n")" "$env_label" "$(step_desc "$n")"
    done
}

# ── 交互式主流程 ─────────────────────────────────────────────────────────
run_interactive() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "         gr00t-mjlab — 选择并执行一个步骤"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    print_step_list
    echo ""
    # 用 printf (Bash 内建会解释 \0XX 八进制转义), 不要用 echo (默认不解释)
    printf "  ${BOLD}[s]${NC} ${CYAN}%-7s${NC} 💻 docker    %s\n" "shell" "进入 collect 镜像的 shell (调试用)"
    printf "  ${BOLD}[q]${NC} ${CYAN}%-7s${NC}              %s\n" "quit"  "退出"
    echo ""
    read -p "输入选项 [1-6 / s / q, 默认 1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1|2|3|4|5|6) run_step "$choice" ;;
        s|S)         start_container_shell ;;
        q|Q)         echo "已退出"; exit 0 ;;
        *)           fail "无效选项: $choice" ;;
    esac
}

# ── 旧的"直接进入镜像"入口 (shell 调试用) ──────────────────────────────
start_container_shell() {
    local image="gr00t-mjlab-collect:latest"
    check_image "$image"

    info "启动镜像: $image (shell 模式)"

    local gpu_args=()
    if check_gpu; then gpu_args+=(--gpus all --network host); fi

    docker run -it --rm \
        "${gpu_args[@]}" \
        -v "$PROJECT_PARENT/$GR00T_REL:/root/$GR00T_REL" \
        -v "$PROJECT_PARENT/$GR00T_REL/data:/root/data" \
        -v "$PROJECT_PARENT/$GR00T_REL/models:/root/models" \
        "$image"
}

# ── 入口 ─────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    run_interactive
else
    first="$1"; shift || true

    # 数字步骤
    if [[ "$first" =~ ^[1-6]$ ]]; then
        run_step "$first" "$@"
        exit $?
    fi

    # 旧名兼容 (collect / infer / shell)
    case "$first" in
        collect)
            run_step 1 "$@"
            ;;
        infer)
            run_step 6 "$@"
            ;;
        shell)
            start_container_shell
            ;;
        -h|--help)
            echo "用法: $0 [step|alias] [args...]"
            echo ""
            echo "步骤 (推荐):"
            print_step_list
            echo ""
            echo "别名 (旧名兼容):"
            echo "  collect   = 步骤 [1] 数据采集"
            echo "  infer     = 步骤 [6] 推理验证"
            echo "  shell     = 进入 collect 镜像的交互 shell (调试用)"
            echo ""
            echo "示例:"
            echo "  $0                       # 交互式菜单"
            echo "  $0 1                     # 步骤 1: 数据采集"
            echo "  $0 1 --robot g1          # 步骤 1 + 透传 --robot g1"
            echo "  $0 3                     # 步骤 3: 云端训练 (提示 SSH)"
            echo "  $0 4 --with-int4         # 步骤 4: 下载 FP16 + INT4"
            echo "  $0 5 --robot g1          # 步骤 5: 本地 INT4 量化"
            echo "  $0 6 --auto-quantize     # 步骤 6: 推理 (自动量化)"
            echo ""
            echo "环境说明:"
            echo "  🐳 docker  = 在对应镜像内执行 (需 ./build.sh 构建)"
            echo "  💻 host    = 在主机 shell 直接跑 (无需 Docker)"
            echo "  ☁️  cloud   = 在 AutoDL 云端执行 (本机提示 SSH)"
            ;;
        *)
            fail "未知步骤/别名: $first (可选: 1-6 / collect / infer / shell)"
            ;;
    esac
fi
