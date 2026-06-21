#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab 统一入口 — 选择并执行任意一个步骤 (Native / 本机模式)
#
# 完整 6 步流程 (与 ./scripts/0N_*.sh 一一对应):
#   [1] 数据采集   → 本机 mjlab, 执行 01_local_collect.sh
#   [2] 上传云端   → 本机 shell, 执行 02_upload_to_autodl.sh
#   [3] 云端训练   → 提示用户 SSH 到云端执行 03_autodl_train.sh
#   [4] 下载模型   → 本机 shell, 执行 04_download_model.sh
#   [5] 本地量化   → 本机 shell, 执行 05_local_quantize.sh
#   [6] 推理验证   → 本机 .venv, 执行 06_local_verify.sh
#
# 运行模式: 全部在主机执行 (本机 venv)
#   - 步骤 1/6 使用项目下的 .venv (含 mjlab / Isaac-GR00T)
#   - 步骤 2/4/5 是纯 shell / SCP / Python, 不需要 venv
#   - 步骤 3 提示 SSH 到云端
#   前置: ./install_native.sh 已成功运行
#
# 用法:
#   ./start.sh                       # 交互式选择步骤
#   ./start.sh 1                     # 执行步骤 1 (数据采集)
#   ./start.sh 1 --robot g1          # 步骤 1 透传额外参数
#   ./start.sh 6 --auto-quantize     # 步骤 6 推理 (自动量化)
#   ./start.sh shell                 # 进入本机 venv shell (激活环境, 调试用)
#   ./start.sh collect               # 旧名: 等价于 步骤 1
#   ./start.sh infer                 # 旧名: 等价于 步骤 6
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
VENV_DIR="$PROJECT_ROOT/.venv"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 步骤定义 (单一事实来源) ─────────────────────────────────────────────
# 字段: 编号 | 名称 | 运行环境 (native/host/manual) | 调用的脚本
declare -A STEP_ENV=(
    [1]="native"  # 本机 .venv (含 mjlab)
    [2]="host"    # 本机 shell (scp)
    [3]="manual"  # 需手动去云端
    [4]="host"    # 本机 shell (scp)
    [5]="host"    # 本机 shell (Python)
    [6]="native"  # 本机 .venv (含 mjlab + GR00T)
)
declare -A STEP_SCRIPT=(
    [1]="01_local_collect.sh"
    [2]="02_upload_to_autodl.sh"
    [3]="03_autodl_train.sh"
    [4]="04_download_model.sh"
    [5]="05_local_quantize.sh"
    [6]="06_local_verify.sh"
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

# ── 路径解析 ─────────────────────────────────────────────────────────────
PROJECT_PARENT="$(cd "$PROJECT_ROOT/.." && pwd)"
GR00T_REL="$(basename "$PROJECT_ROOT")"

# ── Native 步骤: 在主机 .venv 中执行 ────────────────────────────────────
# 用法: run_native_step <step_num> [args...]
#
# 前置: ./install_native.sh 已成功运行
run_native_step() {
    local step="$1"; shift
    local script="$(step_script "$step")"
    local script_path="$SCRIPTS_DIR/$script"

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        fail "未找到 $VENV_DIR, 请先运行: ./install_native.sh (默认 infer)"
    fi
    if [ ! -f "$script_path" ]; then
        fail "脚本不存在: $script_path"
    fi

    info "环境:       本机 (host venv)"
    info "venv:       $VENV_DIR"
    info "步骤:       [${step}] $(step_name "$step")"
    info "执行脚本:   scripts/$script"
    info "数据:       $GR00T_REL/data"
    info "模型:       $GR00T_REL/models"
    echo ""

    # ── 设置 host 环境变量 ────────────────────────────────────────────
    export GR00T_MJLAB_ROOT="$PROJECT_ROOT"
    export UNITREE_ROOT="$PROJECT_PARENT"
    export DATA_DIR="$PROJECT_ROOT/data"
    export MODELS_DIR="$PROJECT_ROOT/models"
    export PYTHONUNBUFFERED=1

    # Isaac-GR00T 源码路径 (PYTHONPATH 注入)
    local isaac_gr00t_dir="$PROJECT_PARENT/Isaac-GR00T"
    if [ -d "$isaac_gr00t_dir" ]; then
        export PYTHONPATH="$isaac_gr00t_dir${PYTHONPATH:+:$PYTHONPATH}"
        info "GR00T:      $isaac_gr00t_dir"
    fi

    # NVIDIA 动态库路径 (cu128 wheel 装在 .venv/lib/.../nvidia/*/lib)
    local nvidia_libs
    nvidia_libs="$(find "$VENV_DIR/lib" -maxdepth 6 -type d -name 'lib' -path '*/nvidia/*' 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
    if [ -n "$nvidia_libs" ]; then
        export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    # 激活 venv (子 shell 内 export PATH, 不修改用户 shell rc)
    export PATH="$VENV_DIR/bin:$PATH"
    export VIRTUAL_ENV="$VENV_DIR"

    # 用 venv python 做健康检查
    "$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用, 请检查驱动'" \
        || warn "torch.cuda.is_available() = False (继续尝试执行)"

    bash "$script_path" "$@"
}

# ── Native shell: 进入 host venv shell (调试用) ──────────────────────
start_native_shell() {
    if [ ! ! -f "$VENV_DIR/bin/python" ]; then
        fail "未找到 $VENV_DIR, 请先运行: ./install_native.sh"
    fi
    info "进入本机 venv shell (venv 已激活, 输入 'exit' 返回)"
    info "  venv: $VENV_DIR"
    info "  提示: 可直接运行 python, mjlab, Isaac-GR00T"
    echo ""

    # 设置环境变量
    export GR00T_MJLAB_ROOT="$PROJECT_ROOT"
    export UNITREE_ROOT="$PROJECT_PARENT"
    export DATA_DIR="$PROJECT_ROOT/data"
    export MODELS_DIR="$PROJECT_ROOT/models"
    export PYTHONUNBUFFERED=1

    local isaac_gr00t_dir="$PROJECT_PARENT/Isaac-GR00T"
    if [ -d "$isaac_gr00t_dir" ]; then
        export PYTHONPATH="$isaac_gr00t_dir${PYTHONPATH:+:$PYTHONPATH}"
    fi

    local nvidia_libs
    nvidia_libs="$(find "$VENV_DIR/lib" -maxdepth 6 -type d -name 'lib' -path '*/nvidia/*' 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
    if [ -n "$nvidia_libs" ]; then
        export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    export PATH="$VENV_DIR/bin:$PATH"
    export VIRTUAL_ENV="$VENV_DIR"

    exec "$VENV_DIR/bin/bash"
}

# ── 主机直接执行某步骤 (纯 shell / SCP) ──────────────────────────────────
# 用法: run_host_step <step_num> [args...]
run_host_step() {
    local step="$1"; shift
    local script="$(step_script "$step")"
    local script_path="$SCRIPTS_DIR/$script"

    if [ ! -f "$script_path" ]; then
        fail "脚本不存在: $script_path"
    fi

    info "环境:       本机 (host shell)"
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
#
# 路由规则 (按 STEP_ENV):
#   native  → run_native_step  (本机 venv, 步骤 1/6)
#   host    → run_host_step    (本机 shell, 步骤 2/4/5)
#   manual  → run_manual_step  (云端 SSH 提示, 步骤 3)
run_step() {
    local step="$1"; shift
    local env="$(step_env "$step")"

    case "$env" in
        native) run_native_step  "$step" "$@" ;;
        host)   run_host_step    "$step" "$@" ;;
        manual) run_manual_step  "$step" ;;
        *)      fail "步骤 [$step] 的环境配置错误: '$env'" ;;
    esac
}

# ── 打印步骤列表 (用于交互菜单 / help) ──────────────────────────────────
print_step_list() {
    for n in 1 2 3 4 5 6; do
        local env_label
        case "$(step_env "$n")" in
            native) env_label="💻 native  ";;
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
    echo "      gr00t-mjlab — 选择并执行一个步骤 (本机模式)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    print_step_list
    echo ""
    printf "  ${BOLD}[s]${NC} ${CYAN}%-7s${NC} 💻 native    %s\n" "shell" "进入本机 venv shell (激活环境, 调试用)"
    printf "  ${BOLD}[q]${NC} ${CYAN}%-7s${NC}              %s\n" "quit"  "退出"
    echo ""
    read -p "输入选项 [1-6 / s / q, 默认 1]: " choice
    choice="${choice:-1}"

    case "$choice" in
        1)
            # [1] 数据采集 — 交互式选择回合数 + 是否启用 viser
            step_args="$(prompt_step1_wizard)"
            # shellcheck disable=SC2086
            run_step 1 $step_args
            ;;
        2|3|4|5|6) run_step "$choice" ;;
        s|S)       start_native_shell ;;
        q|Q)       echo "已退出"; exit 0 ;;
        *)         fail "无效选项: $choice" ;;
    esac
}

# ── [1] 数据采集交互式 wizard ───────────────────────────────────────────
# UI 全部输出到 stderr (避免被命令替换捕获)
# stdout 只返回要传给 run_step 的额外参数 (如 "--episodes 200 --viser")
prompt_step1_wizard() {
    local eps viser_flag ep_choice viser_choice

    echo "" >&2
    echo "───────────────────────────────────────────────────" >&2
    echo "  [1] 数据采集 — 选择参数" >&2
    echo "───────────────────────────────────────────────────" >&2
    echo "" >&2
    echo "  选择要收集的回合数:" >&2
    echo "    1) 10    (smoke test, ~30s)" >&2
    echo "    2) 50    (快速验证, ~3min)" >&2
    echo "    3) 100   (官方推荐最小, ~6min)" >&2
    echo "    4) 200   (推荐, ~12min)" >&2
    echo "    5) 500   (充分训练, ~30min)" >&2
    echo "    6) 1000  (大规模, ~1h)" >&2
    echo "    7) 自定义" >&2
    echo "" >&2
    read -p "  回合数 [1-7, 默认 3 (100)]: " ep_choice >&2
    case "${ep_choice:-3}" in
        1) eps=10 ;;
        2) eps=50 ;;
        3|"") eps=100 ;;
        4) eps=200 ;;
        5) eps=500 ;;
        6) eps=1000 ;;
        7)
            read -p "  请输入回合数 [1-10000]: " eps >&2
            if ! [[ "$eps" =~ ^[0-9]+$ ]] || [ "$eps" -lt 1 ] || [ "$eps" -gt 10000 ]; then
                warn "无效回合数 '$eps', 回退到 100" >&2
                eps=100
            fi
            ;;
        *)
            warn "无效选项 '$ep_choice', 回退到 100" >&2
            eps=100
            ;;
    esac

    echo "" >&2
    read -p "  启用 viser 浏览器查看? (浏览器打开 http://localhost:8080) [y/N, 默认 N]: " viser_choice >&2
    case "${viser_choice:-N}" in
        [yY]|[yY][eE][sS])
            viser_flag="--viser"
            echo "  ✅ viser 将启动" >&2
            ;;
        *)
            viser_flag=""
            echo "  ⏭️  跳过 viser" >&2
            ;;
    esac

    echo "" >&2
    info "准备启动: 1 回合数=$eps  $viser_flag" >&2
    echo "" >&2

    # 只返回参数到 stdout (命令替换会捕获这里)
    echo "--episodes $eps $viser_flag"
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
            start_native_shell
            ;;
        -h|--help)
            echo "用法: $0 [step|alias] [args...]"
            echo ""
            echo "本机模式: 全部在主机 .venv / shell 中执行"
            echo "前置: ./install_native.sh 已成功运行"
            echo ""
            echo "步骤 (推荐):"
            print_step_list
            echo ""
            echo "别名 (旧名兼容):"
            echo "  collect   = 步骤 [1] 数据采集"
            echo "  infer     = 步骤 [6] 推理验证"
            echo "  shell     = 进入本机 venv shell"
            echo ""
            echo "示例:"
            echo "  $0                       # 交互式菜单"
            echo "  $0 1                     # 步骤 1: 数据采集"
            echo "  $0 1 --robot g1          # 步骤 1 + 透传 --robot g1"
            echo "  $0 3                     # 步骤 3: 云端训练 (提示 SSH)"
            echo "  $0 4 --with-int4         # 步骤 4: 下载 FP16 + INT4"
            echo "  $0 5 --robot g1          # 步骤 5: 本机 INT4 量化"
            echo "  $0 6 --auto-quantize     # 步骤 6: 推理 (自动量化)"
            echo ""
            echo "环境说明:"
            echo "  💻 native  = 在本机 .venv 执行 (需 ./install_native.sh)"
            echo "  💻 host    = 在本机 shell 直接跑 (无需 venv)"
            echo "  ☁️  cloud   = 在 AutoDL 云端执行 (本机提示 SSH)"
            ;;
        *)
            fail "未知步骤/别名: $first (可选: 1-6 / collect / infer / shell)"
            ;;
    esac
fi
