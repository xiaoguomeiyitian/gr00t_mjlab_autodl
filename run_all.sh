#!/usr/bin/env bash
# ============================================================================
# GR00T × unitree_rl_mjlab × AutoDL — 一键全流程脚本 (推荐)
#
# 覆盖完整训练流程:
#   [1] 数据采集     — 本地 unitree_rl_mjlab 收集 episodes 并打包
#   [2] 云端训练     — AutoDL 上 LoRA 微调 + INT4 量化
#   [3] 模型下载     — SCP 下载 FP16 全量 + INT4 量化 包
#   [4] 本地推理     — 加载模型并验证推理输出
#
# 用法:
#   ./run_all.sh                                    # 交互式选择步骤
#   ./run_all.sh --step all                         # 完整 4 步
#   ./run_all.sh --step 1                           # 仅数据采集
#   ./run_all.sh --step 1,4                         # 仅本地 (采集+验证, 不上云)
#   ./run_all.sh --step 2,3                         # 仅云端 (训练+下载)
#   ./run_all.sh --step 1-3                         # 范围: 1,2,3
#   ./run_all.sh --step 1,3,4                       # 自定义组合
#   ./run_all.sh --robot g1 --episodes 200          # 自定义参数
#
# --step 支持的写法:
#   all / full       → 1,2,3,4
#   local            → 1,4    (跳过云端, 纯本地)
#   cloud            → 2,3    (仅云端: 训练+下载)
#   1 | 2 | 3 | 4    → 单步
#   1,2,3            → 逗号分隔的若干步
#   1-3              → 范围展开
#   1,3-4            → 混合写法
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"  # 项目根 = 脚本所在目录
SRC_DIR="$PROJECT_ROOT/src"

# ── 显示用的相对路径: 以 gr00t_mjlab_autodl/ 的父目录为基准
#     (该父目录下 gr00t_mjlab_autodl/ 与 unitree_rl_mjlab/ 是兄弟)
#     如: /home/.../unitree/gr00t_mjlab_autodl  →  gr00t_mjlab_autodl
PROJECT_PARENT="$(cd "$SCRIPT_DIR/.." && pwd)"
relpath() {
    local target="$1"
    local abs
    if abs="$(cd "$target" 2>/dev/null && pwd)"; then
        case "$abs" in
            "$PROJECT_PARENT"/*) echo "${abs#$PROJECT_PARENT/}"; return ;;
        esac
        echo "$abs"
    else
        echo "$target"
    fi
}

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 步骤名称定义 (单一事实来源, 所有 UI 文本都从这里取) ──────────────────────
declare -A STEP_NAMES=(
    [1]="数据采集"
    [2]="云端训练"
    [3]="模型下载"
    [4]="本地推理"
)
declare -A STEP_DESCS=(
    [1]="本地 mjlab 收集 episodes 并打包为训练包"
    [2]="AutoDL 上 LoRA 微调 + INT4 量化"
    [3]="SCP 下载 FP16 全量 + INT4 量化模型包"
    [4]="加载模型 + 验证推理输出"
)

# 取第 N 步的中文名 (如: step_label 1 → "数据采集")
step_label()  { echo "${STEP_NAMES[$1]:-未知步骤}"; }
# 取第 N 步的详细描述
step_desc()   { echo "${STEP_DESCS[$1]:-}"; }
# 取第 N 步的完整标题 (如: step_full 1 → "第 1 步: 数据采集")
step_full()   { echo "第 $1 步: $(step_label "$1")"; }
# 把一串步骤号渲染成 "1 数据采集 → 2 云端训练 → ..." 的形式
render_steps() {
    local result="" sep=""
    for n in $1; do
        result+="${sep}${n} $(step_label "$n")"
        sep=" → "
    done
    echo "$result"
}
# 打印完整的步骤列表 (用于 banner / help)
render_step_list() {
    for n in 1 2 3 4; do
        printf "    ${BOLD}[%s]${NC} ${CYAN}%-8s${NC}  %s\n" "$n" "$(step_label "$n")" "$(step_desc "$n")"
    done
}

# ── 参数 ────────────────────────────────────────────────────────────────────
STEP=""
ROBOT="g1"
NUM_EPISODES=100
NUM_EPOCHS=10
INSTRUCTION="walk forward"
SSH_HOST=""
SSH_PORT=""
SSH_PASS=""
MODEL_SIZE="1.7-3B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2";               shift 2 ;;
        --robot)         ROBOT="$2";              shift 2 ;;
        --episodes)      NUM_EPISODES="$2";       shift 2 ;;
        --epochs)        NUM_EPOCHS="$2";         shift 2 ;;
        --instruction)   INSTRUCTION="$2";        shift 2 ;;
        --ssh-host)      SSH_HOST="$2";           shift 2 ;;
        --ssh-port)      SSH_PORT="$2";           shift 2 ;;
        --model-size)    MODEL_SIZE="$2";         shift 2 ;;
        -h|--help)
            echo "用法: $0 [--step SPEC] [--robot g1|go2] [--episodes N] [--epochs N]"
            echo "      [--instruction 'text'] [--ssh-host user@host] [--ssh-port PORT]"
            echo ""
            echo "流程步骤:"
            render_step_list
            echo ""
            echo "步骤选择 (--step), 可省略进入交互式菜单:"
            echo "  all                完整 4 步 (1,2,3,4)"
            echo "  local              仅本地 (1,4, 跳过云端)"
            echo "  cloud              仅云端 (2,3)"
            echo "  1|2|3|4            单步执行"
            echo "  1,3,4              逗号分隔组合"
            echo "  1-3                范围展开 (等价于 1,2,3)"
            echo ""
            echo "示例:"
            echo "  $0 --step 1                   # 只跑 [1] 数据采集"
            echo "  $0 --step 2,3 --ssh-host user@host -p 12345   # 跑 [2] 云端训练 + [3] 模型下载"
            echo "  $0 --step 1-3 --robot go2     # 跑 [1] 数据采集 → [2] 云端训练 → [3] 模型下载"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
MODEL_PACK_NAME="${ROBOT}_gr00t_model.tar.gz"

# ── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${BOLD}  GR00T × unitree_rl_mjlab × AutoDL — 一键训练全流程            ${NC}${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "项目目录:  $(relpath "$PROJECT_ROOT")"
info "机器人:    $ROBOT"
info "Episodes:  $NUM_EPISODES"
info "Epochs:    $NUM_EPOCHS"
info "模型:      GR00T-N1-$MODEL_SIZE"
info "语言指令:  $INSTRUCTION"
echo ""
echo -e "${CYAN}[流程步骤]${NC}"
render_step_list
echo ""

# ════════════════════════════════════════════════════════════════════════════
# [1] 数据采集 — 本地 unitree_rl_mjlab 收集数据 + 打包
# ════════════════════════════════════════════════════════════════════════════
run_step1() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    step "$(step_full 1) — $(step_desc 1)"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    bash "$SCRIPT_DIR/scripts/01_local_collect.sh" \
        --robot "$ROBOT" \
        --episodes "$NUM_EPISODES" \
        --instruction "$INSTRUCTION"

    PACK_PATH="$PROJECT_ROOT/$PACK_NAME"
    if [ -f "$PACK_PATH" ]; then
        info "打包文件: $PACK_PATH"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# [2] 云端训练 — AutoDL LoRA 微调 + INT4 量化
# ════════════════════════════════════════════════════════════════════════════
run_step2() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    step "$(step_full 2) — $(step_desc 2)"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # ── 上传训练包 ────────────────────────────────────────────────────
    PACK_PATH="$PROJECT_ROOT/$PACK_NAME"
    if [ ! -f "$PACK_PATH" ]; then
        fail "未找到训练包: $PACK_PATH, 请先运行 [1] 数据采集"
    fi

    if [ -z "$SSH_HOST" ]; then
        warn "未提供 SSH_HOST, 跳过自动上传"
        echo ""
        echo "请手动执行以下步骤:"
        echo "  1. 租用 AutoDL 实例 (RTX 5090 32GB 推荐)"
        echo "  2. 上传训练包:"
        echo "     scp -P <port> $PACK_PATH root@<host>:~/workspace/"
        echo "  3. SSH 登录并训练:"
        echo "     ssh -p <port> root@<host>"
        echo "     bash 02_autodl_train.sh --robot $ROBOT --epochs $NUM_EPOCHS"
        return 0
    fi

    if [ -z "$SSH_PORT" ]; then
        fail "请提供 --ssh-port 参数"
    fi

    step "上传训练包到 $SSH_HOST:$SSH_PORT..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "$PACK_PATH" "$SSH_HOST:~/workspace/$PACK_NAME"

    # ── 同步训练脚本 ─────────────────────────────────────────────────
    step "同步训练脚本到云端..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/scripts/02_autodl_train.sh" "$SSH_HOST:~/workspace/"

    # ── 远程执行训练 ─────────────────────────────────────────────────
    step "在云端启动训练..."
    echo -e "${YELLOW}⚠️  训练可能需要 1-4 小时, 请耐心等待${NC}"
    echo ""

    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST" \
        "cd ~/workspace && bash 02_autodl_train.sh --robot $ROBOT --epochs $NUM_EPOCHS --model-size $MODEL_SIZE"

    info "✅ [2] 云端训练完成!"
}

# ════════════════════════════════════════════════════════════════════════════
# [3] 模型下载 — SCP 下载 FP16 全量 + INT4 量化 包
# ════════════════════════════════════════════════════════════════════════════
run_step3() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    step "$(step_full 3) — $(step_desc 3)"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ]; then
        bash "$SCRIPT_DIR/scripts/03_download_model.sh" \
            "$SSH_HOST" -p "$SSH_PORT" \
            --robot "$ROBOT"
    else
        warn "未提供 SSH 信息, 请手动下载:"
        echo "  ./scripts/03_download_model.sh root@xxx.autodl.com -p 12345 --robot $ROBOT"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# [4] 本地推理 — 加载模型 + 验证推理输出
# ════════════════════════════════════════════════════════════════════════════
run_step4() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    step "$(step_full 4) — $(step_desc 4)"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    bash "$SCRIPT_DIR/scripts/04_local_verify.sh" \
        --robot "$ROBOT" \
        --instruction "$INSTRUCTION"
}

# ════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════

# ── 将步骤描述展开成有序、去重的步骤列表 (空格分隔) ────────────────────────
# 输入示例:
#   all         → "1 2 3 4"
#   local       → "1 4"
#   cloud       → "2 3"
#   1           → "1"
#   1,3,4       → "1 3 4"
#   1-3         → "1 2 3"
#   1,3-4       → "1 3 4"
expand_steps() {
    local spec="$1"
    spec="${spec,,}"              # 转小写, 方便 all/All/ALL
    local out=""

    case "$spec" in
        ""|all|full)  echo "1 2 3 4"; return ;;
        local)        echo "1 4";    return ;;
        cloud)        echo "2 3";    return ;;
    esac

    # 把逗号分隔的每段独立处理 (支持 "1,3-4" 这类混合写法)
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part// /}"        # 去空白
        if [[ -z "$part" ]]; then continue; fi

        if [[ "$part" =~ ^([1-4])-([1-4])$ ]]; then
            local s="${BASH_REMATCH[1]}"
            local e="${BASH_REMATCH[2]}"
            if (( s > e )); then
                warn "忽略非法范围: $part (起点大于终点)"
                continue
            fi
            for (( i=s; i<=e; i++ )); do
                out+="$i "
            done
        elif [[ "$part" =~ ^[1-4]$ ]]; then
            out+="$part "
        else
            warn "忽略无效步骤: $part (应为 1-4 / 1-3 这类写法)"
        fi
    done

    # 去重并保持原顺序
    local seen="" result=""
    for n in $out; do
        if [[ " $seen " != *" $n "* ]]; then
            result+="$n "
            seen+="$n "
        fi
    done

    if [ -z "$result" ]; then
        fail "步骤描述解析失败: '$spec' (没有有效的步骤)"
    fi
    # 去掉尾部空格, 让单步输出更干净
    echo "${result% }"
}

# ── 交互式确认 (未指定 --step 时) ───────────────────────────────────────────
if [ -z "${STEP:-}" ]; then
    echo "请选择执行模式:"
    echo "  1) 完整流程      (1→2→3→4, 推荐)"
    echo "  2) 仅本地流程    (1→4, 跳过云端, 用于本地模型验证)"
    echo "  3) 仅云端流程    (2→3, 训练 + 下载, 假设数据已上传)"
    echo "  4) 自定义步骤    (手动输入, 如 1 / 1,3 / 1-3)"
    echo ""
    read -p "输入选项 [1/2/3/4, 默认 1]: " MODE
    MODE="${MODE:-1}"

    case "$MODE" in
        1) STEP="all" ;;
        2) STEP="local" ;;
        3) STEP="cloud" ;;
        4)
            read -p "输入步骤 (例: 1 / 1,3 / 1-3 / 1,3-4): " STEP
            ;;
        *) fail "无效选项: $MODE" ;;
    esac
fi

# ── 展开步骤并按顺序执行 ────────────────────────────────────────────────────
STEPS=$(expand_steps "${STEP:-}")
info "将执行步骤: $(render_steps "$STEPS")"
echo ""

for s in $STEPS; do
    case "$s" in
        1) run_step1 ;;
        2) run_step2 ;;
        3) run_step3 ;;
        4) run_step4 ;;
        *) warn "跳过未知步骤: $s" ;;
    esac
done

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${BOLD}  🎉 已完成: $(render_steps "$STEPS")${NC}${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
