#!/bin/bash
# ─── GR00T MJLab AutoDL 统一入口 ───
#
# 用法:
#   ./start.sh              # 交互模式（菜单选择）
#   ./start.sh help         # 查看帮助
#
#   ./start.sh init         云端环境初始化
#   ./start.sh server       云端启动 Policy Server
#   ./start.sh tunnel       本地建立 SSH 隧道
#   ./start.sh collect [robot] [num_episodes] [episode_length] [action_mode]
#   ./start.sh upload [robot] [lerobot_dir]
#   ./start.sh train [robot]
#   ./start.sh download [robot]
#   ./start.sh quantize [robot]
#   ./start.sh verify [robot] [vis_mode: viser|mujoco]
#   ./start.sh viser-infer [robot] [host] [port] [viser_port]
#   ./start.sh mujoco-infer [robot] [host] [port]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

show_menu() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   GR00T MJLab AutoDL — 云端推理 + 本地训练编排          ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}═══ 云端操作 ═══${NC}                                        ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}1)${NC} 云端 — 环境初始化（一次性）                         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}2)${NC} 云端 — 启动 Policy Server                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}3)${NC} 云端 — 微调训练                                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}═══ 本地操作 ═══${NC}                                        ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}4)${NC} 本地 — 建立 SSH 隧道                                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}5)${NC} 本地 — MJLab 数据采集 + 格式转换                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}6)${NC} 本地 — 上传数据集到 AutoDL                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}7)${NC} 本地 — 下载模型                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}8)${NC} 本地 — INT4 量化                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}9)${NC} 本地 — 推理验证                                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${MAGENTA}10)${NC} Viser + Policy Server 推理可视化                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${MAGENTA}11)${NC} MuJoCo + Policy Server 推理可视化                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}S)${NC} 查看配置                                            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}H)${NC} 查看帮助                                            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${RED}0)${NC} 退出                                                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

ROBOT_CHOICES=("g1" "h1" "h1_with_hand" "h1_2" "h2" "go2")
select_robot() {
    echo ""
    echo "  🤖 选择机器人:"
    echo "    [0] G1 人形机器人 (29-DOF) ← 默认"
    echo "    [1] H1 人形机器人 (20-DOF)"
    echo "    [2] H1 人形机器人 (带手, 46-DOF)"
    echo "    [3] H1.2 人形机器人 (52-DOF)"
    echo "    [4] H2 人形机器人 (32-DOF)"
    echo "    [5] Go2 四足机器人 (12-DOF)"
    echo -n "  请选择 [0-5] (默认 0): " && read ridx
    ridx="${ridx:-0}"
    robot="${ROBOT_CHOICES[$ridx]:-g1}"
    echo "  → 已选择: $robot"
}

get_defaults() {
    MODEL_PATH="nvidia/GR00T-N1.7-3B"
    EMBODIMENT_TAG="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    PORT=5555
    HOST="127.0.0.1"
}

run_init() {
    echo -e "${GREEN}🚀 云端环境初始化...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/00_autodl_init.sh"
}

run_server() {
    echo -e "${GREEN}🚀 启动 Policy Server...${NC}"
    echo ""
    echo "   模型: $MODEL_PATH"
    echo "   具身: $EMBODIMENT_TAG"
    echo "   端口: $PORT"
    echo ""
    bash "$SCRIPT_DIR/scripts/01_start_server.sh" "$MODEL_PATH" "$EMBODIMENT_TAG" "$PORT"
}

run_tunnel() {
    echo -e "${GREEN}🔒 建立 SSH 隧道...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/02_local_tunnel.sh"
}

run_collect() {
    local robot="${1:-g1}"
    local num_episodes="${2:-50}"
    local episode_length="${3:-300}"
    local action_mode="${4:-delta}"
    local output_dir="${5:-$SCRIPT_DIR/output/${robot}_raw}"
    local lerobot_dir="$SCRIPT_DIR/output/${robot}_lerobot"
    echo -e "${GREEN}🤖 数据采集 + 格式转换 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/04_local_collect.sh" "$robot" "$num_episodes" "$episode_length" "$action_mode" "$output_dir"

    echo ""
    echo -e "${CYAN}🔄 自动转换为 LeRobot v2 格式...${NC}"
    echo "   输入: $output_dir"
    echo "   输出: $lerobot_dir"
    echo ""
    $PYTHON -m src.convert_to_lerobot \
        --input-dir "$output_dir" \
        --output-dir "$lerobot_dir" \
        --robot "$robot"
    echo ""
    echo "✅ 采集 + 转换完成: $lerobot_dir"
}

select_lerobot_dir() {
    local robot="$1"
    local -a dirs=()
    local idx=0

    # 扫描 output/ 下所有 *_lerobot 目录
    for d in "$SCRIPT_DIR/output/"*_lerobot; do
        [ -d "$d" ] || continue
        dirs+=("$d")
        idx=$((idx + 1))
        printf "    [%d] %s\n" "$idx" "$(basename "$d")"
    done

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "   ⚠️  未找到 LeRobot 数据集目录"
        echo "   请先运行数据采集 (选项 5)"
        return 1
    fi

    echo ""
    echo -n "   请选择要上传的目录 [1-$idx] (默认 1): " && read sel
    sel="${sel:-1}"

    if ! [[ "$sel" =~ ^[0-9]+$ ]] || [ "$sel" -lt 1 ] || [ "$sel" -gt "$idx" ]; then
        echo "   ❌ 无效选择: $sel，使用默认值 1"
        sel=1
    fi

    LEROBOT_DIR_SELECTED="${dirs[$((sel - 1))]}"
    echo "   → 已选择: $(basename "$LEROBOT_DIR_SELECTED")"
    return 0
}

run_upload() {
    local robot="${1:-g1}"
    echo -e "${GREEN}� 上传数据集到 AutoDL...${NC}"
    echo ""

    if ! select_lerobot_dir "$robot"; then
        return 1
    fi

    local lerobot_dir="$LEROBOT_DIR_SELECTED"
    echo ""
    bash "$SCRIPT_DIR/scripts/05_upload_to_autodl.sh" "$robot" "" "$lerobot_dir"
}

run_train() {
    local robot="${1:-g1}"
    echo -e "${GREEN}🚀 云端微调训练 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/06_autodl_train.sh" "$robot"
}

run_download() {
    local robot="${1:-g1}"
    echo -e "${GREEN}📥 下载模型 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/07_download_model.sh" "$robot"
}

run_quantize() {
    local robot="${1:-g1}"
    echo -e "${GREEN}⚙️  INT4 量化 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/08_local_quantize.sh" "$robot"
}

run_verify() {
    local robot="${1:-g1}"
    local vis_mode="${2:-viser}"
    echo -e "${GREEN}🔍 本地推理验证 (${robot}, ${vis_mode})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/09_local_verify.sh" "$robot" "" "" "" "$vis_mode"
}

run_viser_infer() {
    local robot="${1:-g1}"
    local host="${2:-127.0.0.1}"
    local port="${3:-5555}"
    local viser_port="${4:-20006}"
    local dataset_path="${5:-$SCRIPT_DIR/output/${robot}_lerobot}"
    echo -e "${GREEN}🌐 Viser + Policy Server 推理可视化 (${robot})...${NC}"
    echo ""
    echo "   Policy Server: ${host}:${port}"
    echo "   Viser 端口: ${viser_port}"
    echo "   数据集: ${dataset_path}"
    echo ""
    $PYTHON -m src.viz.viser_infer \
        --robot "$robot" \
        --host "$host" \
        --port "$port" \
        --viser-port "$viser_port" \
        --dataset "$dataset_path" \
        --embodiment-tag "$EMBODIMENT_TAG"
}

run_mujoco_infer() {
    local robot="${1:-g1}"
    local host="${2:-127.0.0.1}"
    local port="${3:-5555}"
    local dataset_path="${4:-$SCRIPT_DIR/output/${robot}_lerobot}"
    echo -e "${GREEN}🖥️  MuJoCo + Policy Server 推理可视化 (${robot})...${NC}"
    echo ""
    echo "   Policy Server: ${host}:${port}"
    echo "   数据集: ${dataset_path}"
    echo ""
    $PYTHON -m src.viz.mujoco_infer \
        --robot "$robot" \
        --host "$host" \
        --port "$port" \
        --dataset "$dataset_path" \
        --embodiment-tag "$EMBODIMENT_TAG"
}

show_config() {
    echo -e "${YELLOW}📋 当前配置:${NC}"
    echo ""
    if [ -f "$SCRIPT_DIR/config/ssh_config.sh" ]; then
        cat "$SCRIPT_DIR/config/ssh_config.sh"
    else
        echo "  配置文件不存在"
    fi
    echo ""
}

show_help() {
    echo -e "${YELLOW}📖 帮助信息:${NC}"
    echo ""
    echo "  交互模式:"
    echo "    ./start.sh"
    echo ""
    echo "  非交互模式:"
    echo "    ./start.sh init         云端环境初始化"
    echo "    ./start.sh server       云端启动 Policy Server"
    echo "    ./start.sh tunnel       本地建立 SSH 隧道"
    echo "    ./start.sh collect [robot] [num_episodes] [episode_length] [action_mode]"
    echo "    ./start.sh upload [robot] [lerobot_dir]"
    echo "    ./start.sh train [robot]"
    echo "    ./start.sh download [robot]"
    echo "    ./start.sh quantize [robot]"
    echo "    ./start.sh verify [robot] [vis_mode: viser|mujoco]"
    echo "    ./start.sh viser-infer [robot] [host] [port] [viser_port]"
    echo "    ./start.sh mujoco-infer [robot] [host] [port]"
    echo ""
    echo "  robot 可选: g1, h1, h1_with_hand, h1_2, h2, go2"
    echo ""
    echo "  端到端示例（G1 机器人）:"
    echo "    # 上传并训练"
    echo "    ./start.sh upload g1"
    echo "    ./start.sh train g1"
    echo "    # 传统流程"
    echo "    ./start.sh collect g1 50 300 delta    # 采集 + 自动转换"
    echo "    ./start.sh upload g1                   # 选择目录后上传"
    echo "    ./start.sh download g1"
    echo "    ./start.sh quantize g1"
    echo "    ./start.sh verify g1 viser"
    echo ""
}

# ─── 非交互模式 ───
case "${1:-}" in
    init)
        get_defaults
        run_init
        exit 0
        ;;
    server)
        get_defaults
        run_server
        exit 0
        ;;
    tunnel)
        get_defaults
        run_tunnel
        exit 0
        ;;
    collect)
        run_collect "$2" "$3" "$4" "$5"
        exit 0
        ;;
    upload)
        run_upload "$2"
        exit 0
        ;;
    train)
        run_train "$2"
        exit 0
        ;;
    download)
        run_download "$2"
        exit 0
        ;;
    quantize)
        run_quantize "$2"
        exit 0
        ;;
    verify)
        run_verify "$2" "$3"
        exit 0
        ;;
    viser-infer)
        get_defaults
        run_viser_infer "$2" "$3" "$4" "$5"
        exit 0
        ;;
    mujoco-infer)
        get_defaults
        run_mujoco_infer "$2" "$3" "$4"
        exit 0
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
esac

# ─── 交互模式 ───
while true; do
    show_menu
    read -p "请选择 [0-11, S, H]: " choice

    case $choice in
        1) run_init ;;
        2)
            get_defaults
            run_server
            ;;
        3)
            select_robot
            run_train "${robot}"
            ;;
        4) run_tunnel ;;
        5)
            select_robot
            echo -n "Episode 数 [50]: " && read num_ep
            num_ep="${num_ep:-50}"
            echo -n "每步长度 [300]: " && read ep_len
            ep_len="${ep_len:-300}"
            echo -n "动作模式 [delta]: " && read act_mode
            act_mode="${act_mode:-delta}"
            echo "  参数: episodes=$num_ep, length=$ep_len, mode=$act_mode"
            run_collect "${robot}" "$num_ep" "$ep_len" "$act_mode"
            ;;
        6)
            select_robot
            run_upload "${robot}"
            ;;
        7)
            select_robot
            run_download "${robot}"
            ;;
        8)
            select_robot
            run_quantize "${robot}"
            ;;
        9)
            select_robot
            echo -n "可视化模式 [viser/mujoco] (默认 viser): " && read vis
            vis="${vis:-viser}"
            run_verify "${robot}" "$vis"
            ;;
        10)
            get_defaults
            select_robot
            echo -n "Policy Server 地址 (默认 ${HOST}): " && read host
            host="${host:-${HOST}}"
            echo -n "Policy Server 端口 (默认 ${PORT}): " && read port
            port="${port:-${PORT}}"
            echo -n "Viser 端口 (默认 20006): " && read viser_port
            viser_port="${viser_port:-20006}"
            run_viser_infer "${robot}" "$host" "$port" "$viser_port"
            ;;
        11)
            get_defaults
            select_robot
            echo -n "Policy Server 地址 (默认 ${HOST}): " && read host
            host="${host:-${HOST}}"
            echo -n "Policy Server 端口 (默认 ${PORT}): " && read port
            port="${port:-${PORT}}"
            run_mujoco_infer "${robot}" "$host" "$port"
            ;;
        [sS]) show_config ;;
        [hH]) show_help ;;
        0)
            echo ""
            echo -e "${CYAN}👋 再见！${NC}"
            echo ""
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}❌ 无效选择，请重新输入${NC}"
            ;;
    esac

done
