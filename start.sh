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
#   ./start.sh demo         本地运行 Demo 推理
#   ./start.sh auto         完整流程：tunnel → demo
#   ./start.sh collect [robot] [num_episodes] [episode_length] [action_mode]
#   ./start.sh upload [robot]
#   ./start.sh train [robot]
#   ./start.sh download [robot]
#   ./start.sh quantize [robot]
#   ./start.sh verify [robot] [model_path] [dataset_path] [vis_mode]
#   ./start.sh viser [robot] [port]
#   ./start.sh mujoco [robot]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── 颜色 ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── 菜单 ───
show_menu() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   GR00T MJLab AutoDL — 云端推理 + 本地训练编排          ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}1)${NC} 云端 — 环境初始化（一次性）                         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}2)${NC} 云端 — 启动 Policy Server                             ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}3)${NC} 本地 — 建立 SSH 隧道                                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}4)${NC} 本地 — Demo 推理                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}5)${NC} 本地 — MJLab 数据采集                                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}6)${NC} 本地 — 转换格式 + 上传到 AutoDL                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${CYAN}7)${NC} 云端 — 微调训练                                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}8)${NC} 本地 — 下载模型                                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}9)${NC} 本地 — INT4 量化                                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}10)${NC} 本地 — 推理验证                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${RED}11)${NC} Viser 浏览器 3D 可视化                              ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${RED}12)${NC} MuJoCo 桌面窗口可视化                               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}S)${NC} 查看配置                                            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}H)${NC} 查看帮助                                            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${RED}0)${NC} 退出                                                ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                          ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ─── 机器人选择 ───
ROBOT_CHOICES=("g1" "h1" "h1_with_hand" "h1_2" "h2" "go2")
select_robot() {
    echo ""
    echo "  🤖 选择机器人:"
    echo "    [0] G1 人形机器人 (29-DOF)"
    echo "    [1] H1 人形机器人 (20-DOF)"
    echo "    [2] H1 人形机器人 (带手, 46-DOF)"
    echo "    [3] H1.2 人形机器人 (52-DOF)"
    echo "    [4] H2 人形机器人 (32-DOF)"
    echo "    [5] Go2 四足机器人 (12-DOF)"
    echo -n "  请选择 [0-5]: " && read ridx
    robot="${ROBOT_CHOICES[$ridx]:-g1}"
    echo "  → 已选择: $robot"
}

# ─── 默认参数 ───
get_defaults() {
    MODEL_PATH="nvidia/GR00T-N1.7-3B"
    EMBODIMENT_TAG="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    PORT=5555
    HOST="127.0.0.1"
    DATASET_PATH="$SCRIPT_DIR/../Isaac-GR00T/demo_data/droid_sample"
    OUTPUT_DIR="$SCRIPT_DIR/../output"
}

# ─── 执行函数 ───
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

run_demo() {
    echo -e "${GREEN}🚀 运行 Demo 推理...${NC}"
    echo ""
    echo "   数据集: $DATASET_PATH"
    echo "   具身: $EMBODIMENT_TAG"
    echo "   服务器: ${HOST}:${PORT}"
    echo "   输出: $OUTPUT_DIR"
    echo ""
    bash "$SCRIPT_DIR/scripts/03_local_demo_eval.sh" "$DATASET_PATH" "$EMBODIMENT_TAG" "$HOST" "$PORT" "$OUTPUT_DIR"
}

run_collect() {
    local robot="${1:-g1}"
    local num_episodes="${2:-50}"
    local episode_length="${3:-300}"
    local action_mode="${4:-delta}"
    local output_dir="${5:-$SCRIPT_DIR/../output/${robot}_raw}"
    echo -e "${GREEN}🤖 数据采集 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/04_local_collect.sh" "$robot" "$num_episodes" "$episode_length" "$action_mode" "$output_dir"
}

run_upload() {
    local robot="${1:-g1}"
    echo -e "${GREEN}📤 转换格式 + 上传到 AutoDL (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/05_upload_to_autodl.sh" "$robot"
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
    local vis_mode="${2:-demo}"
    echo -e "${GREEN}🔍 本地推理验证 (${robot}, ${vis_mode})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/09_local_verify.sh" "$robot" "" "" "" "$vis_mode"
}

run_viser() {
    local robot="${1:-g1}"
    local port="${2:-20006}"
    echo -e "${GREEN}🌐 Viser 浏览器可视化 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/10_local_viser_client.sh" "$robot" "" "$port"
}

run_mujoco() {
    local robot="${1:-g1}"
    echo -e "${GREEN}🖥️  MuJoCo 桌面可视化 (${robot})...${NC}"
    echo ""
    bash "$SCRIPT_DIR/scripts/11_local_mujoco_client.sh" "$robot"
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
    echo "    ./start.sh demo         本地运行 Demo 推理"
    echo "    ./start.sh auto         完整流程：tunnel → demo"
    echo "    ./start.sh collect [robot] [num_episodes] [episode_length] [action_mode]"
    echo "    ./start.sh upload [robot]"
    echo "    ./start.sh train [robot]"
    echo "    ./start.sh download [robot]"
    echo "    ./start.sh quantize [robot]"
    echo "    ./start.sh verify [robot] [vis_mode: demo|viser|mujoco]"
    echo "    ./start.sh viser [robot] [port]"
    echo "    ./start.sh mujoco [robot]"
    echo ""
    echo "  robot 可选: g1, h1, h1_with_hand, h1_2, h2, go2"
    echo ""
    echo "  端到端示例（G1 机器人）:"
    echo "    ./start.sh collect g1 50 300 delta"
    echo "    ./start.sh upload g1"
    echo "    ./start.sh train g1"
    echo "    ./start.sh download g1"
    echo "    ./start.sh quantize g1"
    echo "    ./start.sh verify g1 demo"
    echo "    ./start.sh viser g1"
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
    demo)
        get_defaults
        run_demo
        exit 0
        ;;
    auto)
        get_defaults
        echo -e "${CYAN}🎯 非交互模式：tunnel → demo${NC}"
        echo ""
        run_tunnel
        run_demo
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
    viser)
        run_viser "$2" "$3"
        exit 0
        ;;
    mujoco)
        run_mujoco "$2"
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
    read -p "请选择 [0-12, S, H]: " choice

    case $choice in
        1) run_init ;;
        2)
            get_defaults
            run_server
            ;;
        3) run_tunnel ;;
        4)
            get_defaults
            run_demo
            ;;
        5)
            select_robot
            echo -n "Episode 数 [50]: " && read num_ep
            echo -n "每步长度 [300]: " && read ep_len
            echo -n "动作模式 [delta]: " && read act_mode
            run_collect "${robot}" "${num_ep:-50}" "${ep_len:-300}" "${act_mode:-delta}"
            ;;
        6)
            select_robot
            run_upload "${robot}"
            ;;
        7)
            select_robot
            run_train "${robot}"
            ;;
        8)
            select_robot
            run_download "${robot}"
            ;;
        9)
            select_robot
            run_quantize "${robot}"
            ;;
        10)
            select_robot
            echo -n "可视化 [demo/viser/mujoco]: " && read vis
            run_verify "${robot}" "${vis:-demo}"
            ;;
        11)
            select_robot
            echo -n "端口 [20006]: " && read port
            run_viser "${robot}" "${port:-20006}"
            ;;
        12)
            select_robot
            run_mujoco "${robot}"
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

    echo ""
    read -p "按 Enter 继续..."
done
