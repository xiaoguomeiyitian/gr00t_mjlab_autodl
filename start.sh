#!/bin/bash
# ─── GR00T MJLab AutoDL 统一入口 ───

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
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   GR00T MJLab AutoDL — 云端推理编排             ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}1)${NC} 云端 — 环境初始化（一次性）               ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}2)${NC} 云端 — 启动 Policy Server                   ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}3)${NC} 本地 — 建立 SSH 隧道                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${GREEN}4)${NC} 本地 — 运行 Demo 推理                      ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}5)${NC} 查看配置                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${YELLOW}6)${NC} 查看帮助                                 ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}   ${RED}0)${NC} 退出                                     ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                  ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ─── 执行 ───
while true; do
    show_menu
    read -p "请选择 [0-6]: " choice

    case $choice in
        1)
            echo ""
            echo -e "${GREEN}🚀 云端环境初始化...${NC}"
            echo ""
            bash "$SCRIPT_DIR/scripts/00_autodl_init.sh"
            ;;
        2)
            echo ""
            echo -e "${GREEN}🚀 启动 Policy Server...${NC}"
            echo ""
            model_path="nvidia/GR00T-N1.7-3B"
            embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
            port=5555

            echo "   模型: $model_path"
            echo "   具身: $embodiment_tag"
            echo "   端口: $port"
            echo ""

            bash "$SCRIPT_DIR/scripts/01_start_server.sh" "$model_path" "$embodiment_tag" "$port"
            ;;
        3)
            echo ""
            echo -e "${GREEN}🔒 建立 SSH 隧道...${NC}"
            echo ""
            bash "$SCRIPT_DIR/scripts/02_local_tunnel.sh"
            ;;
        4)
            echo ""
            echo -e "${GREEN}🚀 运行 Demo 推理...${NC}"
            echo ""
            dataset_path="$SCRIPT_DIR/../Isaac-GR00T/demo_data/droid_sample"
            embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
            host="127.0.0.1"
            port=5555
            output_dir="$SCRIPT_DIR/../output"

            echo "   数据集: $dataset_path"
            echo "   具身: $embodiment_tag"
            echo "   服务器: ${host}:${port}"
            echo "   输出: $output_dir"
            echo ""

            bash "$SCRIPT_DIR/scripts/03_local_demo_eval.sh" "$dataset_path" "$embodiment_tag" "$host" "$port" "$output_dir"
            ;;
        5)
            echo ""
            echo -e "${YELLOW}� 当前配置:${NC}"
            echo ""
            if [ -f "$SCRIPT_DIR/config/ssh_config.sh" ]; then
                cat "$SCRIPT_DIR/config/ssh_config.sh"
            else
                echo "  配置文件不存在"
            fi
            echo ""
            ;;
        6)
            echo ""
            echo -e "${YELLOW}� 帮助信息:${NC}"
            echo ""
            echo "  使用流程:"
            echo "  �──────────────────────────────────────────────┐"
            echo "  │ 1. 云端: ./start.sh → 1 (环境初始化)          │"
            echo "  │ 2. 云端: ./start.sh → 2 (启动 Server)         │"
            echo "  │ 3. 本地: ./start.sh → 3 (SSH 隧道)           │"
            echo "  │ 4. 本地: ./start.sh → 4 (Demo 推理)          │"
            echo "  └──────────────────────────────────────────────┘"
            echo ""
            echo "  或直接执行脚本:"
            echo "    bash scripts/00_autodl_init.sh"
            echo "    bash scripts/01_start_server.sh"
            echo "    bash scripts/02_local_tunnel.sh"
            echo "    bash scripts/03_local_demo_eval.sh"
            echo ""
            ;;
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
