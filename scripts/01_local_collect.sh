#!/usr/bin/env bash
# ============================================================================
# 第 1 步: 本地收集数据 + 转换格式 + 打包
#
# 运行环境: 本地 (基于 unitree_rl_mjlab)
# 作用: 在 mjlab (MuJoCo) 仿真中收集 G1/Go2 演示数据, 转为 LeRobot v2 格式, 打包上传
#
# 前置条件:
#   - 已安装 unitree_rl_mjlab: cd ../unitree_rl_mjlab && pip install -e .
#
# 用法:
#   ./01_local_collect.sh
#   ./01_local_collect.sh --robot g1 --episodes 200 --instruction "walk to the door"
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
TASK="Unitree-G1-Flat"
NUM_EPISODES=100
EPISODE_LENGTH=200
INSTRUCTION="walk forward"
SPEED=0.5
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";          shift 2 ;;
        --task)         TASK="$2";           shift 2 ;;
        --episodes)     NUM_EPISODES="$2";   shift 2 ;;
        --length)       EPISODE_LENGTH="$2"; shift 2 ;;
        --instruction)  INSTRUCTION="$2";    shift 2 ;;
        --speed)        SPEED="$2";          shift 2 ;;
        --seed)         SEED="$2";           shift 2 ;;
        -h|--help)
            echo "用法: $0 [--robot g1|go2] [--task TASK_ID] [--episodes N]"
            echo "      [--length N] [--instruction 'text'] [--speed F] [--seed N]"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# 根据 robot 自动选择默认任务
if [[ "$ROBOT" == "go2" && "$TASK" == "Unitree-G1-Flat" ]]; then
    TASK="Unitree-Go2-Flat"
fi

# ── 路径 ────────────────────────────────────────────────────────────────────
RAW_DIR="$PROJECT_ROOT/data/${ROBOT}_raw"
LEROBOT_DIR="$PROJECT_ROOT/data/${ROBOT}_lerobot"
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${BOLD}  GR00T × unitree_rl_mjlab × AutoDL — 第 1 步: 本地收集数据    ${NC}${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "机器人:     $ROBOT"
info "任务:       $TASK"
info "Episodes:   $NUM_EPISODES"
info "Episode 长度: $EPISODE_LENGTH steps"
info "语言指令:   $INSTRUCTION"
info "步态速度:   $SPEED"
echo ""

# ── 检查 unitree_rl_mjlab ──────────────────────────────────────────────────
step "检查 unitree_rl_mjlab 安装..."
if [ ! -d "$RL_MJLAB_ROOT" ]; then
    fail "未找到 unitree_rl_mjlab 目录: $RL_MJLAB_ROOT"
fi
if ! python3 -c "import mjlab" 2>/dev/null; then
    warn "mjlab 未安装, 请先安装 unitree_rl_mjlab:"
    echo "  cd $RL_MJLAB_ROOT && pip install -e ."
    fail "缺少 mjlab 依赖"
fi
info "unitree_rl_mjlab: OK"

# ════════════════════════════════════════════════════════════════════════════
# Step 1: 收集数据
# ════════════════════════════════════════════════════════════════════════════
step "Step 1/3: 在 unitree_rl_mjlab 仿真中收集演示数据..."
python3 "$SRC_DIR/collect_data.py" \
    --task "$TASK" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    --instruction "$INSTRUCTION" \
    --output-dir "$RAW_DIR" \
    --speed "$SPEED" \
    --seed "$SEED"

if [ ! -d "$RAW_DIR" ]; then
    fail "数据收集失败: $RAW_DIR 不存在"
fi
EPISODE_COUNT=$(ls "$RAW_DIR"/episode_*.npz 2>/dev/null | wc -l)
info "已收集 $EPISODE_COUNT 个 episodes"

# ════════════════════════════════════════════════════════════════════════════
# Step 2: 转换为 LeRobot v2 格式
# ════════════════════════════════════════════════════════════════════════════
step "Step 2/3: 转换为 LeRobot v2 格式..."
python3 "$SRC_DIR/convert_to_lerobot.py" \
    --robot "$ROBOT" \
    --data-dir "$RAW_DIR" \
    --output-dir "$LEROBOT_DIR"

if [ ! -d "$LEROBOT_DIR" ]; then
    fail "格式转换失败: $LEROBOT_DIR 不存在"
fi
info "LeRobot v2 格式: $LEROBOT_DIR"

# ════════════════════════════════════════════════════════════════════════════
# Step 3: 打包
# ════════════════════════════════════════════════════════════════════════════
step "Step 3/3: 打包上传..."

# 打包 LeRobot 数据 (主要数据)
cd "$PROJECT_ROOT"
tar -czf "$PACK_NAME" -C "$PROJECT_ROOT/data" "$(basename "$LEROBOT_DIR")"
PACK_SIZE=$(du -h "$PACK_NAME" | cut -f1)
info "打包完成: $PACK_NAME ($PACK_SIZE)"

# ── 上传提示 ──────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🎉 第 1 步完成!${NC}"
echo ""
echo "上传到 AutoDL 云端:"
echo "  scp -P <port> $PACK_NAME root@<host>:~/workspace/"
echo ""
echo "进入第 3 步 (云端训练, 两种方式):"
echo "  方式 A (推荐): 回到本地, 运行  ./start.sh 3"
echo "  方式 B: SSH 到云端, 运行  bash scripts/03_autodl_train.sh --robot $ROBOT --epochs 10"
echo ""
