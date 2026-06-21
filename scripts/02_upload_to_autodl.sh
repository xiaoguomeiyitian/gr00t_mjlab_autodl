#!/usr/bin/env bash
# ============================================================================
# 第 2 步: 上传训练数据 + 训练脚本到 AutoDL 云端
#
# 运行环境: 本地
# 作用: SCP 把 [1] 生成的训练包 + 训练脚本同步到 AutoDL 云端
#       (云端不需要 git clone, 只需要 bash 03_autodl_train.sh)
#
# 前置条件:
#   - 已运行 [1] ./scripts/01_local_collect.sh 生成 {robot}_gr00t_training.tar.gz
#   - 已填写 scripts/_ssh_config.sh 中的 SSH_HOST / SSH_PORT
#   - AutoDL 实例已开机, 端口已开放
#
# 用法:
#   ./02_upload_to_autodl.sh                           # 默认上传当前 robot 的训练包
#   ./02_upload_to_autodl.sh --robot g1
#   ./02_upload_to_autodl.sh --host root@x.com -p 12345   # 命令行覆盖 SSH
#   ./02_upload_to_autodl.sh --dry-run                 # 只检查不真传
# ============================================================================
set -euo pipefail

# ── 加载 SSH 配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_ssh_config.sh"

# ── 颜色 ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
log()   { echo -e "${CYAN}[·]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ───────────────────────────────────────────────────────────────────
ROBOT="g1"
DRY_RUN=false
UPLOAD_TRAIN_SCRIPT=true          # 默认同步训练脚本 (03_autodl_train.sh)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";            shift 2 ;;
        --host)         SSH_HOST="$2";         shift 2 ;;
        --port|-p)      SSH_PORT="$2";         shift 2 ;;
        --user)         SSH_USER="$2";         shift 2 ;;
        --key)          SSH_KEY="$2";          shift 2 ;;
        --remote-dir)   REMOTE_DIR="$2";       shift 2 ;;
        --no-script)    UPLOAD_TRAIN_SCRIPT=false; shift ;;
        --dry-run)      DRY_RUN=true;          shift   ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── 端口默认值 (归一化放到 SSH_HOST 校验后, 避免空 host 变成 "root@") ──
[ -z "$SSH_PORT" ] && SSH_PORT="22"

# ── 路径 ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACK_NAME="${ROBOT}_gr00t_training.tar.gz"
PACK_PATH="$PROJECT_ROOT/$PACK_NAME"
TRAIN_SCRIPT="$SCRIPT_DIR/03_autodl_train.sh"

# ═══════════════════════════════════════════════════════════════════════════
# 校验
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${BOLD}  GR00T × unitree_rl_mjlab × AutoDL — 第 2 步: 上传到云端      ${NC}${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
$DRY_RUN && warn "🔍 DRY-RUN 模式: 只检查, 不真传"

# SSH 配置检查 (放在归一化前, 避免空 host 变成 "root@")
step "检查 SSH 配置..."
if [ -z "$SSH_HOST" ] || [ "$SSH_HOST" = "root@your-host.com" ]; then
    _ssh_config_warn
    fail "SSH_HOST 未填写, 请编辑 scripts/_ssh_config.sh 或用 --host 传参"
fi

# SSH 信息归一化 (此时 SSH_HOST 一定非空)
if [[ "$SSH_HOST" != *@* ]]; then
    if [ -n "$SSH_USER" ]; then
        SSH_HOST="${SSH_USER}@${SSH_HOST}"
    else
        SSH_HOST="root@${SSH_HOST}"
    fi
fi

info "SSH:    $SSH_HOST"
info "端口:   $SSH_PORT"
info "远端:   $REMOTE_DIR"
if [ -n "$SSH_KEY" ]; then
    info "认证:   密钥 ($SSH_KEY)"
elif [ -n "$SSH_PASS" ]; then
    warn "认证:   密码 (不推荐明文, 建议改用密钥)"
else
    warn "认证:   密码 (脚本会提示输入)"
fi
echo ""

# 训练包检查
step "检查训练包..."
if [ ! -f "$PACK_PATH" ]; then
    fail "训练包不存在: $PACK_PATH
请先运行 [1]: ./scripts/01_local_collect.sh --robot $ROBOT"
fi
PACK_SIZE=$(du -h "$PACK_PATH" | cut -f1)
info "训练包: $PACK_PATH ($PACK_SIZE)"

# 训练脚本检查
if $UPLOAD_TRAIN_SCRIPT; then
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        fail "训练脚本不存在: $TRAIN_SCRIPT"
    fi
fi

# SSH 可达性测试
step "测试 SSH 连接..."
SSH_BASE_ARGS=(-p "$SSH_PORT" $SSH_OPTS)
[ -n "$SSH_KEY" ] && SSH_BASE_ARGS+=(-i "$SSH_KEY")

if $DRY_RUN; then
    info "DRY-RUN: 跳过 SSH 连接测试"
elif ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "echo 'SSH OK'" 2>/dev/null; then
    info "✅ SSH 连接正常"
else
    fail "SSH 连接失败: $SSH_HOST:$SSH_PORT
排查:
  1. AutoDL 实例是否已开机?
  2. 端口是否正确 (在 AutoDL 控制台 '实例' → '更多' → '自定义服务' 查看)?
  3. 防火墙 / 安全组是否放行?
  4. 密钥路径 / 密码是否正确?"
fi

# ═══════════════════════════════════════════════════════════════════════════
# 上传
# ═══════════════════════════════════════════════════════════════════════════

# SCP 参数
SCP_ARGS=(-P "$SSH_PORT" $SSH_OPTS)
[ -n "$SSH_KEY" ] && SCP_ARGS+=(-i "$SSH_KEY")
[ -n "$SCP_BANDWIDTH_LIMIT" ] && SCP_ARGS+=(-l "$SCP_BANDWIDTH_LIMIT")

# 远端目录
step "准备云端目录 $REMOTE_DIR..."
if ! $DRY_RUN; then
    ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/models $REMOTE_DIR/logs"
    info "云端目录已就绪"
fi

# ── 上传训练包 ────────────────────────────────────────────────────────────
step "上传训练包 $PACK_NAME ($PACK_SIZE)..."
LOG_UPLOAD="$PROJECT_ROOT/logs/upload_${ROBOT}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_UPLOAD")"

if $DRY_RUN; then
    info "DRY-RUN: scp ${SCP_ARGS[*]} $PACK_PATH $SSH_HOST:$REMOTE_DIR/"
else
    echo "  日志: $LOG_UPLOAD"
    if scp "${SCP_ARGS[@]}" "$PACK_PATH" "$SSH_HOST:$REMOTE_DIR/$PACK_NAME" 2>&1 | tee "$LOG_UPLOAD"; then
        info "✅ 训练包上传成功"
    else
        fail "训练包上传失败, 日志: $LOG_UPLOAD"
    fi
fi

# ── 上传训练脚本 ──────────────────────────────────────────────────────────
if $UPLOAD_TRAIN_SCRIPT; then
    step "上传训练脚本 03_autodl_train.sh..."
    if $DRY_RUN; then
        info "DRY-RUN: scp ${SCP_ARGS[*]} $TRAIN_SCRIPT $SSH_HOST:$REMOTE_DIR/"
    else
        if scp "${SCP_ARGS[@]}" "$TRAIN_SCRIPT" "$SSH_HOST:$REMOTE_DIR/03_autodl_train.sh" 2>&1 | tee -a "$LOG_UPLOAD"; then
            # 确保可执行
            ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "chmod +x $REMOTE_DIR/03_autodl_train.sh"
            info "✅ 训练脚本上传成功"
        else
            fail "训练脚本上传失败, 日志: $LOG_UPLOAD"
        fi
    fi
fi

# ── 远端校验 ──────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    step "远端校验..."
    REMOTE_PACK_SIZE=$(ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "du -h '$REMOTE_DIR/$PACK_NAME' | cut -f1" 2>/dev/null || echo "?")
    REMOTE_SCRIPT_OK=$(ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "test -x '$REMOTE_DIR/03_autodl_train.sh' && echo OK || echo MISSING" 2>/dev/null)
    info "远端训练包: $REMOTE_PACK_SIZE"
    info "远端脚本:   $REMOTE_SCRIPT_OK"
fi

# ═══════════════════════════════════════════════════════════════════════════
# 下一步指引
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🎉 第 2 步完成!${NC}"
echo ""
echo "云端就绪文件:"
echo "  📦 $REMOTE_DIR/$PACK_NAME    ($PACK_SIZE)"
$UPLOAD_TRAIN_SCRIPT && echo "  📜 $REMOTE_DIR/03_autodl_train.sh"
echo ""
echo -e "${BOLD}进入第 3 步: 云端训练${NC}"
echo ""
echo "方式 A — 在云端手动训练 (推荐):"
echo "  ssh -p $SSH_PORT $SSH_HOST"
echo "  cd $REMOTE_DIR"
echo "  bash 03_autodl_train.sh --robot $ROBOT --epochs 10"
echo ""
echo "方式 B — 本地一键触发 (脚本已支持, 见 run_all.sh --step 3):"
echo "  ssh -p $SSH_PORT $SSH_HOST 'cd $REMOTE_DIR && bash 03_autodl_train.sh --robot $ROBOT'"
echo ""
