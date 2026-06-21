#!/usr/bin/env bash
# ============================================================================
# 第 3 步: 从 AutoDL 下载训练好的模型到本地
#
# 运行环境: 本地
# 作用: SCP 下载模型包 → 解压 → 验证
#
# 用法:
#   ./03_download_model.sh root@xxx.autodl.com -p 12345 --robot g1
# ============================================================================
set -euo pipefail

# 项目根目录 = 脚本所在目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 参数 ────────────────────────────────────────────────────────────────────
ROBOT="g1"
PACK_NAME=""               # 主模型包 (FP16 全量)
INT4_PACK=""               # INT4 量化包
SSH_HOST=""
SSH_PORT=""
LOCAL_MODEL_DIR="$PROJECT_ROOT/models"
EXTRACT_DIR="$PROJECT_ROOT/models"
SKIP_VERIFY=false
INCLUDE_INT4=true          # 默认下载 INT4 量化模型
INCLUDE_FP16=true          # 默认下载 FP16 全量模型

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";          shift 2 ;;
        --port|-p)      SSH_PORT="$2";       shift 2 ;;
        --pack-name)    PACK_NAME="$2";      shift 2 ;;
        --local-dir)    LOCAL_MODEL_DIR="$2"; shift 2 ;;
        --skip-verify)  SKIP_VERIFY=true;    shift   ;;
        --no-int4)      INCLUDE_INT4=false;  shift   ;;
        --no-fp16)      INCLUDE_FP16=false;  shift   ;;
        -h|--help)
            echo "用法: $0 <user@host> -p <port> [--robot g1|go2]"
            echo "      [--no-int4] [--no-fp16]   # 跳过某个模型"
            exit 0
            ;;
        -*)
            fail "未知选项: $1"
            ;;
        *)
            if [ -z "$SSH_HOST" ]; then
                SSH_HOST="$1"
            else
                fail "多余参数: $1"
            fi
            shift
            ;;
    esac
done

# 默认包名 (与 02_autodl_train.sh 一致)
[ -z "$PACK_NAME" ] && PACK_NAME="${ROBOT}_gr00t_model.tar.gz"
[ -z "$INT4_PACK" ] && INT4_PACK="${ROBOT}_gr00t_int4_model.tar.gz"
# FP16 全量包的别名 (兼容两种命名)
FP16_PACK_NAMES=(
    "${ROBOT}_gr00t_full_fp16.tar.gz"     # 新名称 (02_autodl_train.sh 生成)
    "${ROBOT}_gr00t_model.tar.gz"         # 别名 (旧名称, 也指向 FP16 全量)
)

if [ -z "$SSH_HOST" ]; then
    fail "请提供 SSH 地址, 例: $0 root@xxx.autodl.com -p 12345"
fi
if [ -z "$SSH_PORT" ]; then
    fail "请提供 SSH 端口 (-p)"
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 第 3 步: 下载模型         ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "SSH:    $SSH_HOST"
info "端口:   $SSH_PORT"
info "机器人: $ROBOT"
echo ""

# ── 测试 SSH ────────────────────────────────────────────────────────────────
step "测试 SSH 连接..."
if ssh -p "$SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SSH_HOST" "echo 'SSH OK'" 2>/dev/null; then
    info "SSH 连接正常"
else
    fail "SSH 连接失败"
fi

# ── 查找远端模型包 ──────────────────────────────────────────────────────────
step "检查远端模型包..."

REMOTE_PACK=""
SEARCH_PATHS=(/root/workspace /workspace /root)
for candidate in "${FP16_PACK_NAMES[@]}"; do
    for base in "${SEARCH_PATHS[@]}"; do
        p="$base/$candidate"
        if ssh -p "$SSH_PORT" "$SSH_HOST" "test -f '$p'" 2>/dev/null; then
            REMOTE_PACK="$p"
            REMOTE_PACK_NAME="$candidate"
            break 2
        fi
    done
done

if [ -z "$REMOTE_PACK" ]; then
    fail "未找到 FP16 模型包 (尝试: ${FP16_PACK_NAMES[*]})\n常见路径: /root/workspace/ 或 /workspace/"
fi
info "远端 FP16 模型包: $REMOTE_PACK"

REMOTE_SIZE=$(ssh -p "$SSH_PORT" "$SSH_HOST" "du -h '$REMOTE_PACK' | cut -f1")
info "远端大小: $REMOTE_SIZE"

# ── 创建本地目录 ────────────────────────────────────────────────────────────
mkdir -p "$LOCAL_MODEL_DIR"

# ── 下载 FP16 全量模型 (供高显存 GPU 使用) ──────────────────────────────────
if $INCLUDE_FP16; then
    step "下载 FP16 全量模型 (供高显存 GPU 使用)..."
    LOCAL_PACK="$LOCAL_MODEL_DIR/$REMOTE_PACK_NAME"
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST:$REMOTE_PACK" "$LOCAL_PACK"
    FP16_SIZE=$(du -h "$LOCAL_PACK" | cut -f1)
    info "✅ FP16 全量模型已下载: $LOCAL_PACK ($FP16_SIZE)"
else
    warn "跳过 FP16 全量模型 (--no-fp16)"
fi

# ── 下载 INT4 量化模型 (供低显存 GPU 使用) ─────────────────────────────────
if $INCLUDE_INT4; then
    REMOTE_INT4=""
    for base in "${SEARCH_PATHS[@]}"; do
        p="$base/$INT4_PACK"
        if ssh -p "$SSH_PORT" "$SSH_HOST" "test -f '$p'" 2>/dev/null; then
            REMOTE_INT4="$p"
            break
        fi
    done

    if [ -n "$REMOTE_INT4" ]; then
        step "下载 INT4 量化模型 (供低显存 GPU 使用)..."
        LOCAL_INT4="$LOCAL_MODEL_DIR/$INT4_PACK"
        scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "$SSH_HOST:$REMOTE_INT4" "$LOCAL_INT4"
        INT4_SIZE=$(du -h "$LOCAL_INT4" | cut -f1)
        info "✅ INT4 量化模型已下载: $LOCAL_INT4 ($INT4_SIZE)"
    else
        warn "远端未找到 INT4 模型包: $INT4_PACK (跳过)"
        REMOTE_INT4=""
    fi
else
    warn "跳过 INT4 量化模型 (--no-int4)"
fi

# ── 解压 ────────────────────────────────────────────────────────────────────
step "解压模型..."
cd "$EXTRACT_DIR"

# ── 解压 FP16 全量模型 ─────────────────────────────────────────────────────
if $INCLUDE_FP16 && [ -f "$REMOTE_PACK_NAME" ]; then
    tar -xzf "$REMOTE_PACK_NAME"
    info "✅ FP16 模型解压完成: $EXTRACT_DIR/${REMOTE_PACK_NAME%.tar.gz}"
fi

# ── 解压 INT4 量化模型 ─────────────────────────────────────────────────────
if $INCLUDE_INT4 && [ -n "$REMOTE_INT4" ] && [ -f "$INT4_PACK" ]; then
    tar -xzf "$INT4_PACK"
    info "✅ INT4 模型解压完成: $EXTRACT_DIR/${INT4_PACK%.tar.gz}"
fi

# ── 验证 ────────────────────────────────────────────────────────────────────
if ! $SKIP_VERIFY; then
    step "验证模型文件..."

    # 验证 FP16
    FP16_DIR="$EXTRACT_DIR/${REMOTE_PACK_NAME%.tar.gz}"
    if $INCLUDE_FP16; then
        if [ ! -d "$FP16_DIR" ]; then
            fail "FP16 模型目录不存在: $FP16_DIR"
        fi
        FILE_COUNT=$(find "$FP16_DIR" -type f | wc -l)
        info "✅ FP16 全量模型: $FILE_COUNT 个文件"
    fi

    # 验证 INT4
    INT4_DIR="$EXTRACT_DIR/${INT4_PACK%.tar.gz}"
    if $INCLUDE_INT4 && [ -n "$REMOTE_INT4" ]; then
        if [ -d "$INT4_DIR" ]; then
            INT4_FILE_COUNT=$(find "$INT4_DIR" -type f | wc -l)
            info "✅ INT4 量化模型: $INT4_FILE_COUNT 个文件"
        else
            warn "INT4 模型目录不存在 (可能未生成)"
        fi
    fi
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🎉 第 3 步完成! 已下载全部模型${NC}"
echo ""
echo "模型位置:"
$INCLUDE_FP16 && [ -d "${EXTRACT_DIR}/${REMOTE_PACK_NAME%.tar.gz}" ] && \
    echo "  📦 FP16 全量 (高显存 GPU, RTX 4090 24GB+):"
    echo "     ${EXTRACT_DIR}/${REMOTE_PACK_NAME%.tar.gz}"
$INCLUDE_INT4 && [ -d "${EXTRACT_DIR}/${INT4_PACK%.tar.gz}" ] && \
    echo "  📦 INT4 量化 (低显存 GPU, RTX 2080 8GB):"
    echo "     ${EXTRACT_DIR}/${INT4_PACK%.tar.gz}"
echo ""
echo "进入第 4 步 (本地推理验证):"
echo "  ./scripts/04_local_verify.sh --robot $ROBOT"
echo ""
echo "💡 提示:"
echo "  - 高显存 GPU (≥24GB): 使用 FP16 全量模型推理质量更高"
echo "  - 低显存 GPU (≤12GB): 使用 INT4 量化模型"
echo ""
