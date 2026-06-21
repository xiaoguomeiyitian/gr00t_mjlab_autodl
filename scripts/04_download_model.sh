#!/usr/bin/env bash
# ============================================================================
# 第 4 步: 从 AutoDL 下载训练好的模型到本地 (下载 → 解压 → 验证)
#
# 运行环境: 本地
# 作用: SCP 下载模型包 (BF16 完整模型 + INT4 量化) → 解压 → 校验
#
# 前置条件:
#   - 已填写 scripts/_ssh_config.sh 中的 SSH_HOST / SSH_PORT
#   - 云端 [3] 已训练完成, 模型包在 /root/workspace/ 下
#
# 用法:
#   ./04_download_model.sh                              # 用 _ssh_config.sh 的配置
#   ./04_download_model.sh --robot g1
#   ./04_download_model.sh --host root@x.com -p 12345  # 命令行覆盖
#   ./04_download_model.sh --with-int4                  # 同时下载 INT4 包
# ============================================================================
set -euo pipefail

# ── 加载 SSH 配置 ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_ssh_config.sh"

# 项目根目录 = 脚本所在目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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
PACK_NAME=""               # 主模型包 (BF16 完整模型, 03_autodl_train.sh 输出)
INT4_PACK=""               # INT4 量化包
SSH_HOST=""
SSH_PORT=""
LOCAL_MODEL_DIR="$PROJECT_ROOT/models"
EXTRACT_DIR="$PROJECT_ROOT/models"
SKIP_VERIFY=false
INCLUDE_INT4=true          # 默认下载 INT4 量化模型
INCLUDE_FP16=true          # 默认下载 BF16 完整模型

# 默认行为 (v2 优化版):
#   - BF16 完整模型: 默认下载 (本地需要用于推理/转换)
#   - INT4 量化模型: 默认不下载 (本地 8GB 显卡可自行量化, 节省下载量)
# 用 --with-int4 显式启用 INT4 下载 (旧行为, 兼容老用户)
INCLUDE_INT4=false
INCLUDE_FP16=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)        ROBOT="$2";          shift 2 ;;
        --port|-p)      SSH_PORT="$2";       shift 2 ;;
        --pack-name)    PACK_NAME="$2";      shift 2 ;;
        --local-dir)    LOCAL_MODEL_DIR="$2"; shift 2 ;;
        --skip-verify)  SKIP_VERIFY=true;    shift   ;;
        # 新行为 (默认): 只下载 BF16 完整模型, INT4 留待本地量化
        --with-int4)    INCLUDE_INT4=true;   shift   ;;
        --no-int4)      INCLUDE_INT4=false;  shift   ;;   # 兼容别名
        --with-fp16)    INCLUDE_FP16=true;   shift   ;;   # 兼容别名
        --no-fp16)      INCLUDE_FP16=false;  shift   ;;
        # SSH 覆盖 (命令行优先)
        --host)         SSH_HOST="$2";       shift 2 ;;
        --user)         SSH_USER="$2";       shift 2 ;;
        --key)          SSH_KEY="$2";        shift 2 ;;
        --remote-dir)   REMOTE_DIR="$2";     shift 2 ;;
        -h|--help)
            cat <<EOF
用法: $0 [user@host] [-p PORT] [--robot g1|go2] [选项]

如果未传 user@host, 会自动读取 scripts/_ssh_config.sh 中的 SSH_HOST

下载选项 (默认只下载 BF16 完整模型, 本地自行量化):
  --with-int4      同时下载云端 INT4 量化包 (~1.5GB)
  --no-int4        不下载 INT4 (默认)
  --no-fp16        不下载 BF16 完整模型
  --with-fp16      下载 BF16 完整模型 (默认)

其他选项:
  --skip-verify    下载后跳过验证
  --local-dir DIR  模型解压目录 (默认: 项目根/models)
  --host HOST      覆盖 SSH 主机
  --user USER      覆盖 SSH 用户
  --key PATH       覆盖 SSH 私钥路径
  --remote-dir DIR 覆盖云端工作目录

示例:
  # 推荐: 用 _ssh_config.sh 配置, 只下载 BF16 完整模型
  $0 --robot g1

  # 命令行覆盖, 同时下 BF16 + INT4
  $0 root@host -p 12345 --robot g1 --with-int4

  # 只下 INT4 (假定本地已有 BF16 完整模型)
  $0 --robot g1 --no-fp16 --with-int4
EOF
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

# SSH_HOST 归一化 (兼容 "host" 或 "user@host")
if [ -n "$SSH_HOST" ] && [[ "$SSH_HOST" != *@* ]] && [ -n "$SSH_USER" ]; then
    SSH_HOST="${SSH_USER}@${SSH_HOST}"
fi
# 端口默认值
[ -z "$SSH_PORT" ] && SSH_PORT="22"

# 默认包名 (与 03_autodl_train.sh 一致)
[ -z "$PACK_NAME" ] && PACK_NAME="${ROBOT}_gr00t_model.tar.gz"
[ -z "$INT4_PACK" ] && INT4_PACK="${ROBOT}_gr00t_int4_model.tar.gz"
# BF16 完整模型包的别名 (兼容两种命名)
FP16_PACK_NAMES=(
    "${ROBOT}_gr00t_model.tar.gz"         # 当前名称 (03_autodl_train.sh 生成, BF16 完整模型)
    "${ROBOT}_gr00t_full_fp16.tar.gz"     # 别名 (03_autodl_train.sh 创建 symlink 指向主包)
)

if [ -z "$SSH_HOST" ]; then
    _ssh_config_warn
    fail "未提供 SSH 信息, 请填写 scripts/_ssh_config.sh 或命令行传 user@host -p PORT"
fi

# 构造 SSH/SCP 基础参数
SSH_BASE_ARGS=(-p "$SSH_PORT" $SSH_OPTS)
[ -n "$SSH_KEY" ] && SSH_BASE_ARGS+=(-i "$SSH_KEY")
SCP_ARGS=(-P "$SSH_PORT" $SSH_OPTS)
[ -n "$SSH_KEY" ] && SCP_ARGS+=(-i "$SSH_KEY")

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × unitree_rl_mjlab × AutoDL — 第 4 步: 下载模型         ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "SSH:    $SSH_HOST"
info "端口:   $SSH_PORT"
[ -n "$SSH_KEY" ] && info "认证:   密钥 ($SSH_KEY)" || warn "认证:   密码 (推荐改用 SSH_KEY)"
info "远端:   $REMOTE_DIR"
info "机器人: $ROBOT"
echo ""
info "下载计划:"
$INCLUDE_FP16 && echo "  ✓ BF16 完整模型 (~7GB)  → 默认下载, 本地推理/量化的基础"
$INCLUDE_FP16 || echo "  ✗ BF16 完整模型  → 跳过 (--no-fp16)"
$INCLUDE_INT4  && echo "  ✓ INT4 量化 (~1.5GB) → 云端已量化, 节省本地 GPU 时间"
$INCLUDE_INT4  || echo "  ✗ INT4 量化  → 跳过 (--with-int4 启用, 或本地 05_local_quantize.sh)"
echo ""
$INCLUDE_FP16 && $INCLUDE_INT4 || warn "默认仅下载 BF16 完整模型. INT4 量化请在本地运行 ./scripts/05_local_quantize.sh"
echo ""

# ── 测试 SSH ────────────────────────────────────────────────────────────────
step "测试 SSH 连接..."
if ssh "${SSH_BASE_ARGS[@]}" -o ConnectTimeout=10 "$SSH_HOST" "echo 'SSH OK'" 2>/dev/null; then
    info "SSH 连接正常"
else
    fail "SSH 连接失败"
fi

# ── 查找远端模型包 ──────────────────────────────────────────────────────────
step "检查远端模型包..."

REMOTE_PACK=""
SEARCH_PATHS=("$REMOTE_DIR" /workspace /root)
for candidate in "${FP16_PACK_NAMES[@]}"; do
    for base in "${SEARCH_PATHS[@]}"; do
        p="$base/$candidate"
        if ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "test -f '$p'" 2>/dev/null; then
            REMOTE_PACK="$p"
            REMOTE_PACK_NAME="$candidate"
            break 2
        fi
    done
done

if [ -z "$REMOTE_PACK" ]; then
    fail "未找到 BF16 模型包 (尝试: ${FP16_PACK_NAMES[*]})\n常见路径: $REMOTE_DIR/  /workspace/"
fi
info "远端 BF16 模型包: $REMOTE_PACK"

REMOTE_SIZE=$(ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "du -h '$REMOTE_PACK' | cut -f1")
info "远端大小: $REMOTE_SIZE"

# ── 创建本地目录 ────────────────────────────────────────────────────────────
mkdir -p "$LOCAL_MODEL_DIR"

# ── 下载 BF16 完整模型 (供高显存 GPU 使用) ──────────────────────────────────
if $INCLUDE_FP16; then
    step "下载 BF16 完整模型 (供高显存 GPU 使用)..."
    LOCAL_PACK="$LOCAL_MODEL_DIR/$REMOTE_PACK_NAME"
    scp "${SCP_ARGS[@]}" "$SSH_HOST:$REMOTE_PACK" "$LOCAL_PACK"
    FP16_SIZE=$(du -h "$LOCAL_PACK" | cut -f1)
    info "✅ BF16 完整模型已下载: $LOCAL_PACK ($FP16_SIZE)"
else
    warn "跳过 BF16 完整模型 (--no-fp16)"
fi

# ── 下载 INT4 量化模型 (供低显存 GPU 使用) ─────────────────────────────────
if $INCLUDE_INT4; then
    REMOTE_INT4=""
    for base in "${SEARCH_PATHS[@]}"; do
        p="$base/$INT4_PACK"
        if ssh "${SSH_BASE_ARGS[@]}" "$SSH_HOST" "test -f '$p'" 2>/dev/null; then
            REMOTE_INT4="$p"
            break
        fi
    done

    if [ -n "$REMOTE_INT4" ]; then
        step "下载 INT4 量化模型 (供低显存 GPU 使用)..."
        LOCAL_INT4="$LOCAL_MODEL_DIR/$INT4_PACK"
        scp "${SCP_ARGS[@]}" "$SSH_HOST:$REMOTE_INT4" "$LOCAL_INT4"
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

# ── 解压 BF16 完整模型 ───────────────────────────────────────────────────
if $INCLUDE_FP16 && [ -f "$REMOTE_PACK_NAME" ]; then
    tar -xzf "$REMOTE_PACK_NAME"
    info "✅ BF16 完整模型解压完成: $EXTRACT_DIR/${REMOTE_PACK_NAME%.tar.gz}"
fi

# ── 解压 INT4 量化模型 ─────────────────────────────────────────────────────
if $INCLUDE_INT4 && [ -n "$REMOTE_INT4" ] && [ -f "$INT4_PACK" ]; then
    tar -xzf "$INT4_PACK"
    info "✅ INT4 模型解压完成: $EXTRACT_DIR/${INT4_PACK%.tar.gz}"
fi

# ── 验证 ────────────────────────────────────────────────────────────────────
if ! $SKIP_VERIFY; then
    step "验证模型文件..."

    # 验证 BF16
    FP16_DIR="$EXTRACT_DIR/${REMOTE_PACK_NAME%.tar.gz}"
    if $INCLUDE_FP16; then
        if [ ! -d "$FP16_DIR" ]; then
            fail "BF16 模型目录不存在: $FP16_DIR"
        fi
        FILE_COUNT=$(find "$FP16_DIR" -type f | wc -l)
        info "✅ BF16 完整模型: $FILE_COUNT 个文件"
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
echo -e "${GREEN}🎉 第 4 步完成!${NC}"
echo ""
echo "已下载模型:"
$INCLUDE_FP16 && [ -d "${EXTRACT_DIR}/${REMOTE_PACK_NAME%.tar.gz}" ] && \
    echo "  📦 BF16 完整模型 (~7GB, 高显存 GPU RTX 4090 24GB+):"
    echo "     ${EXTRACT_DIR}/${REMOTE_PACK_NAME%.tar.gz}"
$INCLUDE_INT4 && [ -d "${EXTRACT_DIR}/${INT4_PACK%.tar.gz}" ] && \
    echo "  📦 INT4 量化 (~1.5GB, 低显存 GPU RTX 2080 8GB):"
    echo "     ${EXTRACT_DIR}/${INT4_PACK%.tar.gz}"
echo ""

# 智能提示下一步
if $INCLUDE_FP16 && ! $INCLUDE_INT4; then
    echo -e "${CYAN}💡 提示:${NC}"
    echo "  你的本地 GPU 仅 8GB? 可运行本地量化脚本生成 INT4 模型:"
    echo "    ./scripts/05_local_quantize.sh --robot $ROBOT"
    echo ""
fi

echo "进入第 6 步 (本地推理验证):"
echo "  ./scripts/06_local_verify.sh --robot $ROBOT"
echo ""
echo "💡 提示:"
echo "  - 高显存 GPU (≥24GB): 使用 BF16 完整模型推理质量更高"
echo "  - 低显存 GPU (≤12GB): 使用 INT4 量化模型"
echo ""
