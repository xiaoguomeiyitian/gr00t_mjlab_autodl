#!/usr/bin/env bash
# ============================================================================
# 第 0 步: AutoDL 云端环境一键初始化
#
# 运行环境: AutoDL 云端实例 (推荐 RTX 5090 32GB / A100 40GB+)
# 实例镜像: PyTorch  2.x  +  CUDA 12.x  (官方或社区镜像均可)
# 作用: 一次性完成云端 GR00T 训练环境搭建
#       (1) 系统依赖  (2) uv 包管理器  (3) Isaac-GR00T 仓库
#       (4) Python 3.10 venv + 训练栈  (5) GR00T-N1 基础模型  (6) 验证
#
# 不依赖: unitree_rl_mjlab (仿真框架, 仅本地数据采集时使用)
#
# 前置条件:
#   - AutoDL 实例已开机并 ssh/sshpass 可达
#   - 已选择 PyTorch 2.x + CUDA 12.x 镜像 (实例市场搜索 "PyTorch")
#
# 用法:
#   # 标准用法 (默认 1.7-3B, /root/Isaac-GR00T)
#   bash 00_autodl_init.sh
#
#   # 自定义模型规格
#   bash 00_autodl_init.sh --model-size 2B
#   bash 00_autodl_init.sh --model-size 1.7-3B
#
#   # 自定义路径 (AutoDL 数据盘场景)
#   bash 00_autodl_init.sh --gr00t-repo /root/Isaac-GR00T \
#                           --models-dir /root/models
#
#   # 跳过基础模型下载 (本地已有 FP16 模型, 想直接 LoRA)
#   bash 00_autodl_init.sh --skip-base-model
#
#   # 从本地 tarball 解压 (含 .git, 跳过 GitHub clone, 推荐国内场景)
#   bash 00_autodl_init.sh --from-tarball /root/Isaac-GR00T.tar.gz
#   bash 00_autodl_init.sh --from-tarball ~/Isaac-GR00T.tar.gz
#
#   # 镜像源 (国内加速)
#   bash 00_autodl_init.sh --pip-mirror tsinghua
#
# 重复运行安全: 已存在的组件会自动跳过
# ============================================================================
set -euo pipefail

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
log()   { echo -e "${CYAN}[·]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
hr()    { echo -e "${CYAN}──────────────────────────────────────────────────────────────${NC}"; }

# ── 默认参数 ────────────────────────────────────────────────────────────────
GR00T_REPO="/root/Isaac-GR00T"
MODELS_DIR="/root/models"
DATA_DIR="/root/data"
VENV_DIR="$GR00T_REPO/.venv"             # uv sync 默认输出到 .venv
MODEL_SIZE="1.7-3B"
GR00T_REF="main"
SKIP_BASE_MODEL=false
SKIP_GIT_LFS=false
HF_ENDPOINT="https://huggingface.co"     # 国内可改为 https://hf-mirror.com
PIP_INDEX=""                             # 国内可改为 https://pypi.tuna.tsinghua.edu.cn/simple
TARBALL_PATH=""                          # 本地 tarball 路径 (跳过 git clone)

# ── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gr00t-repo)        GR00T_REPO="$2";      VENV_DIR="$GR00T_REPO/.venv"; shift 2 ;;
        --models-dir)        MODELS_DIR="$2";      shift 2 ;;
        --data-dir)          DATA_DIR="$2";        shift 2 ;;
        --model-size)        MODEL_SIZE="$2";      shift 2 ;;
        --gr00t-ref)         GR00T_REF="$2";       shift 2 ;;
        --skip-base-model)   SKIP_BASE_MODEL=true; shift   ;;
        --skip-git-lfs)      SKIP_GIT_LFS=true;    shift   ;;
        --from-tarball)      TARBALL_PATH="$2";    shift 2 ;;
        --hf-endpoint)       HF_ENDPOINT="$2";     shift 2 ;;
        --pip-mirror)
            case "$2" in
                tsinghua)  PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" ;;
                aliyun)    PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/" ;;
                douban)    PIP_INDEX="https://pypi.douban.com/simple/" ;;
                huawei)    PIP_INDEX="https://repo.huaweicloud.com/repository/pypi/simple" ;;
                *)         PIP_INDEX="$2" ;;
            esac
            shift 2
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) fail "未知参数: $1 (--help 查看用法)" ;;
    esac
done

BASE_MODEL_DIR="$MODELS_DIR/GR00T-N1-${MODEL_SIZE}"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  GR00T × AutoDL 云端 — 第 0 步: 环境一键初始化                     ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Isaac-GR00T:   $GR00T_REPO  (ref: $GR00T_REF)"
info "基础模型:       GR00T-N1-${MODEL_SIZE}  → $BASE_MODEL_DIR"
info "训练数据目录:   $DATA_DIR"
info "Python venv:    $VENV_DIR"
info "HF endpoint:    $HF_ENDPOINT"
[ -n "$PIP_INDEX" ] && info "Pip 镜像:       $PIP_INDEX"
echo ""

# ── 网络探测 (国内加速提示) ────────────────────────────────────────────────
step "检测网络环境..."
LATENCY_HF=$(curl -o /dev/null -s -w '%{time_connect}' --max-time 3 "$HF_ENDPOINT" 2>/dev/null || echo "999")
LATENCY_PYPI=$(curl -o /dev/null -s -w '%{time_connect}' --max-time 3 https://pypi.org 2>/dev/null || echo "999")
if (( $(echo "$LATENCY_HF > 1.0" | bc -l 2>/dev/null || echo 0) )); then
    warn "HuggingFace 延迟 ${LATENCY_HF}s 偏高, 建议设置 --hf-endpoint https://hf-mirror.com"
fi
if (( $(echo "$LATENCY_PYPI > 1.0" | bc -l 2>/dev/null || echo 0) )); then
    warn "PyPI 延迟 ${LATENCY_PYPI}s 偏高, 建议设置 --pip-mirror aliyun"
fi
info "HF: ${LATENCY_HF}s | PyPI: ${LATENCY_PYPI}s"

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: 系统依赖
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 1/6: 系统依赖 (git / git-lfs / ffmpeg / build-essential)"
hr

if command -v apt-get &>/dev/null; then
    NEED_INSTALL=()
    command -v git &>/dev/null    || NEED_INSTALL+=(git)
    command -v git-lfs &>/dev/null && SKIP_GIT_LFS_INSTALLED=true || NEED_INSTALL+=(git-lfs)
    command -v ffmpeg &>/dev/null || NEED_INSTALL+=(ffmpeg)
    command -v gcc &>/dev/null    || NEED_INSTALL+=(build-essential)

    if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
        step "安装缺失依赖: ${NEED_INSTALL[*]}"
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${NEED_INSTALL[@]}"
    else
        info "系统依赖已完整"
    fi

    if [ "${SKIP_GIT_LFS:-false}" != "true" ] && [ -z "${SKIP_GIT_LFS_INSTALLED:-}" ]; then
        git lfs install --system || git lfs install
        info "git-lfs 已初始化"
    fi
elif command -v yum &>/dev/null; then
    warn "检测到 RHEL/CentOS, 请确保 git / git-lfs / ffmpeg / gcc 已安装"
else
    warn "未识别包管理器, 假设依赖已安装"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: GPU + PyTorch + CUDA 验证
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 2/6: GPU + PyTorch + CUDA 验证"
hr

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    info "GPU: $GPU_NAME ($GPU_MEM, driver $GPU_DRIVER)"
else
    fail "未检测到 nvidia-smi, 请选择 PyTorch + CUDA 镜像"
fi

PYTHON_BIN=$(command -v python3 || command -v python)
[ -z "$PYTHON_BIN" ] && fail "未找到 python3"
PY_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python: $PY_VERSION  ($PYTHON_BIN)"

# 检查 torch + CUDA
if $PYTHON_BIN -c 'import torch' 2>/dev/null; then
    TORCH_INFO=$($PYTHON_BIN -c 'import torch; print(f"{torch.__version__} | CUDA {torch.version.cuda} | avail={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")')
    info "PyTorch: $TORCH_INFO"
    if ! $PYTHON_BIN -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
        fail "PyTorch 已装但 CUDA 不可用, 请选择正确的 PyTorch + CUDA 镜像"
    fi
else
    fail "未找到 PyTorch, 请选择 PyTorch 镜像 (实例市场搜索 'PyTorch 2.')"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: uv 包管理器
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 3/6: uv 包管理器"
hr

if command -v uv &>/dev/null; then
    info "uv 已安装: $(uv --version)"
else
    step "安装 uv..."
    if [ -n "$PIP_INDEX" ]; then
        $PYTHON_BIN -m pip install -i "$PIP_INDEX" uv
    else
        $PYTHON_BIN -m pip install uv
    fi
    info "uv 已安装: $(uv --version)"
fi

# uv 配置 pip 镜像
if [ -n "$PIP_INDEX" ]; then
    uv pip config set global.index-url "$PIP_INDEX" 2>/dev/null || \
        mkdir -p ~/.config/uv && echo "[pip]\nindex-url = \"$PIP_INDEX\"" > ~/.config/uv/uv.toml
    info "uv pip 镜像已设为: $PIP_INDEX"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 4: Isaac-GR00T 仓库
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 4/6: Isaac-GR00T 仓库 (ref: $GR00T_REF)"
hr

mkdir -p "$(dirname "$GR00T_REPO")"

# ── 优先: 从本地 tarball 解压 (含 .git, 跳过 GitHub clone) ──────────────
if [ -n "$TARBALL_PATH" ]; then
    if [ ! -f "$TARBALL_PATH" ]; then
        fail "tarball 不存在: $TARBALL_PATH"
    fi
    step "从本地 tarball 解压: $TARBALL_PATH ($(du -h "$TARBALL_PATH" | cut -f1))..."
    # 解压到 GR00T_REPO 同级目录 (Isaac-GR00T/ 是 tarball 内根目录)
    rm -rf "$GR00T_REPO"
    tar -xzf "$TARBALL_PATH" -C "$(dirname "$GR00T_REPO")"
    cd "$GR00T_REPO"
    # 验证 git 仓库可用
    if ! git rev-parse --git-dir &>/dev/null; then
        fail "tarball 解压后 .git 不可用, 请检查 tarball 是否完整"
    fi
    log "当前 commit: $(git rev-parse --short HEAD) ($(git log -1 --pretty=%s | head -c 80))"
    info "Isaac-GR00T 已从 tarball 还原, 之后可用 git pull / git checkout 更新"

# ── 回退: 从 GitHub 克隆 ─────────────────────────────────────────────────
elif [ -d "$GR00T_REPO/.git" ]; then
    info "Isaac-GR00T 已存在: $GR00T_REPO"
    cd "$GR00T_REPO"
    git fetch --tags --quiet 2>/dev/null || warn "git fetch 失败 (网络问题), 继续"
    git checkout "$GR00T_REF" 2>/dev/null || warn "checkout $GR00T_REF 失败, 继续使用现有分支"
    log "当前 commit: $(git rev-parse --short HEAD)"
else
    step "克隆 Isaac-GR00T (含 submodules, 首次 ~5min)..."
    git clone --branch "$GR00T_REF" --recurse-submodules --shallow-since="6 months ago" \
        https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_REPO"
    cd "$GR00T_REPO"
    info "已克隆: $(git rev-parse --short HEAD)"
fi

# LFS pull (仅在 tarball 解压时跳过, 因为 tarball 可能未包含 LFS 大文件)
if [ "${SKIP_GIT_LFS:-false}" != "true" ] && [ -z "$TARBALL_PATH" ]; then
    cd "$GR00T_REPO"
    git lfs pull 2>&1 | tail -3 || warn "git lfs pull 失败, 继续 (大文件可能未下载)"
elif [ -n "$TARBALL_PATH" ]; then
    log "跳过 git lfs pull (tarball 模式, 训练不需要 LFS 大文件)"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 5: Python 3.10 venv + 训练栈
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 5/6: Python 3.10 venv + 训练栈 (transformers/peft/bnb/accelerate)"
hr

cd "$GR00T_REPO"

# 创建 / 重用 Python 3.10 venv
if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
    VENV_PY=$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$VENV_PY" = "3.10" ]; then
        info "Python 3.10 venv 已存在: $VENV_DIR"
    else
        warn "venv 是 Python $VENV_PY, Isaac-GR00T 要求 3.10, 重建中..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    step "创建 Python 3.10 venv..."
    # AutoDL PyTorch 镜像通常不含 3.10, 用 uv 自动下载 3.10
    if ! command -v python3.10 &>/dev/null; then
        uv python install 3.10
    fi
    uv venv --python 3.10 "$VENV_DIR"
    info "venv 已创建: $VENV_DIR"
fi

# 升级基础工具
uv pip install --python "$VENV_DIR/bin/python" --upgrade pip setuptools wheel 2>&1 | tail -3

# 安装 Isaac-GR00T + 训练栈
step "安装 Isaac-GR00T (editable)..."
uv pip install --python "$VENV_DIR/bin/python" -e . 2>&1 | tail -5

step "安装训练栈 (transformers / peft / bitsandbytes / accelerate)..."
uv pip install --python "$VENV_DIR/bin/python" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "peft>=0.11.0" \
    "bitsandbytes>=0.43.0" \
    safetensors einops timm scipy \
    "huggingface_hub>=0.24.0" 2>&1 | tail -5

# ════════════════════════════════════════════════════════════════════════════
# Phase 6: GR00T-N1 基础模型下载 (可选)
# ════════════════════════════════════════════════════════════════════════════
hr
step "Phase 6/6: GR00T-N1-${MODEL_SIZE} 基础模型"
hr

mkdir -p "$MODELS_DIR"

if $SKIP_BASE_MODEL; then
    info "跳过基础模型下载 (--skip-base-model)"
elif [ -d "$BASE_MODEL_DIR" ] && [ "$(ls -A "$BASE_MODEL_DIR" 2>/dev/null)" ]; then
    info "基础模型已存在: $BASE_MODEL_DIR ($(du -sh "$BASE_MODEL_DIR" | cut -f1))"
else
    step "下载 GR00T-N1-${MODEL_SIZE} (~7GB)..."
    HF_ENDPOINT="$HF_ENDPOINT" uv pip install --python "$VENV_DIR/bin/python" \
        "huggingface_hub>=0.24.0" 2>&1 | tail -2 || true

    HF_ENDPOINT="$HF_ENDPOINT" "$VENV_DIR/bin/huggingface-cli" download \
        "nvidia/GR00T-N1.${MODEL_SIZE}" \
        --local-dir "$BASE_MODEL_DIR" \
        --exclude "*.msgpack" "*.onnx" 2>&1 | tail -10
    info "基础模型已下载到: $BASE_MODEL_DIR"
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 7: 环境验证
# ════════════════════════════════════════════════════════════════════════════
hr
step "环境验证 (Python 3.10 + GR00T + 训练栈)"
hr

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

set +e
VERIFY_LOG=$(python3 -c '
import sys
checks = []

def chk(name, cond, detail=""):
    checks.append((name, cond, detail))

# Python 版本
chk("Python 3.10", sys.version_info[:2] == (3, 10), f"{sys.version_info.major}.{sys.version_info.minor}")

# torch + CUDA
try:
    import torch
    chk("torch", True, f"{torch.__version__} CUDA {torch.version.cuda}")
    chk("CUDA 可用", torch.cuda.is_available())
    if torch.cuda.is_available():
        chk("GPU 数量", torch.cuda.device_count() >= 1, f"{torch.cuda.device_count()}")
        chk("GPU 名称", True, torch.cuda.get_device_name(0))
        # 简单算力测试
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.matmul(x, x)
        chk("GPU 计算", True, "matmul(1024x1024) OK")
except Exception as e:
    chk("torch", False, str(e))

# 训练栈
for mod, name in [("transformers", "transformers"),
                  ("accelerate", "accelerate"),
                  ("peft", "peft"),
                  ("bitsandbytes", "bitsandbytes"),
                  ("safetensors", "safetensors"),
                  ("einops", "einops"),
                  ("timm", "timm"),
                  ("huggingface_hub", "huggingface_hub")]:
    try:
        m = __import__(mod)
        chk(name, True, getattr(m, "__version__", "?"))
    except Exception as e:
        chk(name, False, str(e))

# GR00T
try:
    import gr00t
    chk("gr00t", True, getattr(gr00t, "__version__", "?"))
except Exception as e:
    chk("gr00t", False, str(e))

# 打印结果
all_ok = True
for name, ok, detail in checks:
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name:<18} {detail}")
    if not ok:
        all_ok = False

print()
print("✅ 环境就绪!" if all_ok else "❌ 环境不完整, 请检查上方失败项")
sys.exit(0 if all_ok else 1)
' 2>&1)
VERIFY_RC=$?
set -e

echo "$VERIFY_LOG"

# ════════════════════════════════════════════════════════════════════════════
# 后续步骤提示
# ════════════════════════════════════════════════════════════════════════════
echo ""
hr
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ✅ 云端 GR00T 训练环境初始化完成                                    ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
hr
echo ""
info "Isaac-GR00T:        $GR00T_REPO  ($(cd "$GR00T_REPO" && git rev-parse --short HEAD))"
info "Python venv:        $VENV_DIR/bin/activate"
info "基础模型:           $BASE_MODEL_DIR  ($( [ -d "$BASE_MODEL_DIR" ] && du -sh "$BASE_MODEL_DIR" 2>/dev/null | cut -f1 || echo '未下载' ))"
echo ""
echo -e "${CYAN}📋 后续操作:${NC}"
echo ""
echo -e "${BLUE}  # 1. 上传本地训练数据 (在本地运行, 非云端)${NC}"
echo -e "     scp g1_gr00t_training.tar.gz root@<host>:/root/"
echo ""
echo -e "${BLUE}  # 2. 解压训练数据${NC}"
echo -e "     cd /root && tar -xzf g1_gr00t_training.tar.gz"
echo ""
echo -e "${BLUE}  # 3. 激活 venv 并启动训练${NC}"
echo -e "     source $VENV_DIR/bin/activate"
echo -e "     cd $GR00T_REPO"
echo -e "     python3 gr00t/experiment/launch_finetune.py \\"
echo -e "         --base-model-path $BASE_MODEL_DIR \\"
echo -e "         --dataset-path $DATA_DIR/g1_lerobot \\"
echo -e "         --output-dir $MODELS_DIR/g1_gr00t \\"
echo -e "         --num-epochs 10 --batch-size 2 --grad-accum 2 \\"
echo -e "         --learning-rate 1e-4 \\"
echo -e "         --embodiment-tag NEW_EMBODIMENT --use-lora"
echo ""
echo -e "${BLUE}  # 或: 直接调用 02_autodl_train.sh (默认跳过环境搭建)${NC}"
echo -e "     bash 02_autodl_train.sh --robot g1 --epochs 10 --skip-setup"
echo ""
hr
