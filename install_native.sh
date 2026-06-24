#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab 本机安装脚本 — 直接在当前主机安装完整环境
#
# 适用场景:
#   - 个人开发机/服务器, 想要本机跑完整流程 (无容器)
#   - 想直接复用系统的 Python (3.10 / 3.11 / 3.12) + venv
#   - 想要更快的迭代 (改代码无需重新构建镜像)
#
# 用法:
#   ./install_native.sh                     # 交互式 (推荐)
#   ./install_native.sh collect             # 只装基础 + 采集依赖
#   ./install_native.sh infer               # 基础 + 采集 + 推理
#   ./install_native.sh all                 # 同 infer
#   ./install_native.sh --mirror cn all     # 国内镜像源
#   ./install_native.sh --no-apt collect    # 跳过 apt (假定依赖已装)
#   ./install_native.sh --recreate all      # 强制重建 venv
#
# 关键设计:
#   - 使用系统 Python (3.10 / 3.11 / 3.12), 不强制装 3.12
#     (mjlab 1.2.0 的 Requires-Python: >=3.10, <3.14)
#   - venv 放在 ./.venv (项目目录内, 符合 Python 惯例, 已在 .gitignore)
#   - PyTorch wheel 根据 GPU 架构自动选择:
#       sm_70/75/80/86/89/90 → torch==2.11.0+cu128
#       sm_120 (Blackwell)   → torch==2.11.0+cu128
#   - 不修改用户的 ~/.bashrc, 使用 `source .venv/bin/activate` 手动激活
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RL_MJLAB_ROOT="$(cd "$PROJECT_ROOT/../unitree_rl_mjlab" && pwd)"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
banner(){ echo -e "\n${BOLD}${BLUE}$1${NC}"; }

# ── 默认参数 ─────────────────────────────────────────────────────────────
INSTALL_MODE=""          # collect | infer | all
MIRROR=""                # "" = official, "cn" = 国内 (aliyun)
NO_APT=false             # 跳过 apt install
RECREATE_VENV=false      # 强制重建 venv
PYTHON_VERSION="3.12"    # 优先 python3.12 (mjlab 1.2.0 要求 >=3.10 <3.14)

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_BIN="$VENV_DIR/bin"

# 统一的版本常量 (与训练环境保持一致)
PYTORCH_VERSION="2.11.0+cu128"
TORCHVISION_VERSION="0.26.0+cu128"
TORCHAUDIO_VERSION="2.11.0+cu128"
TRITON_VERSION="3.6.0"
MUJOCO_RANGE=">=3.5.0,<3.6.0"
WARP_RANGE=">=1.12.0,<1.13.0"
MJLAB_VERSION="1.2.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"

# ── 参数解析 ────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
用法: $0 [选项] <collect|infer|all>

选项:
  --mirror <cn|official>  镜像源 (默认 official)
                          cn: 使用阿里云 (apt + pip)
  --no-apt               跳过 apt install (假定系统依赖已装)
  --recreate             强制重建 .venv (删除现有)
  --python <ver>         Python 版本偏好 (3.10|3.11|3.12), 默认 3.10
  -h, --help             显示本帮助

安装模式:
  collect                基础 + 采集依赖 (~6GB)
  infer                  基础 + 采集 + 推理 (~9GB)
  all                    同 infer

示例:
  $0                                # 交互式
  $0 collect                        # 最快安装 (只要采集功能)
  $0 --mirror cn all                # 国内用户完整安装
  $0 --no-apt infer                 # 跳过 apt, 假定环境已就绪
  $0 --recreate all                 # 强制重建 venv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        collect|infer|all)
            INSTALL_MODE="$1"; shift ;;
        --mirror)
            MIRROR="$2"; shift 2 ;;
        --no-apt)
            NO_APT=true; shift ;;
        --recreate)
            RECREATE_VENV=true; shift ;;
        --python)
            PYTHON_VERSION="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            fail "未知参数: $1 (用 --help 查看用法)" ;;
    esac
done

# ── 交互式选择 (若未指定模式) ──────────────────────────────────────────
prompt_install_mode() {
    banner "选择安装模式"
    echo -e "${BOLD}请选择要安装的组件:${NC}"
    echo ""
    echo "  1) collect  — 基础 + 采集 (PyTorch + mjlab + 仿真 ~6GB)"
    echo "                适用于: 本地数据采集, 不需要 GR00T 推理"
    echo ""
    echo "  2) infer    — 基础 + 采集 + 推理 (PyTorch + mjlab + GR00T + transformers/peft/bnb ~9GB)"
    echo "                适用于: 完整本地流程 (采集 + 推理验证)"
    echo ""
    echo "  3) all      — 同 infer"
    echo ""
    while true; do
        read -p "请选择 [1-3, 默认 2]: " choice
        case "${choice:-2}" in
            1) INSTALL_MODE="collect"; return ;;
            2) INSTALL_MODE="infer"; return ;;
            3) INSTALL_MODE="all"; return ;;
            *) echo "无效选择, 请输入 1, 2 或 3" ;;
        esac
    done
}

prompt_mirror() {
    banner "选择镜像源"
    echo "  1) official — PyPI 官方 (国际网络)"
    echo "  2) cn       — 阿里云 (国内推荐)"
    echo ""
    read -p "请选择 [1-2, 默认 1]: " choice
    case "${choice:-1}" in
        1) MIRROR="" ;;
        2) MIRROR="cn" ;;
    esac
}

# ── 环境检测 ───────────────────────────────────────────────────────────
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
    else
        fail "无法识别操作系统 (缺少 /etc/os-release)"
    fi

    case "$OS_ID" in
        ubuntu)
            if [ "${OS_VERSION%%.*}" != "22" ] && [ "${OS_VERSION%%.*}" != "24" ]; then
                warn "Ubuntu 版本 $OS_VERSION 未在测试列表 (22.04 / 24.04), 可能可以工作但不保证"
            fi
            ;;
        *)
            warn "OS $OS_ID 未在测试列表 (Ubuntu 22.04 / 24.04), 可能可以工作但不保证"
            ;;
    esac
    info "OS: $PRETTY_NAME"
}

detect_gpu() {
    GPU_CC=""
    GPU_NAME=""
    if ! command -v nvidia-smi &>/dev/null; then
        warn "未找到 nvidia-smi, 跳过 GPU 检测 (PyTorch 将装 CPU 版本不可用, 建议安装 NVIDIA 驱动)"
        return 1
    fi

    if ! nvidia-smi &>/dev/null; then
        warn "nvidia-smi 调用失败, 可能驱动未正常工作"
        return 1
    fi

    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

    info "GPU: $GPU_NAME (compute_cap=$GPU_CC, driver $DRIVER_VER)"

    # CUDA 12.x 需要 Driver ≥ 525.60 (12.0) 或 ≥ 535 (12.2)
    # Driver 595 远高于要求, 完美兼容 cu128
    if [ -n "$DRIVER_VER" ]; then
        DRIVER_MAJOR="${DRIVER_VER%%.*}"
        if [ "$DRIVER_MAJOR" -lt 525 ] 2>/dev/null; then
            warn "Driver $DRIVER_VER 较旧 (建议 ≥ 525 支持 CUDA 12.x), PyTorch 可能无法使用 GPU"
        fi
    fi

    return 0
}

detect_python() {
    PYTHON_BIN=""

    # 按优先级查找
    local candidates=(
        "$VENV_BIN/python3"        # 现有 venv
        "/usr/bin/python${PYTHON_VERSION}"
        "/usr/bin/python3.12"
        "/usr/bin/python3.11"
        "/usr/bin/python3.10"
        "/usr/local/bin/python${PYTHON_VERSION}"
        "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11"
        "/usr/local/bin/python3.10"
        "$(command -v python3)"
        "$(command -v python)"
    )

    for c in "${candidates[@]}"; do
        if [ -x "$c" ]; then
            local ver
            ver=$("$c" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            if [ -n "$ver" ]; then
                local major="${ver%%.*}"
                local minor="${ver##*.}"
                # 要求 Python >= 3.10 且 < 3.14 (mjlab 要求)
                if [ "$major" = "3" ] && [ "$minor" -ge 10 ] && [ "$minor" -lt 14 ] 2>/dev/null; then
                    PYTHON_BIN="$c"
                    PYTHON_VERSION="$ver"
                    break
                fi
            fi
        fi
    done

    if [ -z "$PYTHON_BIN" ]; then
        fail "未找到可用的 Python 3.10+ (mjlab 需要 >= 3.10 < 3.14)"
    fi

    info "Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
}

# ── apt 安装系统依赖 ───────────────────────────────────────────────────
install_system_deps() {
    if [ "$NO_APT" = true ]; then
        info "跳过 apt (--no-apt)"
        return 0
    fi

    banner "[1/8] 安装系统依赖 (apt)"

    local pkgs=(
        # Python 与构建工具
        python3-dev python3-venv python3-pip
        python3-setuptools python3-wheel
        build-essential cmake ninja-build pkg-config
        # Git 与 LFS
        git git-lfs curl wget ca-certificates
        # OpenGL / 图形 (mujoco offscreen rendering 需要)
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
        libegl1-mesa libopengl0 libosmesa6
        # 网络工具
        net-tools iputils-ping
        # ffmpeg (录制视频)
        ffmpeg
    )

    # 选 python3.12-venv / dev (若系统已有该版本)
    case "$PYTHON_VERSION" in
        3.12)
            if apt-cache show python3.12-venv &>/dev/null 2>&1; then
                pkgs+=(python3.12 python3.12-dev python3.12-venv)
            fi
            ;;
        3.11)
            if apt-cache show python3.11-venv &>/dev/null 2>&1; then
                pkgs+=(python3.11 python3.11-dev python3.11-venv)
            fi
            ;;
        # 3.10 默认已装, 不需要单独加
    esac

    # 镜像源
    if [ "$MIRROR" = "cn" ]; then
        info "使用阿里云 apt 镜像"
        sudo sed -i.bak 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g; s|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list
    fi

    info "更新 apt 索引..."
    sudo apt-get update

    info "安装 ${#pkgs[@]} 个包 (需要 sudo)..."
    sudo apt-get install -y --no-install-recommends "${pkgs[@]}"

    info "apt 安装完成"
}

# ── 创建 venv ──────────────────────────────────────────────────────────
create_venv() {
    banner "[2/8] 创建 Python 虚拟环境"

    if [ -d "$VENV_DIR" ] && [ "$RECREATE_VENV" = false ]; then
        info "检测到现有 venv: $VENV_DIR"
        # 验证现有 venv 可用
        if "$VENV_BIN/python" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            info "现有 venv 可用, 跳过重建 (用 --recreate 强制重建)"
            return 0
        fi
        warn "现有 venv 不可用, 将删除重建"
    fi

    if [ -d "$VENV_DIR" ]; then
        info "删除旧 venv..."
        rm -rf "$VENV_DIR"
    fi

    info "创建 venv: $VENV_DIR"
    info "使用 Python: $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    # 关键: 不要升级 setuptools 到 82+
    # PyTorch 2.11.0+cu128 requires setuptools<82
    # mjlab 1.2.0 的某些依赖也要求 setuptools<82
    # 升到 82.0.1 会导致后续 pip install -e . 失败
    info "升级 pip + wheel (保持 setuptools<82)..."
    "$VENV_BIN/pip" install --upgrade "pip>=23" "wheel" "setuptools<82"

    info "venv 创建完成"
}

# ── 配置 pip 镜像源 ───────────────────────────────────────────────────
setup_pip_mirror() {
    if [ "$MIRROR" = "cn" ]; then
        info "配置 pip 镜像源: 阿里云"
        "$VENV_BIN/pip" config set global.index-url "https://mirrors.aliyun.com/pypi/simple/"
        "$VENV_BIN/pip" config set global.trusted-host "mirrors.aliyun.com"
    else
        # 清除可能的旧设置
        "$VENV_BIN/pip" config unset global.index-url 2>/dev/null || true
        info "使用 PyPI 官方源"
    fi
}

# ── 安装 PyTorch + CUDA runtime ──────────────────────────────────────
install_pytorch() {
    banner "[3/8] 安装 PyTorch ($PYTORCH_VERSION)"

    # sm_75 (Turing, 如 RTX 2080) 在 cu128 wheel 中可工作
    # PyTorch cu128 最低支持 sm_70 (Volta), sm_75 完全可用
    if [ -n "$GPU_CC" ]; then
        info "GPU 架构 sm_$GPU_CC → 使用 cu128 wheel"
    else
        warn "未检测到 GPU, 仍安装 cu128 wheel (无 GPU 时 PyTorch 会自动 fallback 到 CPU)"
    fi

    # 3a. PyTorch 主包
    info "3a. PyTorch 主包 (~850MB)..."
    "$VENV_BIN/pip" install --no-cache-dir --no-deps \
        --index-url "$CUDA_INDEX" \
        "torch==$PYTORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION"

    # 3b. CUDA 运行时
    info "3b. CUDA 运行时 (cudnn/cublas/cufft/curand/cusparse/cusolver) (~2.5GB)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "nvidia-cudnn-cu12==9.19.0.56" \
        "nvidia-cublas-cu12==12.8.4.1" \
        "nvidia-cufft-cu12==11.3.3.83" \
        "nvidia-curand-cu12==10.3.9.90" \
        "nvidia-cusparse-cu12==12.5.8.93" \
        "nvidia-cusolver-cu12==11.7.3.90"

    info "3c. CUDA 通信 (nccl/nvshmem/nvjitlink/...) (~1GB)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "nvidia-nccl-cu12==2.28.9" \
        "nvidia-nvshmem-cu12==3.4.5" \
        "nvidia-nvjitlink-cu12==12.8.93" \
        "nvidia-cuda-nvrtc-cu12==12.8.93" \
        "nvidia-nvtx-cu12==12.8.90" \
        "nvidia-cuda-cupti-cu12==12.8.90" \
        "nvidia-cuda-runtime-cu12==12.8.90" \
        "nvidia-cufile-cu12==1.13.1.3" \
        "nvidia-cusparselt-cu12==0.7.1"

    info "3d. triton + cuda-bindings (~200MB)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "triton==$TRITON_VERSION" \
        "cuda-bindings==12.9.4" \
        "cuda-pathfinder==1.2.2" \
        "cuda-toolkit==12.8.1"

    # 验证 GPU 可用
    info "验证 PyTorch + GPU..."
    if "$VENV_BIN/python" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  Compute capability: {torch.cuda.get_device_capability(0)}')
    # 实测一个小张量
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'  GPU compute test: OK (sum={y.sum().item():.2f})')
" 2>&1 | grep -q "GPU compute test: OK"; then
        info "PyTorch + GPU 验证成功"
    else
        warn "PyTorch + GPU 验证失败, 但不影响安装, 请检查 CUDA driver / GPU 兼容性"
        "$VENV_BIN/python" -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()}')" 2>&1
    fi
}

# ── 安装 mujoco + warp ────────────────────────────────────────────────
install_mujoco_warp() {
    banner "[4/8] 安装 mujoco + warp-lang"

    # 注意: pip install 需要 "包名 + 版本" 的组合, 不能只给 version specifier
    info "mujoco 3.5.x..."
    "$VENV_BIN/pip" install --no-cache-dir "mujoco>=3.5.0,<3.6.0"

    info "warp-lang 1.12.x..."
    "$VENV_BIN/pip" install --no-cache-dir "warp-lang>=1.12.0,<1.13.0"

    # 验证
    info "验证 mujoco + warp..."
    "$VENV_BIN/python" -c "
import mujoco
import warp as wp
print(f'  mujoco: {mujoco.__version__}')
print(f'  warp: {wp.__version__}')
print(f'  warp devices: {wp.get_devices()}')
# 简单测试
wp.init()
print(f'  warp.init(): OK')
" || warn "mujoco/warp 验证有警告, 不一定致命"
}

# ── 安装 mjlab (editable from unitree_rl_mjlab) ──────────────────────
install_mjlab() {
    banner "[5/8] 安装 mjlab (editable from ../unitree_rl_mjlab)"

    if [ ! -d "$RL_MJLAB_ROOT" ]; then
        fail "未找到兄弟项目: $RL_MJLAB_ROOT (期望 ../unitree_rl_mjlab 存在)"
    fi
    if [ ! -f "$RL_MJLAB_ROOT/setup.py" ]; then
        fail "$RL_MJLAB_ROOT/setup.py 不存在"
    fi

    info "RL_MJLAB_ROOT: $RL_MJLAB_ROOT"

    # 关键: mjlab editable 安装 (含 90+ 依赖)
    info "pip install -e . (mjlab + 90+ 依赖, 约 1GB, 可能需要 5-10 分钟)..."
    (cd "$RL_MJLAB_ROOT" && "$VENV_BIN/pip" install --no-cache-dir -e .)

    # 1d. mjlab 漏装的依赖 (scipy)
    info "安装 scipy (mjlab.terrains 需要)..."
    "$VENV_BIN/pip" install --no-cache-dir "scipy>=1.11.0"

    # 1e. 安装 mediapy 并修复 numpy 2.x 兼容性问题
    # mediapy 1.2.6 使用 npt.NDArray (numpy 1.x API), numpy 2.x 已移除该 API
    info "安装 mediapy + 修复 numpy 2.x 兼容性..."
    "$VENV_BIN/pip" install --no-cache-dir "mediapy>=1.2.0"
    # 查找 mediapy __init__.py 路径
    local mediapy_init
    mediapy_init=$("$VENV_BIN/python" -c "import mediapy; print(mediapy.__file__)" 2>/dev/null)
    if [ -n "$mediapy_init" ] && [ -f "$mediapy_init" ]; then
        # 修复: class _VideoArray(npt.NDArray[Any]) → class _VideoArray(np.ndarray)
        if grep -q "npt.NDArray" "$mediapy_init" 2>/dev/null; then
            info "修补 mediapy _VideoArray (numpy 2.x 兼容性)..."
            sed -i 's/class _VideoArray(npt\.NDArray\[Any\])/class _VideoArray(np.ndarray)/' "$mediapy_init"
            # 确保 import numpy as np 存在 (通常已有)
            info "mediapy 修补完成: $mediapy_init"
        else
            info "mediapy 无需修补 (已兼容 numpy 2.x)"
        fi
    else
        warn "未找到 mediapy __init__.py, 跳过修补"
    fi

    # 验证
    info "验证 mjlab..."
    if "$VENV_BIN/python" -c "
import mjlab
import mujoco
print(f'  mjlab: {getattr(mjlab, \"__version__\", \"unknown\")}')
print(f'  mujoco: {mujoco.__version__}')
" 2>&1 | grep -q "mujoco:"; then
        info "mjlab 验证成功"
    else
        warn "mjlab 验证有警告, 但已安装"
    fi
}

# ── 安装数据依赖 ───────────────────────────────────────────────────────
install_data_deps() {
    banner "[6/8] 安装数据依赖 (采集 + 转换)"

    "$VENV_BIN/pip" install --no-cache-dir \
        "pandas>=2.0.0" \
        "pyarrow>=14.0.0" \
        "tqdm" \
        "huggingface-hub>=0.24.0"

    info "数据依赖安装完成"
}

# ── 安装推理依赖 (infer 模式) ─────────────────────────────────────────
install_infer_deps() {
    banner "[7/8] 安装推理依赖 (GR00T + transformers/peft/bnb)"

    info "GR00T 基础依赖..."
    "$VENV_BIN/pip" install --no-cache-dir \
        "transformers>=4.45.0" \
        "accelerate>=0.34.0" \
        "peft>=0.11.0" \
        "bitsandbytes>=0.43.0"

    # GR00T 源码 — 从 PyPI (gr00t) 或 git clone
    info "Isaac-GR00T..."
    if [ -d "$PROJECT_ROOT/../Isaac-GR00T" ]; then
        info "检测到本地 Isaac-GR00T/ 仓库, 用 editable install"
        (cd "$PROJECT_ROOT/../Isaac-GR00T" && "$VENV_BIN/pip" install --no-cache-dir -e .) \
            || warn "Isaac-GR00T editable install 失败, 请手动安装"
    else
        info "从 GitHub 克隆 Isaac-GR00T..."
        GR00T_DIR="$PROJECT_ROOT/../Isaac-GR00T"
        if [ ! -d "$GR00T_DIR" ]; then
            git clone --depth=1 https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_DIR" \
                || warn "Isaac-GR00T 克隆失败, 请手动安装"
        fi
        if [ -d "$GR00T_DIR" ]; then
            (cd "$GR00T_DIR" && "$VENV_BIN/pip" install --no-cache-dir -e .) \
                || warn "Isaac-GR00T editable install 失败"
        fi
    fi

    info "推理依赖安装完成"
}

# (历史遗留: native.sh 入口已合并到 start.sh, 此函数保留为空 stub)
create_native_entrypoint() {
    :  # no-op
}

# ── 健康检查 ─────────────────────────────────────────────────────────
health_check() {
    banner "健康检查"

    local errors=0

    echo ""
    echo -e "${BOLD}=== 环境摘要 ===${NC}"
    echo "  OS:        $PRETTY_NAME"
    echo "  Python:    $PYTHON_BIN ($PYTHON_VERSION)"
    echo "  GPU:       ${GPU_NAME:-未检测到} (sm_$GPU_CC)"
    echo "  venv:      $VENV_DIR"
    echo "  模式:      $INSTALL_MODE"
    echo "  镜像:      ${MIRROR:-official}"
    echo ""

    "$VENV_BIN/python" -c "
import sys
checks = []
try:
    import torch
    checks.append(('torch', f'{torch.__version__} (CUDA={torch.cuda.is_available()})'))
except Exception as e:
    checks.append(('torch', f'FAIL: {e}'))

try:
    import mujoco
    checks.append(('mujoco', mujoco.__version__))
except Exception as e:
    checks.append(('mujoco', f'FAIL: {e}'))

try:
    import warp as wp
    wp.init()
    checks.append(('warp-lang', wp.__version__))
except Exception as e:
    checks.append(('warp-lang', f'FAIL: {e}'))

try:
    import mjlab
    checks.append(('mjlab', getattr(mjlab, '__version__', 'installed')))
except Exception as e:
    checks.append(('mjlab', f'FAIL: {e}'))

try:
    import pandas
    checks.append(('pandas', pandas.__version__))
except Exception as e:
    checks.append(('pandas', f'FAIL: {e}'))

try:
    import pyarrow
    checks.append(('pyarrow', pyarrow.__version__))
except Exception as e:
    checks.append(('pyarrow', f'FAIL: {e}'))

if '$INSTALL_MODE' in ('infer', 'all'):
    try:
        import transformers
        checks.append(('transformers', transformers.__version__))
    except Exception as e:
        checks.append(('transformers', f'FAIL: {e}'))
    try:
        import peft
        checks.append(('peft', peft.__version__))
    except Exception as e:
        checks.append(('peft', f'FAIL: {e}'))
    try:
        import bitsandbytes
        checks.append(('bitsandbytes', bitsandbytes.__version__))
    except Exception as e:
        checks.append(('bitsandbytes', f'FAIL: {e}'))

for name, ver in checks:
    status = '\033[0;32mOK\033[0m   ' if not ver.startswith('FAIL') else '\033[0;31mFAIL\033[0m '
    print(f'  {status}{name:20s} {ver}')
    if ver.startswith('FAIL'):
        global errors
        errors += 1
" || true

    echo ""
    if [ "$errors" -eq 0 ]; then
        info "健康检查通过 ✅"
    else
        warn "健康检查发现 $errors 个问题"
    fi
}

# ── 显示使用说明 ─────────────────────────────────────────────────────
print_usage() {
    cat <<EOF

${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════╗
║              Native 安装完成                                 ║
╚════════════════════════════════════════════════════════════════╝${NC}

${BOLD}下一步:${NC}

  1. 激活虚拟环境 (新开 shell 时需要):
     ${CYAN}source $VENV_DIR/bin/activate${NC}

  2. 执行流程 (统一入口 ./start.sh):
     ${CYAN}./start.sh${NC}                 # 交互式菜单
     ${CYAN}./start.sh 1${NC}               # 数据采集
     ${CYAN}./start.sh 1 --robot g1${NC}    # 采集 G1 数据
     ${CYAN}./start.sh 6${NC}               # 推理验证
     ${CYAN}./start.sh 6 --auto-quantize${NC}  # 自动量化 INT4 后推理

  3. 后续使用:

     ${CYAN}./start.sh 1${NC}                   # 步骤 1: 数据采集
     ${CYAN}./start.sh 6 --auto-quantize${NC}   # 步骤 6: 推理验证

${BOLD}注意事项:${NC}
  - venv 路径: ${CYAN}$VENV_DIR${NC} (已在 .gitignore, 无需担心 commit)
  - PyTorch:   $PYTORCH_VERSION (cu128, sm_$GPU_CC)
  - 镜像源:    ${MIRROR:-official}
  - 如需重装:  ${CYAN}./install_native.sh --recreate $INSTALL_MODE${NC}

${BOLD}常见问题:${NC}
  Q: 推理时 OOM (RTX 2080 8GB)?
  A: 使用 ./start.sh 6 --auto-quantize (自动 BF16 → INT4 量化)

  Q: 想换 PyTorch 版本?
  A: 编辑脚本顶部的 PYTORCH_VERSION 变量, 或用 conda 装老版

  Q: 想升级某个包?
  A: source .venv/bin/activate && pip install -U <pkg>
EOF
}

# ── 主流程 ───────────────────────────────────────────────────────────
main() {
    banner "gr00t-mjlab 本机安装"

    # 检测环境
    step "检测主机环境..."
    detect_os
    detect_gpu || true
    detect_python

    # 交互式选择
    if [ -z "$INSTALL_MODE" ]; then
        prompt_install_mode
    fi
    if [ -z "$MIRROR" ]; then
        # 仅在 apt/pip 操作前询问
        if [ "$NO_APT" = false ]; then
            prompt_mirror
        fi
    fi

    echo ""
    echo -e "${BOLD}安装计划:${NC}"
    echo "  模式:    $INSTALL_MODE"
    echo "  镜像:    ${MIRROR:-official}"
    echo "  Python:  $PYTHON_VERSION ($PYTHON_BIN)"
    echo "  GPU:     sm_${GPU_CC:-none}"
    echo "  apt:     $([ "$NO_APT" = true ] && echo "跳过" || echo "执行")"
    echo "  venv:    $([ "$RECREATE_VENV" = true ] && echo "重建" || echo "复用或新建")"
    echo ""
    read -p "继续? [Y/n]: " confirm
    case "${confirm:-Y}" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "已取消"; exit 0 ;;
    esac

    install_system_deps
    create_venv
    setup_pip_mirror

    install_pytorch
    install_mujoco_warp
    install_mjlab
    install_data_deps

    if [ "$INSTALL_MODE" = "infer" ] || [ "$INSTALL_MODE" = "all" ]; then
        install_infer_deps
    fi

    create_native_entrypoint

    health_check
    print_usage
}

main "$@"
