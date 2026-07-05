#!/bin/bash
# ─── 云端环境初始化（一次性）—— 在 AutoDL 服务器上执行 ───
set -e

echo "🚀 GR00T MJLab AutoDL — 云端环境初始化"
echo "======================================"

echo ""
echo "📦 [1/5] 安装系统依赖..."
sudo apt-get update && sudo apt-get install -y ffmpeg git-lfs tmux
git lfs install
echo "✅ 系统依赖安装完成"

echo ""
echo "📥 [2/5] 安装 uv 包管理器..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    echo "✅ uv 安装完成"
else
    echo "✅ uv 已存在，跳过"
fi

echo ""
echo "📂 [3/5] 克隆 Isaac-GR00T..."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ISAAC_DIR="$PARENT_DIR/Isaac-GR00T"

if [ ! -d "$ISAAC_DIR" ]; then
    cd "$PARENT_DIR"
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
    echo "✅ Isaac-GR00T 克隆完成"
else
    echo "✅ Isaac-GR00T 已存在，跳过"
fi

echo ""
echo "🐍 [4/5] 创建 Python 环境..."
cd "$ISAAC_DIR"
uv sync --python 3.10
echo "✅ Python 环境创建完成"

echo ""
echo "======================================"
echo "✅ 云端环境初始化完成！"
echo "   项目路径: $SCRIPT_DIR"
echo "   Isaac-GR00T: $ISAAC_DIR"
echo "   下一步: bash $SCRIPT_DIR/scripts/01_start_server.sh"
