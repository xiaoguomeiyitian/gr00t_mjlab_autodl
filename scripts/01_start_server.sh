#!/bin/bash
# ─── 云端启动 Policy Server — 在 AutoDL 服务器上执行 ───
set -e

MODEL_PATH="${1:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${2:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
PORT="${3:-5555}"
HOST="0.0.0.0"

echo "🚀 启动 GR00T Policy Server"
echo "   模型: $MODEL_PATH"
echo "   具身: $EMBODIMENT_TAG"
echo "   地址: $HOST:$PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_DIR="$SCRIPT_DIR/../Isaac-GR00T"

if [ ! -d "$ISAAC_DIR" ]; then
    echo "❌ 未找到 Isaac-GR00T: $ISAAC_DIR"
    echo "   请先执行: bash $SCRIPT_DIR/scripts/00_autodl_init.sh"
    exit 1
fi

cd "$ISAAC_DIR"

if tmux has-session -t gr00t 2>/dev/null; then
    echo "⚠️  tmux session 'gr00t' 已存在，停止旧会话"
    tmux kill-session -t gr00t
fi

echo ""
echo "🔧 启动 Policy Server (tmux: gr00t)..."
tmux new-session -d -s gr00t \
    "uv run python gr00t/eval/run_gr00t_server.py \
        --model-path $MODEL_PATH \
        --embodiment-tag $EMBODIMENT_TAG \
        --device cuda:0 \
        --host $HOST \
        --port $PORT" 2>&1

sleep 5

if tmux has-session -t gr00t 2>/dev/null; then
    echo ""
    echo "✅ Policy Server 已启动！"
    echo "   查看日志: tmux attach -t gr00t"
    echo "   停止服务: tmux kill-session -t gr00t"
    echo "   本地隧道: ssh -N -L ${PORT}:localhost:${PORT} root@<AutoDL地址> -p <SSH端口>"
else
    echo "❌ 启动失败，请检查日志"
    exit 1
fi
