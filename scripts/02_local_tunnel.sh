#!/bin/bash
# ─── 本地建立 SSH 隧道 — 在本地电脑上执行（保持运行） ───
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$SCRIPT_DIR/config/ssh_config.sh"

if [ ! -f "$CONFIG" ]; then
    echo "❌ 未找到配置文件: $CONFIG"
    echo "   请编辑 config/ssh_config.sh，填写你的 AutoDL 信息"
    exit 1
fi

source "$CONFIG"

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ "$SSH_HOST" = "region-xx.autodl.pro" ]; then
    echo "❌ SSH 配置未填写完整"
    echo "   请编辑 config/ssh_config.sh，设置 SSH_HOST, SSH_PORT, SSH_USER"
    exit 1
fi

echo "🔒 建立 SSH 隧道"
echo "   本地端口: $LOCAL_PORT → 远程端口: $SERVER_PORT"
echo "   远程地址: ${SSH_USER}@${SSH_HOST}:${SSH_PORT}"
echo "   (Ctrl+C 退出)"
echo ""

ssh -N -L ${LOCAL_PORT}:localhost:${SERVER_PORT} \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=6 \
    -o ExitOnForwardFailure=yes \
    -p ${SSH_PORT} \
    ${SSH_USER}@${SSH_HOST}

echo "🔌 SSH 隧道已关闭"
