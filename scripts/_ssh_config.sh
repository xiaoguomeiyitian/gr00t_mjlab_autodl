#!/usr/bin/env bash
# SSH 连接参数 — 填写你的 AutoDL 云端信息
# 优先顺序: 命令行参数 > 环境变量 > 本文件默认值

# ── 必填 ───────────────────────────────────────────────────────────────────
SSH_HOST=""           # ← AutoDL SSH 地址, 例: "root@region-9.xxxx.autodl.com"
SSH_PORT=""           # ← AutoDL SSH 端口 (在 AutoDL 控制台 "自定义服务" 查看)
SSH_USER=""           # ← SSH 用户名 (AutoDL 通常是 root)

# ── 认证 (二选一, 优先用密钥) ──────────────────────────────────────────────
SSH_KEY=""            # ← SSH 私钥路径, 例: "$HOME/.ssh/id_rsa_autodl"
                      #    留空 = 走密码认证 (每次会提示输入密码)
                      #    注意: 不支持明文密码, 请使用 ssh-agent 或密钥认证

# ── 云端路径 (一般不用改) ──────────────────────────────────────────────────
REMOTE_DIR="/root/workspace"        # 云端工作目录 (训练包 + 模型都放这里)
CONDA_ENV=""                        # 云端 Python 虚拟环境名 (留空用 base / system)

# ── 上传选项 ───────────────────────────────────────────────────────────────
SCP_BANDWIDTH_LIMIT=""              # 限速, 例: "100M" (留空不限)
                                    #   网络不稳时建议 "50M", 减少断连概率
# ── SSH 安全选项 ──────────────────────────────────────────────────────
# 默认禁用 StrictHostKeyChecking (自动化场景), 但可通过环境变量覆盖
# 安全建议: 生产环境设置 SSH_STRICT=yes 启用主机密钥验证
if [ "${SSH_STRICT:-}" = "yes" ]; then
    STRICT_OPT="-o StrictHostKeyChecking=yes"
else
    STRICT_OPT="-o StrictHostKeyChecking=no"
fi
SSH_OPTS="$STRICT_OPT -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
                                    # 保活参数 (每 30s 心跳, 6 次失败重连)

# 示例:
#   SSH_HOST="root@region-9.autodl.com"
#   SSH_PORT="32451"
#   SSH_KEY="$HOME/.ssh/id_rsa_autodl"
#   REMOTE_DIR="/root/workspace"

# ── 环境变量覆盖 (CI 场景) ─────────────────────────────────────────────────
SSH_HOST="${AUTODL_SSH_HOST:-$SSH_HOST}"
SSH_PORT="${AUTODL_SSH_PORT:-$SSH_PORT}"
SSH_USER="${AUTODL_SSH_USER:-$SSH_USER}"
SSH_KEY="${AUTODL_SSH_KEY:-$SSH_KEY}"
SSH_PASS="${AUTODL_SSH_PASS:-$SSH_PASS}"
REMOTE_DIR="${AUTODL_REMOTE_DIR:-$REMOTE_DIR}"

# ── 校验 ───────────────────────────────────────────────────────────────────
_ssh_config_warn() {
    echo -e "\033[1;33m[⚠️  SSH 配置未完成] 请在 scripts/_ssh_config.sh 顶部填写:\033[0m" >&2
    echo "    SSH_HOST=\"root@your-host.com\"    # ← 改成你的 AutoDL 主机" >&2
    echo "    SSH_PORT=\"12345\"                  # ← 改成你的 AutoDL 端口" >&2
    if [ -z "$SSH_KEY" ]; then
        echo "    SSH_KEY=\"\\\$HOME/.ssh/id_rsa\"    # ← 推荐用密钥" >&2
    fi
}

# 导出所有变量供其他脚本使用
export SSH_HOST SSH_PORT SSH_USER SSH_KEY SSH_PASS REMOTE_DIR CONDA_ENV
export SCP_BANDWIDTH_LIMIT SSH_OPTS
