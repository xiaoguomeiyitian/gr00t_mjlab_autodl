#!/usr/bin/env bash
# ============================================================================
# gr00t-mjlab 容器统一入口
# 用法:
#   entrypoint.sh collect [args...]   # 数据采集模式
#   entrypoint.sh train   [args...]   # 训练模式 (GR00T)
#   entrypoint.sh infer   [args...]   # 推理模式
#   entrypoint.sh shell              # 进入交互 shell
#   entrypoint.sh python [args...]   # 直接运行 python
# ============================================================================
set -euo pipefail

MODE="${1:-shell}"
shift || true

case "$MODE" in
    collect)
        exec /usr/local/bin/collect.sh "$@"
        ;;
    train)
        exec /usr/local/bin/train.sh "$@"
        ;;
    infer)
        exec /usr/local/bin/infer.sh "$@"
        ;;
    shell|bash)
        exec bash
        ;;
    python|python3|py)
        exec python3 "$@"
        ;;
    *)
        # 透传其他命令 (兼容直接 docker run ... bash -c "..." 模式)
        exec "$MODE" "$@"
        ;;
esac
