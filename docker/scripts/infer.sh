#!/usr/bin/env bash
# ============================================================================
# 容器内: GR00T 推理验证入口
# ============================================================================
set -euo pipefail

cd "${GR00T_MJLAB_ROOT:-/root/gr00t_mjlab_autodl}"

ROBOT="${ROBOT:-g1}"
MODEL_PATH="${MODEL_PATH:-/root/models/${ROBOT}_gr00t_int4}"
INSTRUCTION="${INSTRUCTION:-walk forward}"
QUANTIZE="${QUANTIZE:-auto}"
EPISODES="${EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"

echo "=========================================="
echo " [4] 推理验证 (本地加载模型 + mjlab 回放)"
echo "=========================================="
echo "机器人:  ${ROBOT}"
echo "模型:    ${MODEL_PATH}"
echo "指令:    ${INSTRUCTION}"
echo "量化:    ${QUANTIZE}"
echo ""

# 自动检测模型量化方式
if [ "${QUANTIZE}" = "auto" ]; then
    if [[ "${MODEL_PATH}" == *int4* ]]; then
        QUANTIZE="4bit"
    elif [[ "${MODEL_PATH}" == *fp16* ]] || [[ "${MODEL_PATH}" == *full* ]]; then
        QUANTIZE="none"
    fi
fi

python3 src/infer.py \
    --robot "${ROBOT}" \
    --model-path "${MODEL_PATH}" \
    --instruction "${INSTRUCTION}" \
    --quantize "${QUANTIZE}" \
    --num-episodes "${EPISODES}" \
    --episode-length "${EPISODE_LENGTH}"
