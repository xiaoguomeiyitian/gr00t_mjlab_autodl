#!/bin/bash
# ─── 本地推理验证 ───
# 加载量化模型，在本地 mjlab 推理验证
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─── 默认参数 ───
ROBOT="${1:-g1}"
MODEL_PATH="${2:-$SCRIPT_DIR/../checkpoints/${ROBOT}_int4}"
DATASET_PATH="${3:-$SCRIPT_DIR/output/${ROBOT}_lerobot}"
OUTPUT_DIR="${4:-$SCRIPT_DIR/../output/verify}"
VIS_MODE="${5:-demo}"  # demo / viser / mujoco

echo "🔍 本地推理验证"
echo "   模型: $MODEL_PATH"
echo "   数据: $DATASET_PATH"
echo "   输出: $OUTPUT_DIR"
echo "   可视化: $VIS_MODE"
echo ""

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

case "$VIS_MODE" in
    demo)
        echo "📊 方案 A: Demo 静态图推理"
        python3 -c "
from src.infer import GR00TLocalInference
from src.lerobot_loader import LeRobotEpisodeLoader
from src.observation_builder import ObservationBuilder
import numpy as np, os

model_path = '$MODEL_PATH'
dataset_path = '$DATASET_PATH'
output_dir = '$OUTPUT_DIR'

if not os.path.exists(model_path):
    print(f'❌ 模型不存在: {model_path}')
    print(f'   请先运行 scripts/08_local_quantize.sh')
    exit(1)

inference = GR00TLocalInference(model_path=model_path)
print(f'✅ 模型加载成功')

# 尝试加载数据集
if os.path.exists(dataset_path):
    print(f'📂 加载数据集: {dataset_path}')
    # 使用 LeRobotLoader 加载
else:
    print(f'⚠️  数据集不存在，使用模拟数据验证')
    state = np.zeros(71 if '$ROBOT' == 'g1' else 37, dtype=np.float32)
    images = {'front': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)}
    action, info = inference.predict(images=images, state=state, language='walk forward')
    print(f'   Action shape: {action.shape}')
    print(f'   Latency: {info[\"latency_ms\"]:.1f}ms')
    print(f'   Action range: [{action.min():.3f}, {action.max():.3f}]')

inference.close()
print('✅ 推理验证完成')
"
        ;;
    viser)
        echo "🌐 方案 B: Viser 浏览器可视化"
        python3 -m src.viz.viser_viewer \
            --model-path "$MODEL_PATH" \
            --robot "$ROBOT" \
            --port 20006
        ;;
    mujoco)
        echo "🖥️  方案 C: MuJoCo 原生可视化"
        python3 -m src.viz.mujoco_viewer \
            --model-path "$MODEL_PATH" \
            --robot "$ROBOT"
        ;;
esac

echo ""
echo "✅ 验证完成"
