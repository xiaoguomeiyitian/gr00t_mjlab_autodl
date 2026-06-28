# GR00T MJLab AutoDL

NVIDIA Isaac-GR00T 的云端推理 + 本地训练编排兄弟项目。

## 定位

本项目**不修改** Isaac-GR00T 代码，通过 ZMQ 接口远程调用云端推理服务，并提供本地数据采集、INT4 量化、多种可视化方案。

```
本地（无 GPU）──SSH 隧道──► AutoDL 云端（GPU）
                                    │
                         Isaac-GR00T Policy Server / Fine-tune
```

## 端到端流程

```
云端: [0] 初始化 → [1] 启动 Server → [3] 微调训练
本地: [4] SSH 隧道 → [5] Demo 推理 → [6] 数据采集 → [7] 转换+上传
      [8] 下载模型 → [9] INT4 量化 → [10] 推理验证
可视化: [11] Viser + Policy Server 推理  [12] MuJoCo + Policy Server 推理
```

## 快速上手

### 1. 统一入口

```bash
./start.sh              # 交互式菜单（13 个功能）
./start.sh help         # 查看所有命令
```

非交互模式：

```bash
./start.sh init         云端环境初始化
./start.sh server       云端启动 Policy Server
./start.sh tunnel       本地建立 SSH 隧道
./start.sh demo         本地运行 Demo 推理
./start.sh auto         完整流程：tunnel → demo

./start.sh collect [robot] [num_episodes] [episode_length] [action_mode]
./start.sh upload [robot]
./start.sh train [robot]

./start.sh download [robot]
./start.sh quantize [robot]
./start.sh verify [robot] [vis_mode: demo|viser|mujoco]

./start.sh viser-infer [robot] [host] [port] [viser_port]
./start.sh mujoco-infer [robot] [host] [port]
```

### 2. Demo 推理

```bash
# 云端（AutoDL）
bash scripts/00_local_prepare_cache.sh                      # 本地准备 uv 缓存（一次性）
bash scripts/00_autodl_init.sh                              # 云端环境初始化
bash scripts/01_start_server.sh nvidia/GR00T-N1.7-3B ...    # 启动推理服务

# 本地
bash scripts/02_local_tunnel.sh                             # SSH 隧道
bash scripts/03_local_demo_eval.sh                          # Demo 推理
```

### 3. 数据采集 + 训练

```bash
# 本地
bash scripts/04_local_collect.sh g1 50 300 delta            # 采集 50 episodes
bash scripts/05_upload_to_autodl.sh g1                      # 转换格式 + 上传

# 云端（AutoDL）
bash scripts/06_autodl_train.sh g1                          # 微调训练
```

### 4. 量化 + 本地推理

```bash
bash scripts/07_download_model.sh g1                        # 下载模型
bash scripts/08_local_quantize.sh g1                        # INT4 量化
bash scripts/09_local_verify.sh g1 demo                     # 推理验证
```

### 5. 推理可视化（Viser / MuJoCo）

```bash
# 交互式
./start.sh
# 选择 11) Viser + Policy Server 推理可视化
# 选择 12) MuJoCo + Policy Server 推理可视化

# 非交互
./start.sh viser-infer g1 127.0.0.1 5555 20006
./start.sh mujoco-infer g1 127.0.0.1 5555
```

## 目录结构

```
gr00t_mjlab_autodl/
├── README.md
├── start.sh                           # 统一入口（交互/非交互 13 个功能）
├── config/
│   └── ssh_config.sh                  # SSH 配置（用户填写）
├── scripts/
│   ├── 00_local_prepare_cache.sh      # [本地] 准备 uv 缓存（一次性）
│   ├── 00_autodl_init.sh              # [云端] 环境初始化
│   ├── 01_start_server.sh             # [云端] 启动 Policy Server
│   ├── 02_local_tunnel.sh             # [本地] SSH 隧道
│   ├── 03_local_demo_eval.sh          # [本地] Demo 推理
│   ├── 04_local_collect.sh            # [本地] MJLab 数据采集
│   ├── 05_upload_to_autodl.sh         # [本地] 格式转换 + SCP 上传
│   ├── 06_autodl_train.sh             # [云端] 微调训练
│   ├── 07_download_model.sh           # [本地] 下载模型
│   ├── 08_local_quantize.sh           # [本地] INT4 量化
│   ├── 09_local_verify.sh             # [本地] 推理验证
├── src/                               # 源码（3955 行 Python）
│   ├── __init__.py
│   ├── policy_client.py               # 纯 ZMQ 客户端（不依赖 torch）
│   ├── observation_builder.py         # 观测格式构建
│   ├── demo_plotter.py                # 静态图渲染
│   ├── demo_eval.py                   # 推理主逻辑
│   ├── lerobot_loader.py              # LeRobot 数据加载器
│   ├── collect_data.py                # MJLab 仿真数据采集
│   ├── convert_to_lerobot.py          # npz+mp4 → LeRobot v2 格式转换
│   ├── export_int4.py                 # INT4 量化导出（BitsAndBytes）
│   ├── export_int4_offline.py         # INT4 离线量化（无网络）
│   ├── quantize_safetensors.py        # NF4 查找表量化核心
│   ├── infer.py                       # 本地推理包装器
│   ├── configs/
│   │   ├── g1_config.py               # G1 人形机器人配置（29 关节）
│   │   ├── go2_config.py              # Go2 四足机器人配置（12 关节）
│   │   ├── g1_modality_config.py      # G1 ModalityConfig（训练用）
│   │   ├── go2_modality_config.py     # Go2 ModalityConfig（训练用）
│   │   ├── h1_modality_config.py      # H1 ModalityConfig（训练用）
│   │   ├── h1_with_hand_modality_config.py  # H1+Hand ModalityConfig
│   │   ├── h1_2_modality_config.py    # H1.2 ModalityConfig（训练用）
│   │   └── h2_modality_config.py      # H2 ModalityConfig（训练用）
│   └── viz/
│       ├── __init__.py
│       ├── viser_viewer.py            # Viser 浏览器 3D 可视化
│       ├── mujoco_viewer.py           # MuJoCo 桌面窗口可视化
│       ├── viser_infer.py             # Viser + Policy Server 推理可视化
│       └── mujoco_infer.py            # MuJoCo + Policy Server 推理可视化
├── tests/                             # 单元测试（116 个测试）
│   ├── conftest.py
│   ├── test_configs.py
│   ├── test_quantize_safetensors.py
│   ├── test_infer.py
│   ├── test_collect_data.py
│   ├── test_convert_to_lerobot.py
│   ├── test_export_int4.py
│   ├── test_observation_builder.py
│   └── test_policy_client.py
├── output/                            # 推理输出（gitignore）
└── plan.md                            # 方案设计文档
```

## 本地依赖

### 纯推理（无需 GPU）

```bash
pip install pyzmq msgpack msgpack-numpy numpy opencv-python matplotlib pandas
```

### 数据采集 + 可视化 + 量化

```bash
# MuJoCo 仿真（数据采集）
pip install mujoco glfw

# Viser 浏览器可视化
pip install viser

# INT4 量化（CPU 可运行，无需 GPU）
pip install safetensors

# BitsAndBytes 量化（需要 GPU + transformers）
pip install bitsandbytes transformers torch

# 测试
pip install pytest
```

## 云端依赖（AutoDL）

云端运行 Isaac-GR00T，依赖其 `pyproject.toml`：

```bash
# 在 Isaac-GR00T 目录下
uv sync --python 3.10
```

## 四种可视化方案

| 方案 | 技术 | 输出 | 适用场景 |
|------|------|------|----------|
| A: Demo 静态图 | matplotlib | JPEG + MSE/MAE | 快速验证、写报告 |
| B: Viser 推理可视化 | viser + Policy Server | 浏览器 GUI (:20006) | 远程查看推理动作 |
| C: MuJoCo 推理可视化 | mujoco.viewer + Policy Server | 桌面窗口 | 本地调试推理动作 |
| D: Viser 纯3D查看 | viser | 浏览器 GUI (:20006) | 仅查看模型（无推理） |

## 支持的机器人

| 机器人 | 关节数 | 状态维度 | 动作维度 | 相机 |
|--------|--------|----------|----------|------|
| G1 人形 | 29 | 71 | 29 | front, wrist |
| H1 人形 | 20 | 53 | 20 | front, wrist |
| H1+Hand 人形 | 46 | 99 | 46 | front, wrist |
| H1.2 人形 | 52 | 105 | 52 | front, wrist |
| H2 人形 | 32 | 65 | 32 | front, wrist |
| Go2 四足 | 12 | 37 | 12 | front, back |

## 测试

```bash
# 运行全部 116 个单元测试
pytest tests/ -v

# 运行指定模块测试
pytest tests/test_quantize_safetensors.py -v
pytest tests/test_collect_data.py -v
```

覆盖：配置、NF4 量化、推理缓冲区、数据采集、格式转换、INT4 导出、观测构建、ZMQ 客户端。

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `Connection refused` | 确认云端 Server 已启动，SSH 隧道已建立 |
| `ZMQError: Address already in use` | 更换端口 `--port 5556` |
| 模型下载慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| SSH 断连 | 使用 tmux 保持隧道，或加 `ServerAliveInterval` |
| `ModuleNotFoundError: gr00t` | 确认 Isaac-GR00T 在同级目录，且已 `uv sync` |
| 量化报错 OOM | 使用 `export_int4_offline.py`（CPU 友好） |
| MuJoCo 窗口打不开 | 需要桌面环境（X11/Wayland），远程用 Viser |
| safetensors 版本报错 | `pip install safetensors>=0.8` |

## 技术文档

| 文档 | 说明 |
|------|------|
| `plan.md` | 方案设计文档 |
| `plan-phase2-4.md` | 详细设计文档（580 行） |

## 当前状态

- ✅ Demo 推理（ZMQ 客户端 + 云端 Server + SSH 隧道）
- ✅ 数据采集 + 格式转换 + 上传脚本
- ✅ INT4 量化（离线/在线） + 本地推理包装器
- ✅ Viser / MuJoCo 可视化
- ✅ 116 个单元测试全部通过
- 📦 源码 3955 行 + 测试 1176 行

---

> 📝 文档版本：v2.1 | 2026-06-28
