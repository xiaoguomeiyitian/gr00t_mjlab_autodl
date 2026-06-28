# GR00T MJLab AutoDL

NVIDIA Isaac-GR00T 的云端推理编排兄弟项目。

## 定位

本项目**不修改** Isaac-GR00T 代码，仅通过 ZMQ 接口远程调用云端推理服务。

```
本地（无 GPU）──SSH 隧道──► AutoDL 云端（GPU）
                                    │
                         Isaac-GR00T Policy Server
```

## 快速上手

### 1. 云端（AutoDL）

```bash
# 克隆本项目（与 Isaac-GR00T 同级目录）
git clone <repo-url> gr00t_mjlab_autodl
cd gr00t_mjlab_autodl

# 一次性环境初始化
bash scripts/00_autodl_init.sh

# 启动推理服务
bash scripts/01_start_server.sh nvidia/GR00T-N1.7-3B OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

### 2. 本地

```bash
# 配置 SSH 连接信息
vim config/ssh_config.sh  # 填写 AutoDL 信息

# 建立 SSH 隧道（保持运行）
bash scripts/02_local_tunnel.sh

# 运行 Demo 推理（另开终端）
bash scripts/03_local_demo_eval.sh
```

## 目录结构

```
gr00t_mjlab_autodl/
├── README.md                          # 本文件
├── .gitignore
├── config/
│   └── ssh_config.sh                  # SSH 配置（用户填写）
├── scripts/
│   ├── 00_autodl_init.sh              # [云端] 环境初始化
│   ├── 01_start_server.sh             # [云端] 启动 Policy Server
│   ├── 02_local_tunnel.sh             # [本地] 建立 SSH 隧道
│   └── 03_local_demo_eval.sh          # [本地] Demo 推理
└── src/
    ├── __init__.py
    ├── policy_client.py               # ZMQ 通信封装
    ├── observation_builder.py         # 观测格式构建
    ├── demo_plotter.py                # 静态图渲染
    └── demo_eval.py                   # 推理主逻辑
```

## 前置条件

- AutoDL 实例：GPU 16GB+（RTX 4090 / L40 / A100）
- 本地：Ubuntu 22.04，Python 3.8+，无需 GPU
- 网络：可访问 GitHub / HuggingFace（或配置镜像）

## 配置说明

### SSH 配置（config/ssh_config.sh）

```bash
SSH_HOST="region-xx.autodl.pro"   # AutoDL 服务器地址
SSH_PORT=xxxxx                     # SSH 端口
SSH_USER="root"                    # 用户名
SERVER_PORT=5555                   # 云端端口
LOCAL_PORT=5555                    # 本地端口
```

### 推理参数（scripts/03_local_demo_eval.sh）

```bash
DATASET_PATH="Isaac-GR00T/demo_data/droid_sample"   # 数据集路径
EMBODIMENT_TAG="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"  # 具身标签
HOST="127.0.0.1"                                   # 服务器地址（隧道后）
PORT=5555                                           # 端口
OUTPUT_DIR="./output"                               # 输出目录
TRAJ_IDS="1 2"                                      # 轨迹 IDs
ACTION_HORIZON=8                                    # 动作步数
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `Connection refused` | 确认云端 Server 已启动，SSH 隧道已建立 |
| `ZMQError: Address already in use` | 更换端口 `--port 5556` |
| 模型下载慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| SSH 断连 | 使用 tmux 保持隧道，或加 `ServerAliveInterval` |
| `ModuleNotFoundError: gr00t` | 确认 Isaac-GR00T 在同级目录，且已 `uv sync` |

## 后续阶段

| 阶段 | 内容 |
|------|------|
| Phase 2 | 本地 mjlab 数据采集 → 上传 AutoDL → 微调训练 |
| Phase 3 | 模型下载 → 本地 INT4 量化 → 本地推理验证 |
| Phase 4 | Viser 浏览器可视化 / MuJoCo 原生可视化 |

---

> 📝 文档版本：v1.0 | 2026-06-28
