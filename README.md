# GR00T × unitree_rl_mjlab × AutoDL

> **目标**: 本地用 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (mjlab / MuJoCo) 收集 G1/Go2 演示数据 → 上传到 AutoDL 云端 fine-tune GR00T → 下载模型回本地推理验证。
>
> **依赖**: 基于官方 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目，不依赖 unitree_tasks。

---

## 📐 架构总览

```
本地 (Linux + GPU)                                  AutoDL 云端 (RTX 5090 / A100)
┌─────────────────────────────────┐                 ┌──────────────────────────────┐
│ ① unitree_rl_mjlab 仿真         │                 │                              │
│   mjlab (MuJoCo) 物理引擎      │  SCP 上传 [2]   │  Isaac-GR00T 仓库            │
│   G1 (29 joints) / Go2 (12)    │ ──────────────▶│  - GR00T-N1.7-3B 模型       │
│   Task: Unitree-{G1,Go2}-Flat   │                 │  - LeRobot v2 数据           │
│   脚本化步态 (G1/Go2)           │  SCP 下载 [4]   │  - Fine-tune (LoRA)         │
│                                 │ ◀──────────────│  - Phase 3: FP16 全量 (~7GB) │
│ ③ 本地 INT4 量化 (可选) [5]     │                 │  - Phase 3.5: INT4 (~1.5GB) │
│   FP16 → INT4 PTQ (8GB 友好)   │                 │  - Phase 4: 双模型打包      │
│ ② 转换 LeRobot v2               │                 │                              │
│   parquet + modality.json       │                 │                              │
│ ④ 本地推理验证 [6]              │                 │                              │
│   ★ 高显存 (24GB+) → FP16       │                 │                              │
│   ★ 低显存 (8GB)  → INT4        │                 │                              │
│   mjlab Viser 回放              │                 │                              │
└─────────────────────────────────┘                 └──────────────────────────────┘
```

---

## 🗺️ 6 步流程图

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │ [1] 01_local_collect.sh      本地 mjlab 收集 episodes + 打包         │
 │       ↓                                                              │
 │ [2] 02_upload_to_autodl.sh   SCP 上传训练包 + 训练脚本到 AutoDL       │
 │       ↓                                                              │
 │ [3] 03_autodl_train.sh       云端解压 + Fine-tune + 打包 (FP16/INT4)  │
 │       ↓                                                              │
 │ [4] 04_download_model.sh     SCP 下载训练好的模型包到本地             │
 │       ↓                                                              │
 │ [5] 05_local_quantize.sh     本地 FP16 → INT4 (可选, 8GB 显存友好)    │
 │       ↓                                                              │
 │ [6] 06_local_verify.sh       加载模型 + 在 mjlab 中推理验证           │
 └─────────────────────────────────────────────────────────────────────┘
```

> **SSH 配置只需填一次**：编辑 [scripts/_ssh_config.sh](scripts/_ssh_config.sh) 顶部 `SSH_HOST` / `SSH_PORT`，
> 上传 ([2]) 和下载 ([4]) 脚本会自动读取。也可命令行 `--ssh-host` 临时覆盖。

---

## 📁 项目结构

```
gr00t_mjlab_autodl/                # 本项目 (与 unitree_rl_mjlab/ 同级)
├── README.md                       # 本文件 (方案说明)
├── requirements.txt                # GR00T 云端训练依赖
├── .gitignore
│
├── start.sh                        # 🎯 统一入口 (交互式 / 选择步骤 1-6)
├── build.sh                        # 🐳 Docker 镜像构建 (交互式 / 命令行)
│
├── scripts/
│   ├── _ssh_config.sh             # ⚙️ SSH 配置 (host/port/key) — 上传/下载共用
│   ├── 00_autodl_init.sh          # [0] 云端环境初始化 (一次性, 只需首次运行)
│   ├── 01_local_collect.sh         # [1] 数据采集: 本地 mjlab 收集 + 打包
│   ├── 02_upload_to_autodl.sh      # [2] 上传云端: SCP 训练包 + 训练脚本
│   ├── 03_autodl_train.sh          # [3] 云端训练: 解压 + LoRA + FP16 + INT4 打包
│   ├── 04_download_model.sh        # [4] 模型下载: SCP 下载 FP16 (默认, 可选 INT4)
│   ├── 05_local_quantize.sh        # [5] 本地量化 (可选): FP16 → INT4 (PTQ, 8GB 友好)
│   └── 06_local_verify.sh          # [6] 本地推理: 验证推理输出 (支持 --auto-quantize)
│
├── src/                            # 核心 Python 代码
│   ├── __init__.py
│   ├── collect_data.py             # 数据收集器 (基于 unitree_rl_mjlab)
│   ├── convert_to_lerobot.py       # npz → LeRobot v2 格式
│   ├── merge_lora.py               # LoRA 合并 → 完整 FP16 模型 (~7GB)
│   ├── export_int4.py              # INT4 量化导出 (~1.5GB)
│   ├── infer.py                    # 本地推理包装器 (mjlab + GR00T)
│   └── configs/
│       ├── __init__.py
│       ├── g1_config.py            # G1 配置 (29 joints, HOME_KEYFRAME)
│       └── go2_config.py           # Go2 配置 (12 joints, INIT_STATE)
│
└── docker/                         # 🐳 本地双镜像 Docker 方案 (训练在云端)
    ├── README.md                   # 镜像方案详细说明
    ├── Dockerfile.base             # 共享基础: CUDA + Python 3.12 + mjlab
    ├── Dockerfile.collect          # [1] 数据采集镜像
    ├── Dockerfile.infer            # [6] 推理验证镜像
    └── scripts/
        ├── entrypoint.sh           # 统一入口 (mode → 调度)
        ├── collect.sh              # [1] 调度脚本
        └── infer.sh                # [6] 调度脚本
    # 注: build.sh + start.sh 已上移到项目根, 方便日常调用
    #     训练镜像已移除: 训练在 AutoDL 云端, 用 ./scripts/02_upload_to_autodl.sh

# 运行时生成 (gitignored):
data/                              # 数据输出
├── g1_raw/                         # mjlab 收集的 npz
├── g1_lerobot/                     # LeRobot v2 格式
├── go2_raw/
└── go2_lerobot/
models/                            # 下载的模型
├── g1_gr00t_full_fp16/             # FP16 全量 (~7GB, 高显存 GPU)
└── g1_gr00t_int4/                  # INT4 量化 (~1.5GB, 低显存 GPU)
```

---

## 📁 项目布局要求

`gr00t_mjlab_autodl/` 必须与 `unitree_rl_mjlab/` 在同一父目录下:

```
unitree/                            # 任意路径, 仅作示例
├── gr00t_mjlab_autodl/             # 本项目 ← 脚本自动检测这个布局
└── unitree_rl_mjlab/                # 官方 mjlab 项目 (必须存在)
```

> `06_local_verify.sh` 会自动查找上级的 `../unitree_rl_mjlab/`。

---

## 🔧 前置依赖

### 1. 安装 unitree_rl_mjlab (官方)

```bash
# 进入 unitree_rl_mjlab 目录
cd /home/kxy/work/unitree/unitree_rl_mjlab

# 系统依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev \
    libeigen3-dev libspdlog-dev libfmt-dev

# Python 依赖 (editable 安装)
pip install -e .
```

### 2. 验证 mjlab 安装

```bash
python3 -c "import mjlab; print('mjlab OK')"

# 列出可用任务
cd /home/kxy/work/unitree/unitree_rl_mjlab
python3 scripts/list_envs.py
```

预期输出（节选）:
```
Available tasks:
  Unitree-A2-Flat
  Unitree-G1-23Dof-Flat
  Unitree-G1-Flat
  Unitree-G1-Rough
  Unitree-Go2-Flat
  Unitree-Go2-Rough
  Unitree-H1_2-Flat
  Unitree-R1-Flat
```

### 3. 安装本项目依赖

```bash
cd /home/kxy/work/unitree/gr00t_mjlab_autodl
pip install -r requirements.txt
```

### 4. (云端训练) 安装 Isaac-GR00T

在 AutoDL 实例上:
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
uv sync --python 3.10
source .venv/bin/activate
```

---

## 🚀 快速开始

### ⚙️ 一次性配置 SSH (推荐)

编辑 [scripts/_ssh_config.sh](scripts/_ssh_config.sh) 顶部填写 AutoDL 连接信息:

```bash
# filepath: scripts/_ssh_config.sh
SSH_HOST="root@region-9.autodl.com"    # ← 改成你的 AutoDL 主机
SSH_PORT="32451"                        # ← 改成你的 AutoDL 端口 (控制台查看)
SSH_KEY="$HOME/.ssh/id_rsa_autodl"      # ← 推荐用密钥 (留空用密码)
SSH_PASS=""                             # ← 密码 (推荐用密钥, 留空)
REMOTE_DIR="/root/workspace"            # 云端工作目录 (一般不用改)
```

> **之后 [2] 上传 和 [4] 下载 都会自动读取**, 无需每次命令行传参。

### 方式一: 通过 start.sh 统一入口 (推荐)

`start.sh` 是**单一入口**, 接收步骤号 1-6 或旧别名 (collect/infer/shell)。每个步骤按其属性自动选择运行环境:

| 步骤 | 名称 | 运行环境 | 调用的脚本 |
|------|------|---------|-----------|
| [1] | 数据采集 | 🐳 Docker (collect 镜像) | `01_local_collect.sh` |
| [2] | 上传云端 | 💻 主机 shell | `02_upload_to_autodl.sh` |
| [3] | 云端训练 | ☁️  AutoDL (本机提示 SSH) | `03_autodl_train.sh` (云端) |
| [4] | 下载模型 | 💻 主机 shell | `04_download_model.sh` |
| [5] | 本地量化 | 💻 主机 shell | `05_local_quantize.sh` |
| [6] | 推理验证 | 🐳 Docker (infer 镜像) | `06_local_verify.sh` |

```bash
cd gr00t_mjlab_autodl/

# 交互式菜单 (列出所有步骤 + 环境说明)
./start.sh

# 直接执行某个步骤
./start.sh 1                     # [1] 数据采集 (启动 collect 镜像)
./start.sh 2                     # [2] 上传云端 (主机 shell, 读 _ssh_config.sh)
./start.sh 3                     # [3] 云端训练 (提示 SSH 到 AutoDL)
./start.sh 4                     # [4] 下载模型 (主机 shell)
./start.sh 5                     # [5] 本地量化 (主机 shell)
./start.sh 6                     # [6] 推理验证 (启动 infer 镜像)

# 透传额外参数给底层脚本
./start.sh 1 --robot g1 --episodes 200
./start.sh 4 --with-int4
./start.sh 5 --robot g1 --offline
./start.sh 6 --auto-quantize --show-viewer

# 旧别名仍可用 (向后兼容)
./start.sh collect               # 等价于 ./start.sh 1
./start.sh infer                 # 等价于 ./start.sh 6
./start.sh shell                 # 进入 collect 镜像的 shell (调试用)

# 查看帮助
./start.sh --help
```

### 方式二: 手动分步执行 (精细控制 / 调试用)

如果 `start.sh` 调度不够灵活, 可以直接调用 `scripts/` 下的脚本:

```bash
cd gr00t_mjlab_autodl/

# [1] 数据采集: 本地 mjlab 收集 + 打包
./scripts/01_local_collect.sh --robot g1 --episodes 100 --instruction "walk forward"
#   产物: g1_gr00t_training.tar.gz

# [2] 上传到云端 (读 _ssh_config.sh 的 SSH 信息)
./scripts/02_upload_to_autodl.sh --robot g1
#   产物: 远端 /root/workspace/g1_gr00t_training.tar.gz + 03_autodl_train.sh

# [3] 云端训练: 远程触发 (在云端解压 + 训练 + 打包)
./scripts/03_autodl_train.sh --robot g1 --epochs 20    # 在云端执行
#   产物: 远端 /root/workspace/g1_gr00t_full_fp16.tar.gz (含 INT4)

# [4] 下载模型到本地
./scripts/04_download_model.sh --robot g1
#   产物: 本地 models/g1_gr00t_full_fp16/

# [5] 本地量化 (可选, 8GB 显存友好)
./scripts/05_local_quantize.sh --robot g1
#   产物: 本地 models/g1_gr00t_int4/

# [6] 本地推理验证
./scripts/06_local_verify.sh --robot g1
```

### 方式三: 云端手动分步 (推荐首次使用, 灵活排错)

云端训练 **不依赖 unitree_rl_mjlab** (仿真框架仅本地采集数据用), 云端只需 Isaac-GR00T + LeRobot v2 数据集 + LoRA。

```bash
# ── A. AutoDL 控制台: 开实例 → 选 PyTorch 2.x + CUDA 12.x 镜像 → 开机 ──

# ── B. ssh 登录 AutoDL, 一次性初始化环境 ──
ssh root@xxx.autodl.com -p 12345
bash 00_autodl_init.sh                   # 一键: 系统依赖 + uv + Isaac-GR00T + Python 3.10 venv + 训练栈 + 基础模型

# ── C. 上传本地训练数据 (在本地执行) ──
#   方式 C1: 用 [2] 脚本
./scripts/02_upload_to_autodl.sh --robot g1
#   方式 C2: 手动 scp
scp g1_gr00t_training.tar.gz root@xxx.autodl.com:/root/workspace/

# ── D. 回到云端, 启动训练 ──
#   方式 D1: 用 [3] 脚本 (推荐)
cd /root/workspace && bash 03_autodl_train.sh --robot g1 --epochs 20
#   方式 D2: 手动 (深入排查)
cd /root/workspace
tar -xzf g1_gr00t_training.tar.gz
source /root/Isaac-GR00T/.venv/bin/activate
cd /root/Isaac-GR00T
python3 gr00t/experiment/launch_finetune.py \
    --base-model-path /root/models/GR00T-N1-1.7-3B \
    --dataset-path /root/data/g1_lerobot \
    --output-dir /root/models/g1_gr00t \
    --num-epochs 10 --batch-size 2 --grad-accum 2 --learning-rate 1e-4 \
    --embodiment-tag NEW_EMBODIMENT --use-lora
```

> **回本地下载模型**: 用 [4] 脚本 `./scripts/04_download_model.sh --robot g1`

### 方式三: 自建 Docker 镜像 (本地双镜像: collect + infer)

> **训练在 AutoDL 云端**, 不构建本地训练镜像。本地 Docker 只承担 [1] 数据采集 和 [6] 推理验证。

| 镜像 | 大小 | 用途 |
|------|------|------|
| `gr00t-mjlab-collect:latest` | ~5GB | [1] 数据采集 (mjlab + collect 脚本) |
| `gr00t-mjlab-infer:latest`   | ~7GB | [6] 推理验证 (transformers + peft + bnb) |
| `gr00t-mjlab-base:latest`    | ~5GB | 共享基础 (CUDA + mjlab) |

[2] 上传 / [4] 下载是纯 SSH/SCP, **不依赖 Docker**;
[5] 本地量化是 Python 推理, **可以用主机或 infer 容器内的 Python**。

```bash
# 首次构建 (国内镜像源, ~10-20 分钟)
./build.sh --mirror cn all

# 通过统一入口执行步骤
./start.sh 1                       # [1] 数据采集 (启动 collect 镜像)
./start.sh 4                       # [4] 下载模型 (主机 shell, 读 _ssh_config.sh)
./start.sh 5                       # [5] 本地量化 (主机 shell)
./start.sh 6                       # [6] 推理验证 (启动 infer 镜像)

# 旧名兼容 (直接走对应镜像)
./start.sh collect                 # 等价于 ./start.sh 1
./start.sh infer                   # 等价于 ./start.sh 6
./start.sh shell                   # 进入 collect 镜像的 shell

# 训练不在本地 Docker, 用 shell 脚本 + AutoDL
./start.sh 2                       # [2] 上传 (主机 shell)
# (云端) ./start.sh 3              # [3] 训练 (提示 SSH 到 AutoDL)

# 环境变量透传 (collect/infer 容器)
ROBOT=g1 NUM_EPISODES=200 ./start.sh 1
INSTRUCTION="walk forward" MODEL_PATH=/root/models/g1_gr00t_int4 ./start.sh 6
```

> `./build.sh` + `./start.sh` 在项目根目录直接调用, Dockerfile 与调度脚本保留在 `docker/` 子目录便于维护。

详见 [`docker/README.md`](docker/README.md)。

---

## 🎯 模型推理: FP16 全量 vs INT4 量化

云端训练完成后, 默认会**生成 FP16 全量模型包** (~7GB), INT4 量化在**本地完成**:

| 模型包 | 大小 | 适用场景 | 推理质量 |
|--------|------|----------|----------|
| `{robot}_gr00t_full_fp16.tar.gz` | ~7GB | **高显存 GPU** (RTX 4090 24GB+, A100 40GB+) | ⭐⭐⭐⭐⭐ 最佳 |
| `{robot}_gr00t_int4_model.tar.gz` | ~1.5GB | **低显存 GPU** (RTX 2080 8GB, RTX 3060 12GB) | ⭐⭐⭐⭐ 良好 |

**Phase 2.5** (`merge_lora.py`): 将训练后的 LoRA adapter 合并到 GR00T 基础模型, 保存完整 FP16 模型。
**Phase 3** (`export_int4.py`): 在合并的 FP16 模型基础上做 INT4 量化 (NF4 + double-quant), **纯后训练量化 (PTQ), 无需重新训练**。

### 💡 v2 优化版: 仅下载 FP16, 本地量化 INT4

为了减少云服务器使用时间和下载量, **默认只下载 FP16 全量模型**, 然后在本地 (8GB+ 显存即可) 量化生成 INT4:

| 对比项 | 旧版（云端量化） | v2 优化版（本地量化） |
|--------|----------------|----------------------|
| 云端 GPU 时间 | 训练 + 合并 + INT4 量化 + 打包 | 训练 + 合并 + 打包 ⬇️ |
| 下载量 | 7GB + 1.5GB = 8.5GB | **仅 7GB** ⬇️ |
| 本地显存要求 | 任意（仅推理） | 8GB+ (量化 + 推理) ⬇️ |
| 是否需要重新训练 | — | **不需要**（PTQ 纯转换） |

**下载脚本** (默认仅下 FP16, SSH 信息从 `_ssh_config.sh` 自动读):
```bash
# ⭐ 推荐: 只下载 FP16 (7GB), 本地再量化
./scripts/04_download_model.sh --robot g1

# 旧行为: 同时下载 FP16 + INT4 (8.5GB)
./scripts/04_download_model.sh --robot g1 --with-int4

# 只下载 INT4 (假定本地已有 FP16)
./scripts/04_download_model.sh --robot g1 --no-fp16 --with-int4
```

**本地量化** (PTQ, 8GB+ 显存, 5-15 分钟):
```bash
# 默认: 从 models/g1_gr00t_full_fp16/ → models/g1_gr00t_int4/
./scripts/05_local_quantize.sh --robot g1

# 离线模式 (避免 HF 在线下载触发限流)
./scripts/05_local_quantize.sh --robot g1 --offline

# 显式指定路径
./scripts/05_local_quantize.sh \
    --input models/g1_gr00t_full_fp16 \
    --output models/g1_gr00t_int4
```

**一键全流程优化** (推荐 8GB 显存用户):
```bash
# [3] 云端训练: 跳过 INT4 量化 (省 GPU 时间), 后续 ./start.sh 5 在本地生成
./start.sh 3 --skip-int4-remote

# [4] 下载: 同时下 FP16 + INT4 (云端生成了)
./start.sh 4 --with-int4

# [5] 跳过本地量化 (用云端 INT4)
./start.sh 5 --skip  # (如需支持, 可加 --skip)

# [6] 本地推理: 自动从 FP16 量化 (若只有 FP16)
./start.sh 6 --auto-quantize
```

本地推理时, 验证脚本会自动选择:
```bash
# 默认: 优先查找 INT4 → FP16 → LoRA (按优先级)
./scripts/06_local_verify.sh --robot g1

# ⭐ 8GB 显卡: 自动从 FP16 量化后推理 (无需先手动量化)
./scripts/04_local_verify.sh --robot g1 --auto-quantize

# 显式指定使用 FP16 全量 (高显存 GPU)
./scripts/04_local_verify.sh --robot g1 \
    --model-path models/g1_gr00t_full_fp16 --quantize none

# 显式指定使用 INT4 (低显存 GPU)
./scripts/04_local_verify.sh --robot g1 \
    --model-path models/g1_gr00t_int4 --quantize 4bit
    --model-path models/g1_gr00t_int4 --quantize 4bit
```

---

## 🎯 支持的任务 (unitree_rl_mjlab 官方)

| Task ID | 机器人 | 关节数 | 地形 |
|---------|--------|--------|------|
| `Unitree-G1-Flat` | G1 (29 DOF) | 29 | 平坦 |
| `Unitree-G1-Rough` | G1 (29 DOF) | 29 | 粗糙 |
| `Unitree-G1-23Dof-Flat` | G1 (23 DOF) | 23 | 平坦 |
| `Unitree-Go2-Flat` | Go2 | 12 | 平坦 |
| `Unitree-Go2-Rough` | Go2 | 12 | 粗糙 |
| `Unitree-A2-Flat` | A2 | 12 | 平坦 |
| `Unitree-H1_2-Flat` | H1_2 | — | 平坦 |
| `Unitree-R1-Flat` | R1 | — | 平坦 |

使用示例:
```bash
python3 collect_data.py --task Unitree-Go2-Rough \
    --num-episodes 100 --instruction "climb stairs"
```

---

## 📊 数据格式 (LeRobot v2)

转换后的数据目录结构:

```
data/g1_lerobot/
├── meta/
│   ├── modality.json       # 模态定义 (G1/Go2 配置)
│   ├── episodes.jsonl      # episode 元数据
│   └── tasks.jsonl         # 任务描述
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # 所有 episode 数据
└── videos/                 # (可选) RGB 视频
```

`modality.json` (G1 示例):
```json
{
  "state": {
    "state.joint_pos": {"dtype": "float32", "shape": [29], "description": "G1 29 关节位置 (rad)"},
    "state.joint_vel": {"dtype": "float32", "shape": [29], "description": "G1 29 关节速度 (rad/s)"},
    "state.base_pos":  {"dtype": "float32", "shape": [3]},
    "state.base_quat": {"dtype": "float32", "shape": [4]},
    "state.base_lin_vel": {"dtype": "float32", "shape": [3]},
    "state.base_ang_vel": {"dtype": "float32", "shape": [3]}
  },
  "action": {
    "action.joint_position_target": {"dtype": "float32", "shape": [29]}
  },
  "video": {
    "video.front_view": {"dtype": "video", "shape": [224, 224, 3], "fps": 50}
  }
}
```

---

## 🖥️ AutoDL 实例配置

| 配置项 | 推荐值 |
|--------|--------|
| GPU | RTX 5090 32GB (性价比最高) |
| 镜像 | PyTorch 2.11 + CUDA 12.4 |
| 磁盘 | ≥100GB |
| 内存 | ≥32GB |

---

## 🔬 与 unitree_rl_mjlab 官方的对应关系

| 本项目 | unitree_rl_mjlab 官方 |
|--------|----------------------|
| `collect_data.py` | `scripts/play.py` (使用 `ManagerBasedRlEnv`, 我们只采集不训练) |
| `configs/g1_config.py` | `src/assets/robots/unitree_g1/g1_constants.py` (HOME_KEYFRAME) |
| `configs/go2_config.py` | `src/assets/robots/unitree_go2/go2_constants.py` (INIT_STATE) |
| 关节名称 | `src/assets/robots/{robot}/xmls/{robot}.xml` (MJCF 定义) |
| Task ID | `src/tasks/velocity/config/{robot}/__init__.py` |
| 环境创建 | `mjlab.tasks.registry.load_env_cfg(task_id)` |

本项目完整复用了 `unitree_rl_mjlab` 官方的:
- ✅ 关节定义 (29 G1 / 12 Go2)
- ✅ 默认姿态 (HOME_KEYFRAME / INIT_STATE)
- ✅ 任务 ID (`Unitree-{G1,Go2}-{Flat,Rough}`)
- ✅ mjlab 仿真环境 (`ManagerBasedRlEnv`)
- ✅ Task 注册机制 (`mjlab.tasks.registry`)

---

## 🛠️ 常见问题

### Q1: 找不到 `unitree_rl_mjlab` 目录?

确保项目布局为兄弟关系:

```
unitree/                            # 任意父目录
├── gr00t_mjlab_autodl/             # 本项目 (当前)
└── unitree_rl_mjlab/                # 官方项目 (必须存在)
```

`06_local_verify.sh` 通过 `../unitree_rl_mjlab` 自动定位, 如布局不同可手动指定 `--rl-mjlab-root`。

### Q2: `mjlab` 导入失败？

```bash
cd /home/kxy/work/unitree/unitree_rl_mjlab
pip install -e .

# 验证
python3 -c "import mjlab, mujoco; print('OK', mjlab.__version__)"
```

### Q3: 收集数据时没有 GPU？

收集器会自动回退到**纯数据模式**（步态生成器 + 模拟物理），仍可生成完整数据用于 GR00T 训练。

### Q4: GR00T 模型加载失败？

确保已在 Python 路径中安装 `gr00t_integration`:
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
export PYTHONPATH=$PYTHONPATH:$PWD/Isaac-GR00T
```

### Q5: 镜像是否符合 AutoDL 发布要求?

**是**。`gr00t-mjlab-*` 本地双镜像均满足 AutoDL 镜像发布审核标准:
> 审核标准: 在 `/root` 目录下包含相应 Git 代码仓库, 且可以成功执行 Git 代码仓库中的简单代码

| 位置 | 内容 | Git 仓库 |
|------|------|---------|
| `/root/gr00t_mjlab_autodl/` | 本项目 (含 `src/`, `scripts/`, `docker/`, `README.md`) | ✅ `.git/` |
| `/root/unitree_rl_mjlab/` | 官方 mjlab (editable 安装) | ✅ `.git/` |

> 训练不在本地 Docker 中, 训练环境 (Isaac-GR00T) 在 AutoDL 云端按需构建, 详见 `scripts/00_autodl_init.sh`。

构建阶段自动执行 `git log` + Python 导入验证 (`AUTODL_BASE_OK`)。

任何镜像内可直接:
```bash
docker run --rm gr00t-mjlab-base:latest bash -c \
    "cd /root/gr00t_mjlab_autodl && git log --oneline -3 && \
     python3 -c 'import mjlab; print(mjlab.__version__)'"
```

构建时使用 `.dockerignore` (位于 `unitree/`) 排除 `data/`, `models/`, `__pycache__/`, `.venv/` 等,但保留 `.git/`。

### Q6: INT4 量化需要重新训练吗？

**不需要**。GR00T 项目用的是 **Post-Training Quantization (PTQ)**, 只是把 FP16 权重转换为 INT4 表示, 没有反向传播, **纯转换操作**:

```python
# 伪代码: 量化前后权重等价, 仅存储格式不同
fp16_weights = load_model()            # 加载训练好的 FP16 模型 (7GB)
int4_weights = quantize_to_int4(fp16_weights)  # 仅转换 (输出 1.5GB)
save(int4_weights)
```

**本地 8GB 显存可以完成 INT4 量化**:
- 量化峰值显存 ~2-3GB (GR00T-N1.7-3B)
- 耗时 5-15 分钟
- 量化后模型正好适合 8GB 显存推理

详见上文 [🎯 模型推理: FP16 全量 vs INT4 量化](#-模型推理-fp16-全量-vs-int4-量化) 的 v2 优化版说明。

---

## 📜 引用

- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) — 官方 mjlab 项目
- [mjlab](https://github.com/mujocolab/mjlab) — MuJoCo RL 框架
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA GR00T N1.7
- [LeRobot](https://github.com/huggingface/lerobot) — 数据格式标准
