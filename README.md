# GR00T × unitree_rl_mjlab × AutoDL

> **目标**: 本地用 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (mjlab / MuJoCo) 收集 G1/Go2 演示数据 → 上传到 AutoDL 云端 fine-tune GR00T → 下载模型回本地推理验证。
>
> **依赖**: 基于官方 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目，不依赖 unitree_tasks。

---

## 📐 架构总览

```
本地 (Linux + GPU)                                  AutoDL 云端 (A100 40GB / L40 48GB / H100)
┌─────────────────────────────────┐                 ┌──────────────────────────────────┐
│ ① unitree_rl_mjlab 仿真         │                 │                                  │
│   mjlab (MuJoCo) 物理引擎      │  SCP 上传 [2]   │  Isaac-GR00T 仓库                │
│   G1 (29 joints) / Go2 (12)    │ ──────────────▶│  - GR00T-N1.7-3B 模型 (7GB)     │
│   Task: Unitree-{G1,Go2}-Flat   │                 │  - LeRobot v2 数据               │
│                                 │ ◀──────────────│  - BF16 完整模型 (~7GB)    │
│ ③ 本地 INT4 量化 (可选) [5]     │                 │  - INT4 量化 (~1.5GB, 可选)          │
│ ② 转换 LeRobot v2               │                 │                                  │
│   parquet + modality.json       │                 │  ⚠️ 官方训练**不支持 LoRA**       │
│ ④ 本地推理验证 [6]              │                 │    (peft 装但未使用)             │
│   ★ 16GB+ VRAM → BF16          │                 │    训练输出本身就是 final model │
│   ★ 8GB  VRAM  → INT4          │                 │                                  │
│   mjlab Viser 回放              │                 │                                  │
└─────────────────────────────────┘                 └──────────────────────────────────┘
```

---

## 🗺️ 6 步流程图

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │ [1] 01_local_collect.sh      本地 mjlab 收集 episodes + 打包         │
 │       ↓                                                              │
 │ [2] 02_upload_to_autodl.sh   SCP 上传训练包 + 训练脚本到 AutoDL       │
 │       ↓                                                              │
 │ [3] 03_autodl_train.sh       云端解压 + Fine-tune + 打包 (BF16/INT4)    │
 │       ↓                                                              │
 │ [4] 04_download_model.sh     SCP 下载训练好的模型包到本地             │
 │       ↓                                                              │
 │ [5] 05_local_quantize.sh     本地 BF16 → INT4 (可选, 8GB 显存友好)    │
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
├── install_native.sh               # 🛠 本机环境一键安装 (Python + PyTorch + mjlab + GR00T)
│
├── scripts/
│   ├── _ssh_config.sh             # ⚙️ SSH 配置 (host/port/key) — 上传/下载共用
│   ├── 00_autodl_init.sh          # [0] 云端环境初始化 (一次性, 只需首次运行)
│   ├── 01_local_collect.sh         # [1] 数据采集: 本地 mjlab 收集 + 打包
│   ├── 02_upload_to_autodl.sh      # [2] 上传云端: SCP 训练包 + 训练脚本
│   ├── 03_autodl_train.sh          # [3] 云端训练: 解压 + Fine-tune + 打包 (BF16/INT4)
│   ├── 04_download_model.sh        # [4] 模型下载: SCP 下载 BF16 (默认, 可选 INT4)
│   ├── 05_local_quantize.sh        # [5] 本地量化 (可选): BF16 → INT4 (PTQ, 8GB 友好)
│   └── 06_local_verify.sh          # [6] 本地推理: 验证推理输出 (支持 --auto-quantize)
│
└── src/                            # 核心 Python 代码
    ├── __init__.py
    ├── collect_data.py             # 数据收集器 (基于 unitree_rl_mjlab)
    ├── convert_to_lerobot.py       # npz → LeRobot v2 格式
    ├── export_int4.py              # INT4 量化导出 (PTQ, 约 7GB → 1.5GB)
    ├── infer.py                    # 本地推理包装器 (mjlab + GR00T)
    └── configs/
        ├── __init__.py
        ├── g1_config.py            # G1 配置 (29 joints, HOME_KEYFRAME)
        └── go2_config.py           # Go2 配置 (12 joints, INIT_STATE)

# 运行时生成 (gitignored):
data/                              # 数据输出
├── g1_raw/                         # mjlab 收集的 npz
├── g1_lerobot/                     # LeRobot v2 格式
├── go2_raw/
└── go2_lerobot/
models/                            # 下载的模型
├── g1_gr00t/                       # BF16 完整模型 (~7GB, 高显存 GPU, 03 输出原生格式)
├── g1_gr00t_full_fp16/             # symlink 别名 (兼容旧脚本) / 旧下载目录
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
  Mjlab-Velocity-Flat-Unitree-G1
  Mjlab-Velocity-Flat-Unitree-Go2
  Mjlab-Velocity-Rough-Unitree-G1
  Mjlab-Velocity-Rough-Unitree-Go2
  ...
```

> **注意**: mjlab 1.2.0 的任务名称格式为 `Mjlab-Velocity-{Terrain}-Unitree-{Robot}`，
> 而非旧版的 `Unitree-{Robot}-{Terrain}`。脚本内部已做映射，用户无需手动转换。

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

## 🛠 本机一键安装 (Native 模式)

本项目默认就是**本机模式** — 所有步骤直接在本机 `.venv` / shell / SCP 中执行,
不需要容器。

### 快速安装

```bash
cd gr00t_mjlab_autodl/

# 1. 交互式安装
./install_native.sh

# 2. 或非交互式
./install_native.sh collect              # 只装采集 (约 6GB)
./install_native.sh infer                # 装采集 + 推理 (约 9GB)
./install_native.sh --mirror cn all      # 国内阿里云源
./install_native.sh --no-apt infer       # 跳过 apt, 假定依赖已装
./install_native.sh --recreate all       # 强制重建 .venv

# 3. 运行步骤
./start.sh                              # 交互式菜单
./start.sh 1                            # 采集
./start.sh 1 --robot g1                 # 采集 G1 数据
./start.sh 6 --auto-quantize            # 推理 (自动量化)

# 4. 进入 shell (激活 .venv, 可手动调 mjlab / Isaac-GR00T)
./start.sh shell
```

### 关键细节

- **Python 版本**: 推荐 Python 3.12，mjlab 1.2.0 要求 `>=3.10, <3.14`（3.14 不可用）
- **venv 位置**: `./.venv` (项目内, 已在 `.gitignore`)
- **不影响 shell rc**: 用 `source .venv/bin/activate` 手动激活, 或直接 `./start.sh ...`
- **PyTorch wheel**: `torch==2.11.0+cu128`
  - 支持 sm_70 (Volta) / sm_75 (Turing, RTX 2080) / sm_80+ (Ampere+) / sm_90 (Hopper) / sm_100/120 (Blackwell)
- **GPU 自动检测**: 自动检查 `nvidia-smi` 和 compute_cap
  - **有 GPU**: 使用 `MUJOCO_GL=egl` 后端（性能最佳）
  - **无 GPU / CPU 模式**: 使用 `MUJOCO_GL=osmesa` 后端（需要 `libosmesa6`）
  - 自动检测并设置环境变量，无需手动配置
- **数据/模型路径**: `data/` `models/` (项目根目录下)
- **NVIDIA 库路径**: 通过 `LD_LIBRARY_PATH` 动态注入 `.venv/lib/.../nvidia/*/lib`
- **Isaac-GR00T**: 如果 `../Isaac-GR00T/` 存在, 自动 `editable install` 并注入 `PYTHONPATH`
- **重装**: `./install_native.sh --recreate all` (删除现有 venv)
- **mediapy 兼容**: 自动修补 mediapy 1.2.6 的 numpy 2.x 兼容性问题（`_VideoArray` 类）
- **依赖版本**: mujoco 3.5.x + warp-lang 1.12.x + scipy 1.11+（mjlab 1.2.0 要求）

---


## 🚀 快速开始

### ⚙️ 一次性配置 SSH (推荐)

编辑 [scripts/_ssh_config.sh](scripts/_ssh_config.sh) 顶部填写 AutoDL 连接信息:

```bash
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
| [1] | 数据采集 | 💻 本机 .venv (含 mjlab) | `01_local_collect.sh` |
| [2] | 上传云端 | 💻 主机 shell | `02_upload_to_autodl.sh` |
| [3] | 云端训练 | ☁️  AutoDL (本机提示 SSH) | `03_autodl_train.sh` (云端) |
| [4] | 下载模型 | 💻 主机 shell | `04_download_model.sh` |
| [5] | 本地量化 | 💻 主机 shell | `05_local_quantize.sh` |
| [6] | 推理验证 | 💻 本机 .venv (含 mjlab + GR00T) | `06_local_verify.sh` |

```bash
cd gr00t_mjlab_autodl/

# 交互式菜单 (列出所有步骤 + 环境说明)
./start.sh

# 直接执行某个步骤
./start.sh 1                     # [1] 数据采集 (在本地 .venv 中执行)
./start.sh 2                     # [2] 上传云端 (主机 shell, 读 _ssh_config.sh)
./start.sh 3                     # [3] 云端训练 (提示 SSH 到 AutoDL)
./start.sh 4                     # [4] 下载模型 (主机 shell)
./start.sh 5                     # [5] 本地量化 (主机 shell)
./start.sh 6                     # [6] 推理验证 (在本地 .venv 中执行)

# 透传额外参数给底层脚本
./start.sh 1 --robot g1 --episodes 200
./start.sh 4 --with-int4
./start.sh 5 --robot g1 --offline
./start.sh 6 --auto-quantize --show-viewer

# 旧别名仍可用 (向后兼容)
./start.sh collect               # 等价于 ./start.sh 1
./start.sh infer                 # 等价于 ./start.sh 6
./start.sh shell                 # 进入本机 venv shell (调试用)

# 查看帮助
./start.sh --help
```

### 方式二: 手动分步执行

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
#   产物: 远端 /root/workspace/g1_gr00t_model.tar.gz (BF16 完整模型, INT4 可选)
#   (旧名 _full_fp16.tar.gz 已改为 symlink 兼容)

# [4] 下载模型到本地
./scripts/04_download_model.sh --robot g1
#   产物: 本地 models/g1_gr00t/  (训练输出, BF16 完整模型, ~7GB)

# [5] 本地量化 (可选, 8GB+ 显存友好)
./scripts/05_local_quantize.sh --robot g1
#   产物: 本地 models/g1_gr00t_int4/  (INT4 量化, ~1.5GB)

# [6] 本地推理验证
./scripts/06_local_verify.sh --robot g1
```

云端训练 **不依赖 unitree_rl_mjlab**（仿真框架仅本地采集数据用）, 云端只需 Isaac-GR00T + LeRobot v2 数据集。

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
# 注: 官方 FinetuneConfig 用 --max-steps, --global-batch-size, --gradient-accumulation-steps
# 不支持 --num-epochs（本项目 03_autodl_train.sh 已封装好）
python3 gr00t/experiment/launch_finetune.py \
    --base-model-path /root/models/GR00T-N1-1.7-3B \
    --dataset-path /root/data/g1_lerobot \
    --output-dir /root/models/g1_gr00t \
    --max-steps 5000 --global-batch-size 2 \
    --gradient-accumulation-steps 2 --learning-rate 1e-4 \
    --embodiment-tag NEW_EMBODIMENT --save-only-model
```

> **回本地下载模型**: 用 [4] 脚本 `./scripts/04_download_model.sh --robot g1`

### 快速参考

```bash
# 安装 + 采 + 推 (一条龙)
./install_native.sh infer                # 本机一键装环境
./start.sh 1 --robot g1                  # 采 G1 数据
./start.sh 1 --robot g1 --episodes 200   # 采 200 回合
./start.sh 6 --auto-quantize             # 推理 (自动量化)

# 采集 G1 + Go2 (多任务)
for r in g1 go2; do
    ./start.sh 1 --robot $r --episodes 200
done

# 环境变量透传 (给底层脚本)
ROBOT=g1 NUM_EPISODES=200 ./start.sh 1
INSTRUCTION="walk forward" MODEL_PATH=./models/g1_gr00t_int4 ./start.sh 6
```

---

## 🎯 模型推理: BF16 完整模型 vs INT4 量化

云端训练完成后, 默认会**生成 BF16 完整模型包** (~7GB), INT4 量化在**本地完成**:

> **为什么是 BF16 不是 FP16?** Isaac-GR00T 官方训练使用 `bf16=True` (PyTorch 训练配置),输出是原生 BF16 完整模型,无需再 merge LoRA 或转 FP16。BF16 与 FP16 在数值精度上几乎等价 (都是 16-bit 浮点, BF16 指数位更多利于训练稳定性, FP16 尾数位更多利于推理精度)。

| 模型包 | 大小 | 适用场景 | 推理质量 |
|--------|------|----------|----------|
| `{robot}_gr00t_model.tar.gz` | ~7GB | **高显存 GPU** (RTX 4090 24GB+, A100 40GB+) | ⭐⭐⭐⭐⭐ 最佳 |
| `{robot}_gr00t_full_fp16.tar.gz` | ~7GB | 同上, 兼容旧脚本 | ⭐⭐⭐⭐⭐ 最佳 |
| `{robot}_gr00t_int4_model.tar.gz` | ~1.5GB | **低显存 GPU** (RTX 2080 8GB, RTX 3060 12GB) | ⭐⭐⭐⭐ 良好 |

**`export_int4.py`**: 在 BF16 完整模型基础上做 INT4 量化 (NF4 + double-quant), **纯后训练量化 (PTQ), 无需重新训练**。
**旧名兼容**: `g1_gr00t_full_fp16.tar.gz` 是 03 脚本创建的 symlink, 指向主包, 旧脚本仍可读。

### 💡 v2 优化版: 仅下载 BF16, 本地量化 INT4

为了减少云服务器使用时间和下载量, **默认只下载 BF16 完整模型**, 然后在本地 (8GB+ 显存即可) 量化生成 INT4:

| 对比项 | 旧版（云端量化） | v2 优化版（本地量化） |
|--------|----------------|----------------------|
| 云端 GPU 时间 | 训练 + INT4 量化 + 打包 | 训练 + 打包 (INT4 可选) ⬇️ |
| 下载量 | 7GB + 1.5GB = 8.5GB | **仅 7GB** ⬇️ |
| 本地显存要求 | 任意（仅推理） | 8GB+ (量化 + 推理) ⬇️ |
| 是否需要重新训练 | — | **不需要**（PTQ 纯转换） |

**下载脚本** (默认仅下 BF16, SSH 信息从 `_ssh_config.sh` 自动读):
```bash
# 推荐: 只下载 BF16 完整模型 (7GB), 本地再量化
./scripts/04_download_model.sh --robot g1

# 旧行为: 同时下载 BF16 + INT4 (8.5GB)
./scripts/04_download_model.sh --robot g1 --with-int4

# 只下载 INT4 (假定本地已有 BF16)
./scripts/04_download_model.sh --robot g1 --no-fp16 --with-int4
```

**本地量化** (PTQ, 8GB+ 显存, 5-15 分钟):
```bash
# 默认: 从 models/g1_gr00t/ → models/g1_gr00t_int4/
./scripts/05_local_quantize.sh --robot g1

# 离线模式 (避免 HF 在线下载触发限流)
./scripts/05_local_quantize.sh --robot g1 --offline

# 显式指定路径
./scripts/05_local_quantize.sh \
    --input models/g1_gr00t \
    --output models/g1_gr00t_int4
```

**一键全流程优化** (推荐 8GB 显存用户):
```bash
# [3] 云端训练: 跳过 INT4 量化 (省 GPU 时间), 后续 ./start.sh 5 在本地生成
./start.sh 3 --no-export-int4

# [4] 下载: 同时下 BF16 + INT4 (云端生成了)
./start.sh 4 --with-int4

# [6] 本地推理: 自动从 BF16 量化 (若只有 BF16)
./start.sh 6 --auto-quantize
```

本地推理时, 验证脚本会自动选择:
```bash
# 默认: 优先查找 INT4 → BF16 (按优先级)
./scripts/06_local_verify.sh --robot g1

# 8GB 显卡: 自动从 BF16 量化后推理 (无需先手动量化)
./scripts/06_local_verify.sh --robot g1 --auto-quantize

# 显式指定使用 BF16 完整模型 (高显存 GPU)
./scripts/06_local_verify.sh --robot g1 \
    --model-path models/g1_gr00t --quantize none

# 显式指定使用 INT4 (低显存 GPU)
./scripts/06_local_verify.sh --robot g1 \
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

## 📹 数据采集选项 (collect_data.py)

`collect_data.py` 已支持 4 种**专家 Agent** 与 3 种**动作空间**,可任意组合用于收集 GR00T fine-tune 数据。

### 4 种 Agent

| Agent | 说明 | 用途 | 备注 |
|-------|------|------|------|
| `scripted` | 脚本化正弦步态 (默认) | CI / 流程测试 | 真实数据收集中**不推荐**(不是 PPO 专家) |
| `trained` | 加载训练好的 PPO checkpoint | **真实演示** | 需先 `cd ../unitree_rl_mjlab && python scripts/train.py --task Unitree-G1-Flat` 训练出 `model_*.pt` |
| `random` | 随机动作 | 探索 / baseline 对比 | — |
| `zero` | 零动作 | sanity check | — |

> ⚠️ 真实数据必须用 `--agent trained` + `--checkpoint model.pt`，否则 GR00T 学不到有用模式。

### 3 种动作空间 (`--action-mode`)

| 模式 | 适用机器人 | 数据字段 | 推荐场景 |
|------|------------|----------|----------|
| `delta` **默认** | G1 / Go2 | `action.joint_position_delta` + `action.joint_position_last` | **locomotion 推荐**(对累积漂移更鲁棒) |
| `absolute` | G1 / Go2 | `action.joint_position_target` | 关节目标位置绝对值 (兼容旧数据/简单回放) |
| `relative_eef` | G1 (有手) | `action.ee_pose_delta` | 末端执行器相对位姿 (manipulation) |

> 📌 默认 action-mode 为 `delta`。改用 `absolute` 时需显式传参，且训练 ModalityConfig 也需对应切换 (`*_absolute.py`)。

### 怎么选？决策表

| 数据 / 场景 | 推荐 mode | 原因 |
|------------|----------|------|
| 真实机器人 (joint_pos 有 estimator 噪声) | `delta` | 增量对 bias 鲁棒，仿真→真机迁移更稳 |
| 仿真 + 完美状态 (mjlab 50Hz) | `delta` 或 `absolute` 都可 | 数据干净时两者收敛接近 |
| Locomotion (G1/Go2 走路/跑步) | `delta` | 跨 episode 起点不同，增量分布更稳定 |
| Manipulation (末端抓取) | `relative_eef` | 抓取点是绝对的，不适合用 joint delta |
| 已有 `absolute` 历史数据想续训 | `absolute` | 重新采集成本高，兼容旧数据 |
| 做消融实验 (RELATIVE vs ABSOLUTE 对比) | 各跑一次 | GR00T 官方支持两种，需对比时分别采 |
| 不确定选哪个 | `delta` | README 主推 + GR00T 官方默认 |

### 视频采集 (`--video`)

GR00T N1.7 是 **VLA** (Vision-Language-Action),**必须包含 RGB 视频**作为视觉输入。本项目支持两种产出:

- **imageio mp4 输出** (优先): 224×224 RGB @ 30 fps,直接写入 `episode_NNNNNN.mp4`
- **frames npz fallback**: 当 imageio 不可用时,降级为 `episode_NNNNNN_frames.npz` 存储原始帧序列,转换阶段自动重新编码

### 使用示例

```bash
# 真实链路: PPO 专家 + delta 动作 + 视频 + 100 episodes
./scripts/01_local_collect.sh \
    --robot g1 \
    --task Unitree-G1-Flat \
    --agent trained \
    --checkpoint ../unitree_rl_mjlab/logs/rsl_rl/Unitree-G1-Flat/2025-XX-XX/model_3000.pt \
    --action-mode delta \
    --video \
    --episodes 200 \
    --instruction "walk forward at 0.5 m/s"

# CI smoke test: 脚本化 + 无视频 (仅验证流程)
./scripts/01_local_collect.sh \
    --robot g1 \
    --agent scripted \
    --no-video \
    --episodes 5

# 末端执行器相对位姿 (manipulation)
./scripts/01_local_collect.sh \
    --robot g1 \
    --agent trained \
    --checkpoint /path/to/ppo.pt \
    --action-mode relative_eef \
    --instruction "pick up the red block"
```

> ⚠️ **重要**: 训练时(`03_autodl_train.sh`)的 `action_mode` 必须与采集时一致。脚本会自动从 `meta/modality.json` 读取并在日志中显示,不匹配时会发出警告。

---

## 🌐 浏览器 3D 可视化 (Viser)

数据采集和本地推理都支持 **Viser** 浏览器 3D 可视化，实时查看机器人运动。

### 原理

```
mjlab 仿真 → qpos/qvel → AsyncViser3DViewer (后台线程) → 浏览器 http://localhost:8080
```

- `src/viser_3d_viewer.py` 封装 `AsyncViser3DViewer`，在后台线程运行 Viser 服务器
- 主线程每步调用 `update()` 同步最新关节位置/速度
- 30fps 渲染，非阻塞

### 数据采集时启用

```bash
# 采集数据 + 3D 可视化
./scripts/01_local_collect.sh --robot g1 --viser --viser-port 8080

# 或直接用 Python
python collect_data.py --agent scripted --viser --num-episodes 10
```

启动后输出类似:
```
🌐 Viser 3D viewer 已启动: http://localhost:8080 (浏览器打开此 URL)
```

### 本地推理时启用

```bash
python infer.py --model-path models/g1_gr00t_int4 --viser --viser-port 8080
```

### 依赖安装

```bash
pip install viser
```

> Viser 是可选依赖。未安装时 `--viser` 会被忽略，回退到文本进度面板。

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

> **配置要求基于 [Isaac-GR00T 官方 `getting_started/hardware_recommendation.md`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md) (v1.7)**
> 项目不内嵌任何未经官方验证的硬件推荐。

### GPU 推荐 (按官方背书排序)

| 等级 | GPU | VRAM | 训练支持 | 备注 |
|------|-----|------|---------|------|
| **官方推荐** | **A100** | 40-80 GB | 默认配置即可 | 性价比最均衡, AutoDL 平台常见 |
| **官方推荐** | **L40** | 48 GB | 默认配置即可 | 推理频率 26 Hz (TensorRT) |
| **官方推荐** | **H100** | 40-80 GB | 默认 + 多卡扩展 | 训练速度最快 (~4× A100) |
| 🟡 官方未测但理论可行 | RTX 5090 | 32 GB | ⚠️ 低于官方最低 40GB, 需 `--batch-size 1` + gradient checkpointing | 不在官方背书清单 |
| 🟡 官方未测但理论可行 | V100-32GB | 32 GB | ⚠️ 同上, 且 CUDA 13+ PyTorch wheel 不含 sm_70 二进制 | 不在官方背书清单 |
| 🟡 官方可接受 | A6000 | 48 GB | "works but may require longer training time" (官方原话) | |
| 🟢 边缘 (仅推理) | RTX 4090 | 24 GB | ❌ 训练 OOM; 推理 BF16 需谨慎 | 仅本地 [6] 推理用 |
| ❌ 不推荐 | < 24 GB | — | ❌ 训练 OOM; 推理仅 INT4 | |

> ⚠️ **32 GB 显存 (RTX 5090 / V100) 不在官方推荐清单**。官方最低是 40 GB+。本项目不会主动推荐 32 GB 显卡跑训练, 因为：
> - 官方默认配置 peak VRAM ~35 GB (已 3 GB 溢出)
> - 32 GB 显卡需要 `--batch-size 1 --grad-accum 16` + 开 gradient checkpointing, 训练速度慢
> - V100 + CUDA 13.0 还有 sm_70 二进制兼容风险
>
> 如果一定要用, 建议用 `02_upload_to_autodl.sh` 上传后, 在云端先跑 `--epochs 1` smoke test 验证再续费。

### 软件需求 (官方要求)

| 项目 | 版本 |
|------|------|
| Python | 3.10 |
| CUDA | 12.6+ (dGPU) / 13.0 (Jetson Thor/DGX Spark) |
| PyTorch | 2.7+ |
| OS | Ubuntu 22.04+ |
| 包管理器 | `uv` (官方推荐) |

> 镜像选择 AutoDL 市场搜索 **"PyTorch 2.7+ CUDA 12.6+"**。不推荐 PyTorch 2.11 / CUDA 12.4 老镜像 (官方已升级到 2.7+)。
> 国内网络可加 `--hf-endpoint https://hf-mirror.com` 和 `--pip-mirror aliyun` 加速 (见 `00_autodl_init.sh --help`)。

### 训练策略 (官方默认 vs 进阶)

| 模式 | 调优范围 | Peak VRAM | 速度 | 适用 |
|------|---------|-----------|------|------|
| **官方默认** ⭐ | projector + diffusion action head | < 35 GB | 100% | 大多数场景, 40 GB+ 显卡推荐 |
| `--tune-llm` | + LLM backbone | 80 GB+ | 慢 | 大数据量, 数据复杂时 |
| `--tune-visual` | + visual encoder (ViT) | 80 GB+ | 慢 | 视觉差异大时 |
| ⚠️ **LoRA** | — | — | — | **官方不支持** (`peft` 是依赖但未调用) |

> **本项目训练不依赖 LoRA**。Isaac-GR00T 官方训练策略是: 全模型加载 + 只更新 projector/diffusion (默认) 或 LLM/visual (进阶)。训练输出 (BF16) 本身就是完整可推理模型, **无需 merge 步骤**。

### 存储需求 (官方建议)

| 项目 | 最低 | 推荐 | 备注 |
|------|------|------|------|
| 系统盘 | 30 GB | 50 GB | Ubuntu + PyTorch + GR00T 仓库 |
| 数据盘 | 100 GB | 500 GB+ | 数据集 (~50 episodes ≈ 50-100 GB) + checkpoint (~25 GB) + INT4 (~1.5 GB) |
| 网络存储 | — | NFS/S3 | 生产环境 (官方推荐) |

### 内存 / CPU

- **系统 RAM**: 官方未明确要求, 至少 32 GB。数据集通过 `--num-shards-per-epoch` 控制预加载量, 内存不足可减小该值。
- **CPU**: 8 核+ (官方未明确, GR00T 数据加载是 CPU 密集型, dataloader 默认 2 workers)

### 多卡训练 (进阶)

官方支持 `torchrun --nproc_per_node=N` 多卡训练。本项目默认单卡, 多卡时需要:
- 在云端手动跑 `torchrun`, 或
- 修改 `03_autodl_train.sh` 调用, 加 `--num-gpus N` (官方参数)

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

### Q5: INT4 量化需要重新训练吗？

**不需要**。GR00T 项目用的是 **Post-Training Quantization (PTQ)**, 只是把 BF16 权重转换为 INT4 表示, 没有反向传播, **纯转换操作**:

```python
# 伪代码: 量化前后权重等价, 仅存储格式不同
bf16_weights = load_model()            # 加载训练好的 BF16 完整模型 (7GB)
int4_weights = quantize_to_int4(bf16_weights)  # 仅转换 (输出 1.5GB)
save(int4_weights)
```

**本地 8GB 显存可以完成 INT4 量化**:
- 量化峰值显存 ~2-3GB (GR00T-N1.7-3B)
- 耗时 5-15 分钟
- 量化后模型正好适合 8GB 显存推理

详见上文 [🎯 模型推理: BF16 完整模型 vs INT4 量化](#-模型推理-bf16-完整模型-vs-int4-量化) 的 v2 优化版说明。

---

## 📜 引用

- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) — 官方 mjlab 项目
- [mjlab](https://github.com/mujocolab/mjlab) — MuJoCo RL 框架
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA GR00T N1.7
- [LeRobot](https://github.com/huggingface/lerobot) — 数据格式标准
