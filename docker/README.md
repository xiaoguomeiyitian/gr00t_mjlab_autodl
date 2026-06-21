# gr00t-mjlab 本地 Docker 方案

## 设计原则

**训练不在本地 Docker 中进行**。完整训练流程在 AutoDL 云端跑:

```
┌──────────────────────────────────────────┐    ┌────────────────────────────┐
│  本地 (Docker)                            │    │  AutoDL 云端               │
│                                          │    │                            │
│  ┌──────────────────────────────────┐   │    │  - Isaac-GR00T             │
│  │ gr00t-mjlab-collect   [1] 数据采集│ ─────────▶  03_autodl_train.sh    │
│  │ gr00t-mjlab-infer     [6] 推理验证│ ◀─────────  SCP 下载模型包       │
│  └──────────────────────────────────┘   │    │                            │
│                                          │    │                            │
│  (上传 [2] + 下载 [4] 在主机跑 shell)    │    │                            │
└──────────────────────────────────────────┘
```

| 步骤 | 运行环境 | 入口 |
|------|---------|------|
| [1] 数据采集 | **本地 Docker (collect)** | `./start.sh collect` |
| [2] 上传云端 | **本地主机 shell** | `./scripts/02_upload_to_autodl.sh` |
| [3] 云端训练 | **AutoDL** | `bash 03_autodl_train.sh` (云端) |
| [4] 下载模型 | **本地主机 shell** | `./scripts/04_download_model.sh` |
| [5] 本地量化 | **本地主机 shell** | `./scripts/05_local_quantize.sh` |
| [6] 推理验证 | **本地 Docker (infer)** | `./start.sh infer` |

---

## 两个本地镜像

| 镜像 | 大小 | 包含 | 用途 |
|------|------|------|------|
| `gr00t-mjlab-collect:latest` | ~5GB  | base (CUDA + mjlab + 采集脚本) | [1] 数据采集 + 转换打包 |
| `gr00t-mjlab-infer:latest`   | ~7GB  | base + transformers + peft + bnb | [6] 推理验证 (支持 INT4) |

`gr00t-mjlab-base:latest` (~5GB) 是两镜像的共享基础。

> **不构建 train 镜像**: 训练在 AutoDL, 不在本地 Docker。

---

## 构建

```bash
cd gr00t_mjlab_autodl/    # 项目根

# 交互式
./build.sh

# 非交互式
./build.sh collect                # 只构建 collect
./build.sh infer                  # 只构建 infer
./build.sh all                    # 构建 collect + infer
./build.sh --mirror cn all        # 国内镜像源 (apt + pip)
./build.sh --base                 # 只构建基础
./build.sh --rebuild-base collect # 强制重建 base + collect
./build.sh --no-rebuild-base infer # 只更新 infer 层, 跳过 base
```

---

## 启动

```bash
cd gr00t_mjlab_autodl/    # 项目根

# 交互式
./start.sh

# 非交互式
./start.sh collect                                # [1] 数据采集
./start.sh infer --robot g1                       # [6] 推理验证
./start.sh infer --robot g1 --model-path /root/models/g1_gr00t_int4
./start.sh shell                                  # 进入 collect 镜像的 shell

# 环境变量 (从宿主机透传到容器)
ROBOT=g1 NUM_EPISODES=200 ./start.sh collect
INSTRUCTION="walk forward" MODEL_PATH=/root/models/g1_gr00t_int4 ./start.sh infer
```

> **注意**: 训练 (`./start.sh train`) 已移除, 用 `./scripts/02_upload_to_autodl.sh` + 云端 `03_autodl_train.sh`。

---

## 容器内目录布局

```
/root/
├── gr00t_mjlab_autodl/          # 本项目 (含 .git/)
├── unitree_rl_mjlab/             # 官方 mjlab (含 .git/, editable)
├── data/                         # 训练数据 (与宿主机双向同步)
└── models/                       # 模型权重 (与宿主机双向同步)
```

> AutoDL 镜像发布要求 `/root/` 下是 Git 仓库, 本项目 + mjlab 都满足。

---

## 与本地工具的关系

| 工具 | 是否需要 Docker | 说明 |
|------|---------------|------|
| `./start.sh collect` | ✅ collect 镜像 | [1] 数据采集 |
| `./start.sh infer` | ✅ infer 镜像 | [6] 推理验证 |
| `./scripts/02_upload_to_autodl.sh` | ❌ 主机 | [2] SCP 上传 (只需 ssh/scp) |
| `./scripts/03_autodl_train.sh` | — 云端 | [3] AutoDL 上跑 (云端) |
| `./scripts/04_download_model.sh` | ❌ 主机 | [4] SCP 下载 (只需 ssh/scp) |
| `./scripts/05_local_quantize.sh` | ❌ 主机 | [5] FP16 → INT4 (本地 Python 即可) |
| `./scripts/06_local_verify.sh` | ✅ infer 镜像 | [6] 推理 (也可用 start.sh) |

> [2]/[4] 是纯 SSH/SCP, 不需要 Docker;
> [5] 是 Python 推理, 可以在主机或 infer 容器内跑 (有 GPU 即可)。

---

## 文件结构

```
gr00t_mjlab_autodl/                  # 项目根
├── build.sh                          # Docker 镜像构建 (上移到根)
├── start.sh                          # Docker 容器启动 (上移到根, 2 镜像模式)
└── docker/                           # Dockerfile + 调度脚本 (维护用)
    ├── Dockerfile.base               # 共享基础
    ├── Dockerfile.collect            # 镜像 1: 数据采集
    ├── Dockerfile.infer              # 镜像 2: 推理验证
    ├── scripts/
    │   ├── entrypoint.sh             # 统一入口 (mode → 调度)
    │   ├── collect.sh                # [1] 调度脚本
    │   └── infer.sh                  # [6] 调度脚本
    └── README.md                     # 本文件
```

---

## 关键技术决策

1. **共享 base 层** — collect + infer 复用 `gr00t-mjlab-base`,节省 ~5GB
2. **CUDA 锁版本** — `torch==2.11.0+cu128` + `mujoco<3.6.0` + `warp<1.13.0`,版本经过严格对齐避免 ABI 不兼容
3. **infer 不装完整 Isaac-GR00T** — 只装 gr00t Python 包 (~500MB 而非 ~5GB)
4. **挂载策略** — 只挂载 `gr00t_mjlab_autodl/` + `data/` + `models/`,源仓库 (mjlab) COPY 进镜像
5. **环境变量透传** — ROBOT/NUM_EPISODES 等从宿主机传到容器,方便 shell 集成
6. **训练外置到云端** — 节省本地 ~12GB 镜像 + ~30 分钟构建时间;云端按需租用,镜像现成
