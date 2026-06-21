# gr00t-mjlab Docker 镜像方案

## ⚠️ AutoDL 镜像发布合规

镜像设计遵循 **AutoDL 镜像发布要求**:
> 审核标准: 在 `/root` 目录下包含相应 Git 代码仓库, 且可以成功执行 Git 代码仓库中的简单代码

本方案实施:
1. **`/root/gr00t_mjlab_autodl/`** — `gr00t_mjlab_autodl` 项目源码 (含 `.git/`, `README.md`, `src/`, `scripts/`, `docker/`, `requirements.txt`)
2. **`/root/unitree_rl_mjlab/`** — `unitree_rl_mjlab` 项目源码 (含 `.git/`, editable 安装)
3. **`/root/unitree_sdk2_python/`** — `unitree_sdk2_python` 项目源码 (含 `.git/`, editable 安装)
4. **`/root/Isaac-GR00T/`** — `train` 镜像专属, GR00T 官方仓库 (含 `.git/`, `uv sync` 安装)
5. **健康检查** — base 镜像构建阶段执行 `git log` + `python3` 导入验证, 证明代码可执行

任何镜像内均可直接:
```bash
cd /root/gr00t_mjlab_autodl && python3 -c "import sys; sys.path.insert(0,'.'); from configs import g1_config; print('OK')"
cd /root/Isaac-GR00T && python3 -c "import gr00t; print('OK')"
```

## 三个独立镜像

| 镜像 | 大小 | 包含 | 用途 |
|------|------|------|------|
| `gr00t-mjlab-collect:latest` | ~5.1GB | base (CUDA + mjlab + 脚本) | [1] 数据采集 + 转换打包 |
| `gr00t-mjlab-train:latest`   | ~12GB  | base + Python 3.10 + transformers + peft + bnb + accelerate + Isaac-GR00T | [2] GR00T 云端训练 |
| `gr00t-mjlab-infer:latest`   | ~7GB   | base + transformers + peft + bnb (无 Isaac-GR00T) | [4] 本地推理验证 |

`gr00t-mjlab-base:latest` (~5GB) 是三镜像的共享基础。

## 构建

```bash
cd /home/kxy/work/unitree/gr00t_mjlab_autodl    # 项目根

# 交互式
./build.sh

# 非交互式
./build.sh collect                # 只构建 collect
./build.sh train                  # 只构建 train
./build.sh infer                  # 只构建 infer
./build.sh all                    # 构建全部 (按依赖顺序)
./build.sh --mirror cn all        # 国内镜像源 (apt + pip)
./build.sh --base                 # 只构建基础
./build.sh --rebuild-base collect # 强制重建 base + collect
./build.sh --no-rebuild-base train # 只更新 train 层, 跳过 base
```

## 启动

```bash
cd /home/kxy/work/unitree/gr00t_mjlab_autodl    # 项目根

# 交互式
./start.sh

# 非交互式
./start.sh collect                                # 进入 collect 模式
./start.sh train --robot g1 --epochs 20           # 进入 train 模式
./start.sh infer --robot g1                       # 进入 infer 模式
./start.sh shell                                  # 进入 collect 镜像的 shell

# 环境变量 (从宿主机透传到容器)
ROBOT=g1 NUM_EPISODES=200 ./start.sh collect
INSTRUCTION="walk forward" MODEL_PATH=/root/models/g1_gr00t_int4 ./start.sh infer
```

## 容器内目录布局

```
/root/
├── gr00t_mjlab_autodl/          # 本项目 (含 .git/, AutoDL 合规)
├── unitree_rl_mjlab/             # 官方 mjlab (含 .git/, editable)
├── unitree_sdk2_python/          # DDS SDK (含 .git/, editable)
├── Isaac-GR00T/                  # train 镜像专属 (含 .git/)
├── data/                         # 训练数据 (与宿主机双向同步)
└── models/                       # 模型权重 (与宿主机双向同步)
```

## 与现有工具的关系

| 工具 | 镜像 | 用途 |
|------|------|------|
| `./start.sh` (项目根) | gr00t-mjlab-* (自建) | **本项目自有三镜像启动** |
| `run_all.sh` / `scripts/01..04.sh` | 无 (直接运行) | 本地无容器场景 |

## 文件结构

```
gr00t_mjlab_autodl/                  # 项目根
├── build.sh                          # Docker 镜像构建 (上移到根)
├── start.sh                          # Docker 容器启动 (上移到根)
└── docker/                           # Dockerfile + 调度脚本 (维护用)
    ├── Dockerfile.base               # 共享基础
    ├── Dockerfile.collect            # 镜像 1
    ├── Dockerfile.train              # 镜像 2
    ├── Dockerfile.infer              # 镜像 3
    ├── scripts/
    │   ├── entrypoint.sh             # 统一入口 (mode → 调度)
    │   ├── collect.sh                # 数据采集
    │   ├── train.sh                  # 训练
    │   └── infer.sh                  # 推理
    └── README.md                     # 本文件
```

## 关键技术决策

1. **共享 base 层** — 三个变体复用 `gr00t-mjlab-base`,节省 ~10GB
2. **CUDA 锁版本** — `torch==2.11.0+cu128` + `mujoco<3.6.0` + `warp<1.13.0`,版本经过严格对齐避免 ABI 不兼容
3. **train 用 Python 3.10** — GR00T 官方要求,在 base 之外独立装
4. **infer 不装完整 Isaac-GR00T** — 只装 gr00t Python 包 (~500MB 而非 ~5GB)
5. **挂载策略** — 只挂载 `gr00t_mjlab_autodl/` + `data/` + `models/`,源仓库 (mjlab, sdk) 全部 COPY 进镜像
6. **环境变量透传** — ROBOT/NUM_EPISODES 等从宿主机传到容器,方便 shell 集成
