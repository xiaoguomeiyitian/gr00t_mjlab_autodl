# 改造方案：robot_retargeter → GR00T 微调数据桥接

> 版本：v1.0 | 2026-06-29

## 1. 目标

将 `robot_retargeter` 产出的**真实人体动作**（经 IK 重定向后的机器人关节轨迹）转换为 GR00T LeRobot v2 格式，替代当前 `collect_data.py` 生成的随机/模拟数据，使 GR00T-N1.7-3B 能学习到有意义的运动模式。

同时支持通过 `unitree_rl_mjlab` 训练的 RL 策略在仿真中采集带物理交互的高质量数据。

---

## 2. 核心数据流

```
robot_retargeter                    gr00t_mjlab_autodl（本项目）
─────────────────                   ──────────────────────────
SMPL-X .npz / Robot CSV
        ↓
  robot_retarget (IK) → qpos CSV
        ↓
  export_npz.py → NPZ
        ↓
  ★ retarget_to_lerobot.py ★  →  LeRobot v2 数据集  →  GR00T 微调
```

---

## 3. 关键差异与挑战

| 维度 | robot_retargeter 输出 | GR00T 训练需要 |
|------|----------------------|----------------|
| 关节轨迹 | CSV: `[pos_xyz(3), quat_xyzw(4), joints(N)]` | NPZ: `states(T,71), actions(T,29)` |
| 动作定义 | 绝对关节角 qpos | delta 关节增量 |
| 图像 | 无（纯运动学） | 需要 mp4 视频 |
| 语言标签 | 无 | 需要 task_description |
| Episode | 单条连续轨迹 | 多 episode 分段 |

---

## 4. 实施步骤

### Phase 1：核心桥接脚本

#### 4.1 创建 `src/retarget_motion_loader.py` — 运动数据加载器

功能：
- 解析 robot_retargeter 的 qpos CSV 格式：`[pos_xyz(3), quat_xyzw(4), joints(N)]`
- 解析 robot_retargeter 的 NPZ 格式（`export_npz.py` 输出）
- 统一接口：返回 `(base_pos, base_quat, joint_pos, fps)`

#### 4.2 创建 `src/mujoco_renderer.py` — MuJoCo 渲染器

功能：
- 加载机器人 MJCF 模型（复用 `robot_retargeter/asset/robot/g1_description/mjcf/g1.xml`）
- 设置相机视角（front, wrist）
- 逐帧渲染 → 输出 mp4

#### 4.3 创建 `src/retarget_to_lerobot.py` — 核心转换脚本

功能：
1. 加载 CSV/NPZ → 解析 qpos（base_pos + base_quat + joint_pos）
2. 计算状态向量（71 维）：
   - `joint_pos` (29) + `joint_vel` (29, 差分) + `base_pos` (3) + `base_quat` (4) + `base_lin_vel` (3, 差分) + `base_ang_vel` (3, 四元数差分)
3. 计算 delta 动作（29 维）：`action[t] = joint_pos[t+1] - joint_pos[t]`
4. 用 MuJoCo 渲染相机图像 → 保存 mp4
5. 滑动窗口切分 episode（每段 300 步，重叠 50%）
6. 自动生成语言标签（从文件名/动作类型推断）
7. 输出 LeRobot v2 格式

#### 4.4 创建 `src/configs/motion_labels.py` — 动作语言标签映射

```python
LABEL_MAP = {
    "walk": "walk forward",
    "run": "run forward",
    "dance": "perform dancing motion",
    "fight": "perform fighting motion",
    "jump": "jump up",
    "fall": "fall and get up",
    "sprint": "sprint forward",
    "grab": "grab object",
    ...
}
```

---

### Phase 2：集成脚本与配置

#### 4.5 创建 `scripts/10_retarget_to_lerobot.sh` — 一键转换脚本

```bash
./scripts/10_retarget_to_lerobot.sh --csv ../robot_retargeter/output_data/robot_motion/xxx_g1.csv --robot g1 --output output/g1_from_retarget
```

#### 4.6 创建 `scripts/11_batch_retarget.sh` — 批量转换

```bash
./scripts/11_batch_retarget.sh --robot g1 --input-dir ../robot_retargeter/output_data/robot_motion/ --output-dir output/g1_all
```

#### 4.7 更新 `start.sh` — 添加菜单选项

新增选项：
- `13) 从 robot_retargeter 动作生成训练数据`
- `14) RL 策略仿真采集`

---

### Phase 3：测试

#### 4.8 创建 `tests/test_retarget_to_lerobot.py`

测试内容：
- CSV 加载与解析
- 状态向量计算正确性（维度、数值）
- Delta 动作计算正确性
- Episode 切分逻辑
- LeRobot v2 输出格式验证
- 端到端集成测试

---

### Phase 4：RL 策略采集数据（后续优化）

#### 4.9 创建 `src/rl_episode_collector.py` — RL 仿真采集器

功能：
- 加载 unitree_rl_mjlab 训练的 RL 策略（`.pt` 或 `.onnx`）
- 在 MJLab 仿真环境中运行策略
- 每步采集：observation + action + reward + 相机图像
- 输出 LeRobot v2 格式

#### 4.10 创建 `scripts/12_rl_collect.sh` — 一键 RL 采集

```bash
./scripts/12_rl_collect.sh --checkpoint ../checkpoints/model_2000.pt --task Unitree-G1-Tracking-No-State-Estimation --robot g1 --num-episodes 100 --output output/g1_rl_raw
```

---

## 5. 新增文件清单

| 文件 | 功能 | Phase |
|------|------|-------|
| `src/retarget_motion_loader.py` | 加载 robot_retargeter CSV/NPZ 数据 | 1 |
| `src/mujoco_renderer.py` | MuJoCo 离线渲染相机图像 | 1 |
| `src/retarget_to_lerobot.py` | 核心转换：动作数据 → LeRobot v2 | 1 |
| `src/configs/motion_labels.py` | 动作文件名 → 语言标签映射 | 1 |
| `scripts/10_retarget_to_lerobot.sh` | 一键转换脚本 | 2 |
| `scripts/11_batch_retarget.sh` | 批量转换脚本 | 2 |
| `start.sh`（修改） | 添加菜单选项 13、14 | 2 |
| `tests/test_retarget_to_lerobot.py` | 桥接脚本测试 | 3 |
| `src/rl_episode_collector.py` | RL 策略仿真采集 | 4 |
| `scripts/12_rl_collect.sh` | 一键 RL 采集脚本 | 4 |

---

## 6. 数据格式规范

### 输入：robot_retargeter qpos CSV

```
# 格式：[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_0, joint_1, ..., joint_28]
0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.1, -0.2, ..., 0.05
```

### 输出：LeRobot v2

```
{robot}_lerobot/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── modality.json
├── data/chunk-000/
│   └── episode_000000.parquet
└── videos/chunk-000/
    ├── episode_000000.mp4
    └── ...
```

### 状态向量（G1, 71 维）

| 索引 | 内容 | 维度 |
|------|------|------|
| [0:29] | joint_pos | 29 |
| [29:58] | joint_vel | 29 |
| [58:61] | base_pos | 3 |
| [61:65] | base_quat | 4 |
| [65:68] | base_lin_vel | 3 |
| [68:71] | base_ang_vel | 3 |

### 动作向量（G1, 29 维）

- `action[t] = joint_pos[t+1] - joint_pos[t]`（delta 增量）
- 与 `g1_modality_config.py` 中 `ActionRepresentation.RELATIVE` 一致

---

## 7. 使用流程

### 7.1 运动学数据（Phase 1-3）

```bash
# 1. 在 robot_retargeter 中重定向动作
cd /root/unitree/robot_retargeter
./start.sh smpl --motion dataset/ACCAD/Form_1_stageii.npz --robots g1

# 2. 转换为本项目格式
cd /root/unitree/gr00t_mjlab_autodl
./start.sh retarget-to-lerobot \
    --csv ../robot_retargeter/output_data/robot_motion/Form_1_stageii_g1.csv \
    --robot g1 \
    --output output/g1_from_retarget

# 3. 上传到 AutoDL 并训练
./start.sh upload g1 output/g1_from_retarget
./start.sh train g1
```

### 7.2 RL 采集数据（Phase 4）

```bash
# 1. 训练 RL 策略（在 AutoDL 云端）
cd /root/unitree/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
    --motion_file=src/assets/motions/g1/dance1_subject2.npz

# 2. 下载策略
scp root@autodl:logs/rsl_rl/g1_tracking/.../model_2000.pt /root/unitree/gr00t_mjlab_autodl/checkpoints/

# 3. RL 仿真采集
cd /root/unitree/gr00t_mjlab_autodl
./start.sh rl-collect \
    --checkpoint checkpoints/model_2000.pt \
    --task Unitree-G1-Tracking-No-State-Estimation \
    --robot g1 \
    --num-episodes 100 \
    --output output/g1_rl_raw

# 4. 转换并训练
./start.sh upload g1 output/g1_rl_raw
./start.sh train g1
```

---

## 8. 验证计划

| 步骤 | 命令 | 预期结果 |
|------|------|----------|
| 1 | `python -m src.retarget_to_lerobot --csv xxx.csv --robot g1 --output /tmp/test` | 成功转换 |
| 2 | `ls /tmp/test/meta/` | info.json, episodes.jsonl, tasks.jsonl, modality.json |
| 3 | `ls /tmp/test/data/chunk-000/` | episode_000000.parquet |
| 4 | `ls /tmp/test/videos/chunk-000/` | episode_000000.mp4, ... |
| 5 | `python -c "import pandas as pd; df=pd.read_parquet('/tmp/test/data/chunk-000/episode_000000.parquet'); print(df.shape)" | 行数 > 0, 列正确 |
| 6 | `pytest tests/test_retarget_to_lerobot.py -v` | 全部通过 |
| 7 | `pytest tests/ -v` | 116 + 新增测试全部通过 |

---

## 9. 设计决策

| 决策 | 说明 |
|------|------|
| 动作表示 | 使用 delta（相对增量），与 `g1_modality_config.py` 一致 |
| 图像来源 | MuJoCo 离线渲染（robot_retargeter 不产生真实相机图像） |
| Episode 切分 | 滑动窗口 300 步，重叠 50% |
| 语言标签 | 从文件名自动推断 + 支持手动覆盖 |
| 不修改 robot_retargeter | 保持两个项目独立，桥接在本项目侧 |

---

## 10. 注意事项

1. MuJoCo 渲染的图像与真实相机差异较大，微调后模型在真实机器人上可能需要 domain adaptation
2. 批量处理 40+ 动作时，渲染是瓶颈，可考虑并行化
3. RL 采集需要 GPU 训练策略，建议在 AutoDL 云端完成
4. 混合训练数据（运动学 + RL 采集）可提高模型泛化能力
