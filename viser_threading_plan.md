# Viser 采集与渲染完全线程化分离方案

## 问题描述

当前架构中，主线程同步执行 `env.step()` + `viser.update()`，导致：
1. 主线程全速占用 CPU，viser 的 websocket 服务无法响应浏览器连接
2. 即使加了后台线程检测连接，主线程仍然阻塞了 viser 的 asyncio event loop

## 目标

- 主线程全力采集，不受 viser 渲染影响
- viser 渲染和连接处理完全在独立线程
- 浏览器连接后自动恢复渲染，断开后自动暂停

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│  主线程 (采集线程) — 全力采集，不渲染                                 │
│                                                                     │
│  for ep in range(num_episodes):                                     │
│      for step in range(episode_length):                             │
│          action = get_action(step)                                  │
│          obs, reward, done, info = env.step(action)  ← 物理仿真      │
│          save_npz(observations, actions)                             │
│          shared_state['qpos'] = qpos  ← 写入共享内存 (线程安全)      │
│          shared_state['qvel'] = qvel                                 │
│          shared_state['progress'] = (ep, step)                      │
│          time.sleep(0.001)  # 1ms 让出 CPU 给其他线程               │
│                                                                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ 共享状态 (threading.Lock 保护)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Viser 渲染线程 — 独立运行, 从共享状态读取                             │
│                                                                     │
│  def _render_loop():                                                │
│      while not stop:                                                │
│          if has_connections:                                        │
│              qpos = shared_state['qpos']  ← 从共享内存读取           │
│              qvel = shared_state['qvel']                             │
│              sync_to_viser(qpos, qvel)                               │
│              render_scene()                                          │
│              sleep(1/fps)                                            │
│          else:                                                      │
│              sleep(0.5)  # 无连接时休眠, 节省 CPU                     │
│                                                                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ websocket (viser 内部独立线程)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Viser HTTP/WS 服务线程 (viser 内部)                                  │
│                                                                     │
│  - 自动处理浏览器连接/断开                                            │
│  - on_client_connect 回调修改 connection_count                      │
│  - 完全独立于主线程和渲染线程                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 线程间通信

使用 `threading.Lock` 保护的共享状态字典：

```python
shared_state = {
    'qpos': None,       # np.ndarray, 关节位置
    'qvel': None,       # np.ndarray, 关节速度
    'ep': 0,            # 当前 episode
    'step': 0,          # 当前步数
    'total_ep': 0,      # 总 episode 数
    'total_step': 0,    # 每 episode 步数
    'instruction': '',  # 当前指令
}
```

## 实现改动

### 1. `src/viser_3d_viewer.py` — AsyncViser3DViewer

**改动**：
- 新增 `shared_state` 参数，渲染线程从共享状态读取 qpos/qvel
- 移除 `update()` 方法（主线程不再调用）
- 新增 `start_rendering()` 方法，启动渲染线程
- 新增 `stop_rendering()` 方法，停止渲染线程
- 渲染线程逻辑：有连接 → 从共享状态读取并渲染；无连接 → sleep(0.5)

**关键代码**：

```python
class AsyncViser3DViewer:
    def __init__(self, env, port=20006, env_idx=0, frame_rate=30.0, show_debug=False):
        # ... 现有初始化 ...
        self._shared_state = None  # 外部注入的共享状态

    def set_shared_state(self, state: dict, lock: threading.Lock):
        """设置共享状态 (主线程写入, 渲染线程读取)."""
        self._shared_state = state
        self._state_lock = lock

    def start_rendering(self):
        """启动独立渲染线程."""
        self._render_thread = threading.Thread(
            target=self._render_loop, name="viser-render", daemon=True
        )
        self._render_thread.start()

    def _render_loop(self):
        """渲染线程主循环 — 从共享状态读取."""
        while not self._stop.is_set():
            if self.has_connections and self._shared_state is not None:
                with self._state_lock:
                    qpos = self._shared_state.get('qpos')
                    qvel = self._shared_state.get('qvel')
                if qpos is not None:
                    self._sync_to_viser(qpos, qvel)
                    self._render_count += 1
                time.sleep(1.0 / self._frame_rate)
            else:
                time.sleep(0.5)  # 无连接时休眠
```

### 2. `src/collect_data.py` — ViserViewer + collect_demonstrations

**ViserViewer 改动**：
- `__init__` 中创建共享状态字典和 Lock
- 将共享状态注入 `AsyncViser3DViewer`
- 启动渲染线程 (`start_rendering()`)
- `update()` 方法改为只更新共享状态，不调用 viser 渲染
- `close()` 中停止渲染线程

**collect_demonstrations 改动**：
- 主循环中不再调用 `viewer.update()` 进行渲染
- 改为更新 `shared_state` 中的 qpos/qvel
- 每步 `time.sleep(0.001)` (1ms) 让出 CPU 给其他线程

**关键代码**：

```python
class ViserViewer:
    def __init__(self, env=None, port=20006, viser_fps=30.0):
        # ... 初始化 viser server ...
        # 创建共享状态
        self._shared_state = {'qpos': None, 'qvel': None, 'ep': 0, 'step': 0}
        self._state_lock = threading.Lock()
        # 注入共享状态到 3D viewer
        if self._viewer_3d:
            self._viewer_3d.set_shared_state(self._shared_state, self._state_lock)
            self._viewer_3d.start_rendering()

    def update(self, ep, total_ep, step, total_step, instruction, base_vel=None, joint_targets=None):
        # 只更新共享状态, 不渲染
        if self._viewer_3d:
            qpos, qvel = self._get_current_state()
            with self._state_lock:
                self._shared_state['qpos'] = qpos
                self._shared_state['qvel'] = qvel
                self._shared_state['ep'] = ep
                self._shared_state['step'] = step
        # 文本 viewer 更新 (开销极小)
        self._update_text_viewer(...)

    def close(self):
        if self._viewer_3d:
            self._viewer_3d.stop_rendering()
        # ... 其余清理 ...
```

### 3. 主循环改动 (collect_demonstrations)

```python
for ep in range(num_episodes):
    for step in range(episode_length):
        # 1. 决定动作
        joint_targets = gait_fn(step, dt=dt, speed=ep_speed, ...)

        # 2. 推环境
        obs, reward, done, info = env_raw.step(action_tensor)

        # 3. 保存数据
        np.savez_compressed(...)

        # 4. 更新共享状态 (供渲染线程使用)
        viewer.update(ep, num_episodes, step, episode_length, instruction)

        # 5. 每步让出 1ms CPU 给渲染线程
        time.sleep(0.001)
```

## 性能分析

| 指标 | 当前方案 | 三线程方案 |
|------|---------|-----------|
| 主线程 CPU 占用 | ~100% (含渲染) | ~99% (仅采集, 1ms sleep) |
| 渲染线程 CPU 占用 | 0 (与主线程共享) | ~30% (30 FPS 渲染) |
| 浏览器连接延迟 | 可能饿死 | ≤0.5s (viser 独立线程) |
| 采集效率 | 正常 | 无影响 (1ms sleep 可忽略) |
| 内存开销 | 低 | 低 (共享状态很小) |

## 关键设计决策

1. **共享状态用 Lock 而非 Queue**：qpos/qvel 是高频率读写，Lock 比 Queue 更高效
2. **每步 sleep(0.001)**：1ms 让出 CPU 足够让渲染线程获得执行机会，对采集效率影响 <0.1%
3. **渲染线程无连接时 sleep(0.5)**：避免空转浪费 CPU
4. **on_client_connect 回调直接修改状态**：由 viser 内部 WS 线程触发，零延迟

## 实施步骤

1. 修改 `src/viser_3d_viewer.py`：新增共享状态、独立渲染线程
2. 修改 `src/collect_data.py`：ViserViewer 改为更新共享状态，主循环加 sleep
3. 运行测试确认功能正确
4. 实际采集验证：启动 → 浏览器连接 → 观察渲染恢复
