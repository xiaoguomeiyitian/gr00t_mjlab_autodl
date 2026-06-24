#!/usr/bin/env python3
"""AsyncViser3DViewer — 异步 Viser 3D 可视化, 不阻塞主线程 env.step().

后台渲染线程以 ~30 fps 渲染, 主线程调用 notify_new_frame() 通知新数据就绪.
使用 threading.Event 同步 (无锁争用), 复用 mjlab 的 ViserMujocoScene 渲染 3D 机器人场景.

架构:
  - 主线程 (采集): env.step() → 写共享状态 → notify_new_frame() (< 1μs)
  - 渲染后台线程 (Daemon): wait(swap_event) → 读共享状态 → 渲染
  - Viser WS 服务线程 (Daemon): 独立处理浏览器连接
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AsyncViser3DViewer:
    """异步 Viser 3D 可视化 — 后台线程渲染, 主线程无阻塞.

    Args:
        env: mjlab ManagerBasedRlEnv 实例 (或 unwrapped env)
        port: Viser HTTP/WS 端口 (浏览器访问 http://<host>:<port>)
        env_idx: 要显示的环境索引 (多 env 时)
        frame_rate: 目标渲染帧率 (fps)
        show_debug: 是否显示 debug 可视化 (接触力, 坐标系等)
    """

    def __init__(
        self,
        env: Any,
        port: int = 20006,
        env_idx: int = 0,
        frame_rate: float = 30.0,
        show_debug: bool = False,
    ):
        self._env = env
        self._port = port
        self._env_idx = env_idx
        self._frame_rate = frame_rate
        self._show_debug = show_debug

        self._server: Any = None
        self._scene: Any = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stop = threading.Event()

        # ── 共享状态 (主线程写, 渲染线程读) ──
        self._shared_state: dict[str, Any] | None = None
        self._state_lock: threading.Lock | None = None

        # ── 帧同步 Event (无锁) ──
        self._swap_event = threading.Event()

        # ── 统计 ──
        self._render_count = 0
        self._error_count = 0
        self._last_render_time = 0.0

        # ── 连接状态追踪 (自适应 FPS) ──
        self._connection_count = 0
        self._connection_count_lock = threading.Lock()
        self._paused = False  # True = 无连接时暂停渲染

    # ── 公开 API ──

    def set_shared_state(self, state: dict, lock: threading.Lock) -> None:
        """设置共享状态 (主线程写入, 渲染线程读取).

        Args:
            state: 共享状态字典, 包含 'qpos', 'qvel' 等键
            lock: 保护共享状态的 threading.Lock
        """
        self._shared_state = state
        self._state_lock = lock

    def start(self) -> None:
        """启动后台渲染线程."""
        if self._thread is not None:
            logger.warning("Viser viewer 已在运行, 忽略重复 start()")
            return

        self._stop.clear()
        self._running.set()
        self._thread = threading.Thread(
            target=self._render_loop,
            name="viser-3d-viewer",
            daemon=True,
        )
        self._thread.start()
        logger.info("🌐 Viser 3D viewer 已启动: http://0.0.0.0:%d", self._port)

    def stop(self) -> None:
        """停止后台渲染线程."""
        self._stop.set()
        self._swap_event.set()  # 唤醒等待中的渲染线程
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Viser 3D viewer 已停止 (共渲染 %d 帧, %d 次异常)",
                     self._render_count, self._error_count)

    def notify_new_frame(self) -> None:
        """主线程调用, 通知渲染线程新数据就绪.

        开销 < 1μs, 不阻塞主线程.
        """
        self._swap_event.set()

    def update(self) -> None:
        """同步最新 env 状态到共享状态 + 通知渲染线程 (非阻塞).

        主线程每步调用, 仅写共享状态和 set event, 不做任何渲染.
        """
        try:
            env = self._env
            if env is None:
                return

            sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
            mj_data = sim.mj_data

            qpos = np.asarray(mj_data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(mj_data.qvel, dtype=np.float64).copy()

            if self._shared_state is not None and self._state_lock is not None:
                with self._state_lock:
                    self._shared_state['qpos'] = qpos
                    self._shared_state['qvel'] = qvel

            self._swap_event.set()
        except Exception as e:
            logger.debug("Viser update 失败: %s", e)

    @property
    def url(self) -> str:
        return f"http://0.0.0.0:{self._port}"

    @property
    def render_count(self) -> int:
        return self._render_count

    @property
    def has_connections(self) -> bool:
        """是否有浏览器连接."""
        return self._connection_count > 0

    @property
    def is_alive(self) -> bool:
        """渲染线程是否存活."""
        return self._thread is not None and self._thread.is_alive()

    def set_fps(self, fps: float) -> None:
        """动态调整渲染 FPS (0 = 暂停渲染)."""
        self._frame_rate = max(0.0, fps)
        if fps > 0:
            self._paused = False
            logger.info("Viser 渲染 FPS 调整为 %.1f", fps)
        else:
            self._paused = True
            logger.info("Viser 渲染暂停 (无连接)")

    def get_render_stats(self) -> dict[str, Any]:
        """查询渲染统计信息."""
        return {
            "alive": self.is_alive,
            "total_frames": self._render_count,
            "error_count": self._error_count,
            "is_connected": self.has_connections,
        }

    def _on_connect(self, connection: Any) -> None:
        """浏览器连接回调."""
        with self._connection_count_lock:
            self._connection_count += 1
            logger.info("🌐 Viser 浏览器连接 (共 %d 个)", self._connection_count)

    def _on_disconnect(self, connection: Any) -> None:
        """浏览器断开回调."""
        with self._connection_count_lock:
            self._connection_count = max(0, self._connection_count - 1)
            logger.info("🌐 Viser 浏览器断开 (剩余 %d 个)", self._connection_count)

    # ── 内部实现 ──

    def _render_loop(self) -> None:
        """后台渲染线程主循环 — 双唤醒: Event + 连接检测.

        策略:
          - 无连接: sleep(0.5) 休眠, 节省 CPU
          - 有连接 + 新数据: 读共享状态 → 渲染
          - 有连接 + 无新数据: sleep(0.02) 等待
        """
        try:
            self._setup_viser()
        except Exception as e:
            logger.error("Viser 初始化失败: %s", e)
            logger.info("提示: pip install viser")
            return

        connected_fps = min(self._frame_rate, 30.0)
        connected_interval = 1.0 / connected_fps
        last_client_count = -1

        while not self._stop.is_set():
            try:
                # ── 1. 检查连接状态 ──
                try:
                    clients = self._server.get_clients()
                    n_clients = len(clients)
                except Exception:
                    n_clients = 1  # 无法获取时默认有连接, 避免跳过渲染

                if n_clients != last_client_count:
                    if n_clients > 0:
                        logger.info(
                            "🌐 浏览器已连接 (客户端数=%d), 渲染 @%.0fFPS",
                            n_clients, connected_fps,
                        )
                    else:
                        logger.info("🌐 无浏览器连接, 暂停渲染 (采集全速运行)")
                    last_client_count = n_clients

                if n_clients == 0:
                    # 无连接时休眠, 节省 CPU
                    time.sleep(0.5)
                    continue

                # ── 2. 等待新数据 (最多 1 秒) ──
                if not self._swap_event.wait(timeout=1.0):
                    continue  # 超时, 继续等待
                self._swap_event.clear()

                # ── 3. 帧率控制 ──
                now = time.time()
                elapsed = now - self._last_render_time
                if elapsed < connected_interval:
                    time.sleep(min(0.02, connected_interval - elapsed))
                    continue
                self._last_render_time = now

                # ── 4. 读取共享状态 (Lock 保护, 快速拷贝) ──
                qpos = None
                qvel = None
                if self._shared_state is not None and self._state_lock is not None:
                    with self._state_lock:
                        qpos = self._shared_state.get('qpos')
                        qvel = self._shared_state.get('qvel')

                if qpos is not None:
                    # ── 5. 同步到 viser 场景 ──
                    self._sync_to_viser(qpos, qvel)
                    self._render_count += 1

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 3:
                    logger.warning("Viser 渲染异常 (%d/3): %s", self._error_count, e)
                elif self._error_count == 4:
                    logger.warning("Viser 渲染异常已达 3 次, 静默忽略后续错误...")
                time.sleep(min(0.5 * self._error_count, 5.0))

    def _setup_viser(self) -> None:
        """初始化 Viser 服务器和 3D 场景."""
        import viser
        from mjlab.viewer.viser.scene import ViserMujocoScene

        env = self._env
        sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim

        self._server = viser.ViserServer(
            host="0.0.0.0",
            port=self._port,
            label="GR00T mjlab",
            verbose=False,
        )

        # 注册连接/断开回调 (自适应 FPS)
        self._server.on_client_connect(self._on_connect)
        self._server.on_client_disconnect(self._on_disconnect)

        self._scene = ViserMujocoScene.create(
            server=self._server,
            mj_model=sim.mj_model,
            num_envs=env.num_envs if hasattr(env, "num_envs") else 1,
        )
        self._scene.env_idx = self._env_idx
        self._scene.debug_visualization_enabled = self._show_debug

        with self._server.gui.add_folder("📊 状态"):
            self._status_html = self._server.gui.add_html("等待数据...")

        logger.info("✅ Viser 3D 场景初始化完成")

    def _sync_to_viser(self, qpos: np.ndarray, qvel: np.ndarray | None) -> None:
        """将最新状态同步到 Viser 3D 场景."""
        if self._scene is None:
            return

        scene = self._scene
        if hasattr(scene, "mj_data") and scene.mj_data is not None:
            import mujoco
            nq = min(len(qpos), scene.mj_data.qpos.shape[0])
            scene.mj_data.qpos[:nq] = qpos[:nq]
            if qvel is not None and scene.mj_data.qvel.shape[0] >= nq:
                scene.mj_data.qvel[:nq] = qvel[:nq]

            try:
                # 先 mj_forward 更新 xpos/xmat, 再同步到 viser 场景
                mujoco.mj_forward(scene.mj_model, scene.mj_data)
                scene.update_from_mjdata(scene.mj_data)
            except Exception:
                self._fallback_update(scene, qpos)

        if hasattr(self, "_status_html") and self._status_html is not None:
            try:
                self._status_html.content = (
                    f"<b>渲染帧</b>: {self._render_count}<br>"
                    f"<b>qpos range</b>: [{qpos.min():.3f}, {qpos.max():.3f}]<br>"
                    f"<b>base z</b>: {qpos[2]:.3f} m"
                )
            except Exception:
                pass

    def _fallback_update(self, scene: Any, qpos: np.ndarray) -> None:
        """回退更新: 当 ViserMujocoScene.update() 不可用时."""
        try:
            if hasattr(scene, "mj_model") and hasattr(scene, "mj_data"):
                import mujoco
                mujoco.mj_forward(scene.mj_model, scene.mj_data)
        except Exception:
            pass

    # ── 上下文管理器 ──

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
