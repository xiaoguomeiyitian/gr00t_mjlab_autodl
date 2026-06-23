#!/usr/bin/env python3
"""AsyncViser3DViewer — 异步 Viser 3D 可视化, 不阻塞主线程 env.step().

后台线程以 ~30 fps 渲染, 主线程调用 update() 同步 qpos/qvel.
复用 mjlab 的 ViserMujocoScene 渲染 3D 机器人场景.
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
        port: int = 8080,
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

        self._lock = threading.Lock()
        self._qpos: np.ndarray | None = None
        self._qvel: np.ndarray | None = None
        self._new_data = threading.Event()

        self._render_count = 0
        self._last_render_time = 0.0

    # ── 公开 API ──

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
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Viser 3D viewer 已停止 (共渲染 %d 帧)", self._render_count)

    def update(self) -> None:
        """同步最新 env 状态到渲染线程 (非阻塞)."""
        try:
            env = self._env
            if env is None:
                return

            sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
            mj_data = sim.mj_data

            qpos = np.asarray(mj_data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(mj_data.qvel, dtype=np.float64).copy()

            with self._lock:
                self._qpos = qpos
                self._qvel = qvel
            self._new_data.set()
        except Exception as e:
            logger.debug("Viser update 失败: %s", e)

    @property
    def url(self) -> str:
        return f"http://0.0.0.0:{self._port}"

    @property
    def render_count(self) -> int:
        return self._render_count

    # ── 内部实现 ──

    def _render_loop(self) -> None:
        """后台渲染线程主循环."""
        try:
            self._setup_viser()
        except Exception as e:
            logger.error("Viser 初始化失败: %s", e)
            logger.info("提示: pip install viser")
            return

        frame_interval = 1.0 / self._frame_rate
        while not self._stop.is_set():
            try:
                self._new_data.wait(timeout=frame_interval)
                self._new_data.clear()

                with self._lock:
                    qpos = self._qpos
                    qvel = self._qvel

                if qpos is not None:
                    self._sync_to_viser(qpos, qvel)
                    self._render_count += 1

                elapsed = time.time() - self._last_render_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._last_render_time = time.time()

            except Exception as e:
                if not self._stop.is_set():
                    logger.debug("Viser 渲染帧失败: %s", e)
                time.sleep(0.1)

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
        )

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
            nq = min(len(qpos), scene.mj_data.qpos.shape[0])
            scene.mj_data.qpos[:nq] = qpos[:nq]
            if qvel is not None and scene.mj_data.qvel.shape[0] >= nq:
                scene.mj_data.qvel[:nq] = qvel[:nq]

            try:
                scene.update()
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
