"""mjlab_env.py — unitree_rl_mjlab 共享工具.

提供 collect_data.py 与 infer.py 共用的:
  - get_per_key_obs(env): 从 mjlab ManagerBasedRlEnv 拿 per-key obs
  - render_frame(env, ...): 从 mjlab env 渲染一帧 RGB 图像
  - load_ppo_policy(env, task_id, checkpoint, device): 加载 PPO 推理策略
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_per_key_obs(env_raw: Any) -> dict[str, Any]:
    """从 unitree_rl_mjlab ManagerBasedRlEnv 取 per-key 观测 (dict[str, tensor]).

    mjlab env.step() 返回的 obs 是按 actor/critic 拼接的 tensor,
    这里通过 observation_manager.compute() + Entity.data 拿分组前的 dict。
    """
    out: dict[str, Any] = {}
    # 1) 尝试从 observation_manager.terms 拿 per-key tensor
    try:
        om = env_raw.unwrapped.observation_manager
        for getter_name in ("terms", "_terms"):
            terms = getattr(om, getter_name, None)
            if not terms:
                continue
            for group_name, group_terms in terms.items():
                if not isinstance(group_terms, dict):
                    continue
                for term_name, term in group_terms.items():
                    data = getattr(term, "data", None)
                    if data is None:
                        continue
                    if hasattr(data, "squeeze"):
                        out[term_name] = data
            break
    except Exception as e:
        logger.debug("observation_manager.terms 失败: %s", e)

    # 2) 兜底: 从 robot Entity.data 拿 base / joint 状态
    try:
        robot = env_raw.unwrapped.scene["robot"]
        rd = robot.data
        if "base_pos" not in out and getattr(rd, "root_link_pos_w", None) is not None:
            out["base_pos"] = rd.root_link_pos_w
        if "base_quat" not in out and getattr(rd, "root_link_quat_w", None) is not None:
            out["base_quat"] = rd.root_link_quat_w
        if "base_lin_vel" not in out and getattr(rd, "root_link_lin_vel_w", None) is not None:
            out["base_lin_vel"] = rd.root_link_lin_vel_w
        if "base_ang_vel" not in out and getattr(rd, "root_link_ang_vel_w", None) is not None:
            out["base_ang_vel"] = rd.root_link_ang_vel_w
        if "joint_pos" not in out and getattr(rd, "joint_pos", None) is not None:
            out["joint_pos"] = rd.joint_pos
        if "joint_vel" not in out and getattr(rd, "joint_vel", None) is not None:
            out["joint_vel"] = rd.joint_vel
    except Exception as e:
        logger.debug("Entity.data 兜底失败: %s", e)

    return out


def render_frame(env_raw: Any, height: int = 224, width: int = 224) -> np.ndarray | None:
    """从 mjlab env 渲染一帧 RGB 图像 (H, W, 3) uint8.

    多种 API 兼容性尝试:
      1) env.render()                — gym-style
      2) env.unwrapped.sim._data     — 直接走 mujoco native offscreen
      3) viser / native viewer (需 DISPLAY)
    """
    if env_raw is None:
        return None
    # 方案 1: gym-style render
    try:
        frame = env_raw.render()
        if frame is not None:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 \
                    else arr.clip(0, 255).astype(np.uint8)
            # 调整尺寸到目标 H x W
            if arr.shape[0] != height or arr.shape[1] != width:
                try:
                    import cv2  # type: ignore
                    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
                except ImportError:
                    try:
                        from PIL import Image
                        img = Image.fromarray(arr)
                        arr = np.asarray(img.resize((width, height)))
                    except ImportError:
                        pass  # 尺寸不对也返回, 至少给个错位警告
            return arr
    except Exception:
        pass
    # 方案 2: mjlab sim 内部 mujoco native
    try:
        sim = env_raw.unwrapped.sim
        model = getattr(sim, "_model", None) or getattr(sim, "model", None)
        data = getattr(sim, "_data", None) or getattr(sim, "data", None)
        if model is not None and data is not None:
            import mujoco  # noqa
            from mujoco import Renderer
            viewer_cfg = env_raw.unwrapped.cfg.viewer
            camera_name = getattr(viewer_cfg, "camera_name", -1)
            renderer = Renderer(height=height, width=width)
            renderer.update_scene(data, camera=camera_name)
            frame = renderer.render()
            return np.asarray(frame).astype(np.uint8)
    except Exception as e:
        logger.debug("mjlab sim.render 失败: %s", e)
    return None


def load_ppo_policy(env_raw: Any, task_id: str, checkpoint_path: str,
                    device: str = "cuda:0"):
    """加载 unitree_rl_mjlab 训练好的 PPO 策略, 返回 inference callable.

    Returns:
        policy_fn: (obs_tensor) → action_tensor
    """
    rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
    if rl_mjlab_root.exists() and str(rl_mjlab_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(rl_mjlab_root))

    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    agent_cfg = load_rl_cfg(task_id)
    env_wrapped = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env_wrapped, agent_cfg, device=device)
    runner.load(checkpoint_path, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def to_numpy(t: Any) -> np.ndarray | None:
    """torch.Tensor / numpy.ndarray → numpy (squeeze batch dim)."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    arr = np.asarray(t)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr
