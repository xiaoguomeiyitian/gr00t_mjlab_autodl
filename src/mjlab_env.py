"""unitree_rl_mjlab 共享工具: get_per_key_obs / render_frame / load_ppo_policy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_per_key_obs(env_raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        robot = env_raw.unwrapped.scene["robot"]
        rd = robot.data
        out["base_pos"] = rd.root_link_pos_w
        out["base_quat"] = rd.root_link_quat_w
        out["base_lin_vel"] = rd.root_link_lin_vel_w
        out["base_ang_vel"] = rd.root_link_ang_vel_w
        out["joint_pos"] = rd.joint_pos
        out["joint_vel"] = rd.joint_vel
    except Exception as e:
        logger.debug("Entity.data 失败: %s", e)

    return out


def render_frame(env_raw: Any, height: int = 224, width: int = 224) -> np.ndarray | None:
    if env_raw is None:
        return None
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
    try:
        sim = env_raw.unwrapped.sim
        mj_model = sim.mj_model
        mj_data = sim.mj_data
        import mujoco  # noqa
        from mujoco import Renderer
        viewer_cfg = env_raw.unwrapped.cfg.viewer
        camera_name = getattr(viewer_cfg, "camera_name", -1)
        renderer = Renderer(model=mj_model, height=height, width=width)
        renderer.update_scene(mj_data, camera=camera_name)
        frame = renderer.render()
        return np.asarray(frame).astype(np.uint8)
    except Exception as e:
        logger.debug("mjlab sim.render 失败: %s", e)
    return None


def load_ppo_policy(env_raw: Any, task_id: str, checkpoint_path: str,
                    device: str = "cuda:0"):
    rl_mjlab_root = Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"
    if rl_mjlab_root.exists() and str(rl_mjlab_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(rl_mjlab_root))

    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    agent_cfg = load_rl_cfg(task_id)
    env_wrapped = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    try:
        from dataclasses import asdict
        train_cfg_dict = asdict(agent_cfg)
    except Exception:
        # 兜底: 手动 __dict__ 拷贝
        train_cfg_dict = {k: v for k, v in vars(agent_cfg).items()
                          if not k.startswith("_")}
    runner = runner_cls(env_wrapped, train_cfg_dict, log_dir=None, device=device)
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
