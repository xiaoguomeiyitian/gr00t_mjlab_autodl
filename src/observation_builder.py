"""
Observation Builder — 从机器人环境数据构建 GR00T 观测字典。

产出格式严格符合 Isaac-GR00T `Gr00tPolicy.check_observation()` 契约：

    {
        "video":    {<cam_key>:    np.ndarray (B, T, H, W, 3) uint8},
        "state":    {<state_key>:  np.ndarray (B, T, D)       float32},
        "language": {<language_key>: [[str]]  (B, T)},
    }

- state 必须按 state_slices 拆分成多个子键（与 modality_config.state.modality_keys 对齐）
- language 必须是 dict，key 与 modality_config.language.modality_keys 一致
- video/state 的 T 维必须等于 num_obs_steps（默认 1）
"""

from typing import Optional

import numpy as np


class ObservationBuilder:
    """将环境数据转换为 GR00T 观测格式（符合 check_observation 契约）。

    产出格式严格符合 Isaac-GR00T `Gr00tPolicy.check_observation()`：

        {
            "video":   {<cam_key>: np.ndarray (B, T, H, W, 3) uint8},
            "state":   {<state_key>: np.ndarray (B, T, D) float32},
            "language": {<language_key>: [[str]]  (B, T)},
        }

    - state 必须按 state_slices 拆分成多个子键（与 modality_config.state.modality_keys 对齐）
    - language 必须是 dict，key 与 modality_config.language.modality_keys 一致
    - video/state 的 T 维必须等于 num_obs_steps（默认 1）
    """

    def __init__(
        self,
        camera_keys: Optional[list] = None,
        state_dim: int = 71,
        image_size: tuple = (224, 224),
        language_instruction: str = "perform the task",
        # 与 modality_config 对齐的拆分配置
        state_slices: Optional[dict] = None,
        language_key: str = "annotation.human.task_description",
        num_obs_steps: int = 1,
    ):
        # 默认相机键与 modality_config 的 video.modality_keys 命名一致
        self.camera_keys = camera_keys or ["front", "wrist"]
        self.state_dim = state_dim
        self.image_size = image_size
        self.language_instruction = language_instruction

        # state_slices: {state_key: (start, end)}，与 modality_config.state.modality_keys 对齐
        # 若未提供，退化为单个 "state" 键覆盖整个向量（不推荐，会与 GR00T 校验冲突）
        self.state_slices = state_slices or {"state": (0, state_dim)}
        self.language_key = language_key
        self.num_obs_steps = num_obs_steps

    def build(
        self,
        images: dict,
        state: np.ndarray,
        language: Optional[str] = None,
    ) -> dict:
        """构建符合 GR00T check_observation 契约的观测字典。

        Args:
            images: {cam_key: (H,W,3) 或 (T,H,W,3) 或 (B,T,H,W,3) uint8}
            state: (D,) 或 (T,D) 或 (B,T,D) float32，整个拼接状态向量
            language: 任务指令字符串；None 时用 language_instruction
        """
        video = self._build_video(images)
        state_dict = self._build_state(state)
        language_dict = self._build_language(language)
        return {"video": video, "state": state_dict, "language": language_dict}

    def _build_video(self, images: dict) -> dict:
        """构建 video 字典：每个相机 (B, T, H, W, 3) uint8。"""
        video = {}
        for key in self.camera_keys:
            if key in images:
                img = np.asarray(images[key])
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                if img.shape[-3:-1] != self.image_size:
                    img = self._resize_image(img, self.image_size)
                img = self._ensure_video_dims(img)  # → (B, T, H, W, 3)
            else:
                import warnings
                warnings.warn(f"相机 '{key}' 缺失，用零张量填充")
                img = np.zeros(
                    (1, self.num_obs_steps, self.image_size[0], self.image_size[1], 3),
                    dtype=np.uint8,
                )
            video[key] = img
        return video

    def _build_state(self, state: np.ndarray) -> dict:
        """构建 state 字典：按 state_slices 拆分成多个子键，每个 (B, T, D) float32。"""
        state = np.asarray(state, dtype=np.float32)
        state = self._ensure_state_dims(state)  # → (B, T, D)

        total_dim = state.shape[-1]
        if total_dim != self.state_dim and self.state_dim > 0:
            import warnings
            warnings.warn(
                f"state 末维 {total_dim} != state_dim {self.state_dim}，可能维度不匹配"
            )

        state_dict = {}
        for state_key, (start, end) in self.state_slices.items():
            if end > total_dim:
                import warnings
                warnings.warn(
                    f"state_slice '{state_key}' [{start}:{end}] 超出 state 末维 {total_dim}，用零填充"
                )
                chunk = np.zeros((*state.shape[:-1], end - start), dtype=np.float32)
                avail = max(0, total_dim - start)
                if avail > 0:
                    chunk[..., :avail] = state[..., start:start + avail]
            else:
                chunk = state[..., start:end]
            state_dict[state_key] = np.ascontiguousarray(chunk, dtype=np.float32)
        return state_dict

    def _build_language(self, language: Optional[str]) -> dict:
        """构建 language 字典：{language_key: [[str]]}，形状 (B, T)。

        官方 check_observation 契约：language[key] = list[list[str]]
          - 外层 list 长度 = B（batch size，固定为 1）
          - 内层 list 长度 = T（= len(language.delta_indices)，通常 1）
          - 每个 T 元素是 1 个 str
        """
        text = language if language is not None else self.language_instruction
        # 外层 B=1，内层 T=num_obs_steps，每步 1 个 str
        return {self.language_key: [[text] * self.num_obs_steps]}

    def _ensure_video_dims(self, img: np.ndarray) -> np.ndarray:
        """将图像统一成 (B, T, H, W, 3) uint8。支持 (H,W,3)/(T,H,W,3)/(B,T,H,W,3)。"""
        if img.ndim == 3:
            img = img[None, None, ...]
        elif img.ndim == 4:
            img = img[None, ...]
        elif img.ndim == 5:
            pass
        else:
            raise ValueError(f"图像维度 {img.ndim} 不支持，期望 3/4/5 维")

        if img.shape[1] != self.num_obs_steps:
            if img.shape[1] == 1:
                img = np.broadcast_to(img, (img.shape[0], self.num_obs_steps, *img.shape[2:]))
                img = np.ascontiguousarray(img)
            else:
                T = self.num_obs_steps
                if img.shape[1] >= T:
                    img = img[:, :T]
                else:
                    pad = np.zeros((img.shape[0], T - img.shape[1], *img.shape[2:]), dtype=img.dtype)
                    img = np.concatenate([img, pad], axis=1)
        return img

    def _ensure_state_dims(self, state: np.ndarray) -> np.ndarray:
        """将状态统一成 (B, T, D) float32。支持 (D,)/(T,D)/(B,T,D)。"""
        if state.ndim == 1:
            state = state[None, None, :]
        elif state.ndim == 2:
            state = state[None, ...]
        elif state.ndim == 3:
            pass
        else:
            raise ValueError(f"state 维度 {state.ndim} 不支持，期望 1/2/3 维")

        if state.shape[1] != self.num_obs_steps:
            if state.shape[1] == 1:
                state = np.broadcast_to(state, (state.shape[0], self.num_obs_steps, state.shape[2]))
                state = np.ascontiguousarray(state)
            else:
                T = self.num_obs_steps
                if state.shape[1] >= T:
                    state = state[:, :T]
                else:
                    pad = np.zeros((state.shape[0], T - state.shape[1], state.shape[2]), dtype=np.float32)
                    state = np.concatenate([state, pad], axis=1)
        return state

    @staticmethod
    def _resize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """resize 图像到 target_size (H, W)。支持任意前导维。"""
        import cv2
        h, w = target_size
        if img.ndim == 3:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif img.ndim == 4:
            return np.stack([cv2.resize(img[t], (w, h), interpolation=cv2.INTER_LINEAR) for t in range(img.shape[0])])
        elif img.ndim == 5:
            return np.stack([
                np.stack([cv2.resize(img[b, t], (w, h), interpolation=cv2.INTER_LINEAR) for t in range(img.shape[1])])
                for b in range(img.shape[0])
            ])
        raise ValueError(f"不支持的图像维度 {img.ndim}")


def state_slices_from_config(robot: str) -> dict:
    """根据机器人类型返回 state_slices（与 modality_config.state.modality_keys 对齐）。

    与 src/configs/<robot>_config.py 的 SLICES 保持一致。
    state 拼接顺序：joint_pos + joint_vel + base_pos(3) + base_quat(4) + base_lin_vel(3) + base_ang_vel(3)
    """
    slices_map = {
        "g1":           {"joint_pos": (0, 29), "joint_vel": (29, 58), "base_pos": (58, 61),
                         "base_quat": (61, 65), "base_lin_vel": (65, 68), "base_ang_vel": (68, 71)},
        "h1":           {"joint_pos": (0, 20), "joint_vel": (20, 40), "base_pos": (40, 43),
                         "base_quat": (43, 47), "base_lin_vel": (47, 50), "base_ang_vel": (50, 53)},
        "h1_with_hand": {"joint_pos": (0, 46), "joint_vel": (46, 92), "base_pos": (92, 95),
                         "base_quat": (95, 99), "base_lin_vel": (99, 102), "base_ang_vel": (102, 105)},
        "h1_2":         {"joint_pos": (0, 52), "joint_vel": (52, 104), "base_pos": (104, 107),
                         "base_quat": (107, 111), "base_lin_vel": (111, 114), "base_ang_vel": (114, 117)},
        "h2":           {"joint_pos": (0, 32), "joint_vel": (32, 64), "base_pos": (64, 67),
                         "base_quat": (67, 71), "base_lin_vel": (71, 74), "base_ang_vel": (74, 77)},
        "go2":          {"joint_pos": (0, 12), "joint_vel": (12, 24), "base_pos": (24, 27),
                         "base_quat": (27, 31), "base_lin_vel": (31, 34), "base_ang_vel": (34, 37)},
    }
    return slices_map.get(robot, slices_map["g1"])
