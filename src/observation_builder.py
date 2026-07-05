"""
Observation Builder — 从机器人环境数据构建 GR00T 观测字典。
"""

from typing import Optional

import numpy as np


class ObservationBuilder:
    """将环境数据转换为 GR00T 观测格式。"""

    def __init__(
        self,
        camera_keys: Optional[list] = None,
        state_dim: int = 71,
        image_size: tuple = (224, 224),
        language_instruction: str = "perform the task",
    ):
        # 默认相机键与 modality_config 的 video.modality_keys 命名一致
        self.camera_keys = camera_keys or ["front", "wrist"]
        self.state_dim = state_dim
        self.image_size = image_size
        self.language_instruction = language_instruction

    def build(
        self,
        images: dict,
        state: np.ndarray,
        language: Optional[str] = None,
    ) -> dict:
        video = {}
        for key in self.camera_keys:
            if key in images:
                img = images[key]
                if img.shape[:2] != self.image_size:
                    img = self._resize_image(img, self.image_size)
                video[key] = img
            else:
                # 缺失相机时用零张量填充并警告，避免下游维度不匹配
                import warnings
                warnings.warn(f"相机 '{key}' 缺失，用零张量填充")
                video[key] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] != self.state_dim and self.state_dim > 0:
            import warnings
            warnings.warn(
                f"state 末维 {state.shape[-1]} != state_dim {self.state_dim}，可能维度不匹配"
            )

        return {
            "video": video,
            "state": state[None, ...],
            "language": language or self.language_instruction,
        }

    @staticmethod
    def _resize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        import cv2
        return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
