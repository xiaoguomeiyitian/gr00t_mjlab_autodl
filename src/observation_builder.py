"""
Observation Builder — 从机器人环境数据构建 GR00T 观测字典。

支持:
- 多相机图像（自动 resize 到 224×224）
- 关节状态拼接
- 语言指令注入
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
        """
        Args:
            camera_keys: 相机 key 列表，如 ["exterior_image_1_left", "wrist_image_left"]
            state_dim: 状态维度（G1=71, Go2=37）
            image_size: 图像尺寸 (H, W)
            language_instruction: 默认语言指令
        """
        self.camera_keys = camera_keys or ["exterior_image_1_left"]
        self.state_dim = state_dim
        self.image_size = image_size
        self.language_instruction = language_instruction

    def build(
        self,
        images: dict,
        state: np.ndarray,
        language: Optional[str] = None,
    ) -> dict:
        """
        构建观测字典。

        Args:
            images: {"camera_key": np.ndarray (H,W,3) uint8}
            state: (state_dim,) float32 关节状态
            language: 语言指令（可选，覆盖默认值）

        Returns:
            GR00T 观测字典
        """
        # 处理图像
        video = {}
        for key in self.camera_keys:
            if key in images:
                img = images[key]
                if img.shape[:2] != self.image_size:
                    img = self._resize_image(img, self.image_size)
                video[key] = img

        # 构建观测
        obs = {
            "video": video,
            "state": state.astype(np.float32)[None, ...],  # (1, state_dim)
            "language": language or self.language_instruction,
        }
        return obs

    @staticmethod
    def _resize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """OpenCV resize"""
        import cv2
        return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
