"""collect_data.py — MJLab 仿真数据采集。"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


ROBOT_CONFIGS = {
    "g1": {
        "task": "Mjlab-Velocity-Flat-Unitree-G1",
        "num_joints": 29,
        "state_dim": 29 + 29 + 3 + 4 + 3 + 3,
        "action_dim": 29,
        "camera_names": ["front", "wrist"],
    },
    "h1": {
        "task": "Mjlab-Velocity-Flat-Unitree-H1",
        "num_joints": 20,
        "state_dim": 20 + 20 + 3 + 4 + 3 + 3,
        "action_dim": 20,
        "camera_names": ["front", "wrist"],
    },
    "h1_with_hand": {
        "task": "Mjlab-Velocity-Flat-Unitree-H1",
        "num_joints": 46,
        "state_dim": 46 + 46 + 3 + 4 + 3 + 3,
        "action_dim": 46,
        "camera_names": ["front", "wrist"],
    },
    "h1_2": {
        "task": "Mjlab-Velocity-Flat-Unitree-H1-2",
        "num_joints": 52,
        "state_dim": 52 + 52 + 3 + 4 + 3 + 3,
        "action_dim": 52,
        "camera_names": ["front", "wrist"],
    },
    "h2": {
        "task": "Mjlab-Velocity-Flat-Unitree-H2",
        "num_joints": 32,
        "state_dim": 32 + 32 + 3 + 4 + 3 + 3,
        "action_dim": 32,
        "camera_names": ["front", "wrist"],
    },
    "go2": {
        "task": "Mjlab-Velocity-Flat-Unitree-Go2",
        "num_joints": 12,
        "state_dim": 12 + 12 + 3 + 4 + 3 + 3,
        "action_dim": 12,
        "camera_names": ["front", "back"],
    },
}


class DataCollector:
    """从 MJLab 仿真环境采集演示数据。"""

    def __init__(
        self,
        robot: str = "g1",
        task: Optional[str] = None,
        action_mode: str = "absolute",  # 存储绝对关节角，absolute→relative 由 GR00T processor 处理
        num_episodes: int = 50,
        episode_length: int = 300,
        fps: int = 30,
        image_size: tuple = (224, 224),
        seed: int = 42,
    ):
        if robot not in ROBOT_CONFIGS:
            raise ValueError(f"不支持的机器人: {robot}，可选: {list(ROBOT_CONFIGS.keys())}")

        self.robot = robot
        self.config = ROBOT_CONFIGS[robot]
        self.task = task or self.config["task"]
        self.action_mode = action_mode
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.fps = fps
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        # absolute 模式的 episode 级相位偏移（在 _collect_episode 中重置）
        self._abs_phase_offset = self.rng.randn(self.config["num_joints"]) * 0.01

    def run(self, output_dir: str = "g1_raw"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"🤖 开始数据采集")
        print(f"   机器人: {self.robot}")
        print(f"   任务: {self.task}")
        print(f"   动作模式: {self.action_mode}")
        print(f"   Episodes: {self.num_episodes}")
        print(f"   每 episode 步数: {self.episode_length}")
        print(f"   输出: {output_path}")
        print()

        env = self._create_env()

        stats = {"total_steps": 0, "start_time": time.time(), "episodes": []}

        for ep_idx in range(self.num_episodes):
            ep_data = self._collect_episode(env, ep_idx, output_path)
            stats["episodes"].append(ep_data)
            stats["total_steps"] += ep_data["steps"]

            elapsed = time.time() - stats["start_time"]
            avg_time = elapsed / (ep_idx + 1)
            eta = avg_time * (self.num_episodes - ep_idx - 1)

            print(f"  ✅ Episode {ep_idx + 1}/{self.num_episodes}  steps={ep_data['steps']}  reward={ep_data['reward']:.2f}  ETA={eta:.0f}s")

        elapsed = time.time() - stats["start_time"]
        print(f"\n📊 采集完成")
        print(f"   总步数: {stats['total_steps']}")
        print(f"   总耗时: {elapsed:.1f}s")
        print(f"   输出目录: {output_path}")

        self._save_metadata(output_path, stats)
        return stats

    def _create_env(self):
        try:
            from src.mjlab_env import MjLabEnv

            env = MjLabEnv(
                task_name=self.task,
                camera_names=self.config["camera_names"],
                image_size=self.image_size,
            )
            print(f"✅ MJLab 环境创建成功: {self.task}")
            return env
        except ImportError as e:
            print(f"⚠️  无法创建 MJLab 环境: {e}")
            print(f"   将使用模拟环境（生成随机数据用于测试）")
            return _MockEnv(self.config, self.rng,
                            episode_length=self.episode_length,
                            image_size=self.image_size)

    def _collect_episode(self, env, ep_idx: int, output_path: Path) -> dict:
        obs = env.reset()
        # 每相机独立帧列表
        frames_by_cam = {cam: [] for cam in self.config["camera_names"]}
        states = []
        actions = []
        rewards = []
        total_reward = 0.0

        # absolute 模式：每 episode 采样一次固定相位偏移，避免每步重采样破坏波形
        self._abs_phase_offset = self.rng.randn(self.config["num_joints"]) * 0.01

        for step in range(self.episode_length):
            state = self._extract_state(obs)
            images = self._extract_image(obs)

            action = self._generate_action(state, step)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            for cam in self.config["camera_names"]:
                frames_by_cam[cam].append(images[cam])
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

            if done:
                break

        steps = len(states)

        npz_path = output_path / f"episode_{ep_idx:04d}.npz"
        np.savez_compressed(
            str(npz_path),
            states=np.stack(states),
            actions=np.stack(actions),
            rewards=np.array(rewards),
            task_name=self.task,
            robot=self.robot,
            action_mode=self.action_mode,
        )

        # 每相机分别保存视频
        mp4_paths = {}
        for cam in self.config["camera_names"]:
            if frames_by_cam[cam]:
                mp4_path = output_path / f"episode_{ep_idx:04d}_{cam}.mp4"
                self._save_video(frames_by_cam[cam], str(mp4_path))
                mp4_paths[cam] = str(mp4_path)

        return {"episode": ep_idx, "steps": steps, "reward": total_reward,
                "npz": str(npz_path), "mp4": mp4_paths}

    def _extract_state(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            qpos = obs.get("qpos", np.zeros(self.config["state_dim"]))
            qvel = obs.get("qvel", np.zeros(self.config["action_dim"]))
            base_pos = obs.get("base_pos", np.zeros(3))
            base_quat = obs.get("base_quat", np.array([1, 0, 0, 0]))
            base_lin_vel = obs.get("base_lin_vel", np.zeros(3))
            base_ang_vel = obs.get("base_ang_vel", np.zeros(3))

            return np.concatenate([
                qpos[:self.config["num_joints"]],
                qvel[:self.config["num_joints"]],
                base_pos, base_quat, base_lin_vel, base_ang_vel,
            ])
        else:
            return np.zeros(self.config["state_dim"])

    def _extract_image(self, obs) -> dict:
        images = {}
        if isinstance(obs, dict):
            for cam_name in self.config["camera_names"]:
                img = obs.get(f"image_{cam_name}", obs.get("images", {}).get(cam_name))
                if img is not None:
                    if img.shape[:2] != self.image_size:
                        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                    images[cam_name] = img
                else:
                    images[cam_name] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        return images

    def _generate_action(self, state: np.ndarray, step: int) -> np.ndarray:
        num_joints = self.config["num_joints"]
        joint_pos = state[:num_joints]

        if self.action_mode == "absolute":
            t = step / self.fps
            freq = 2.0 * np.pi * 0.5
            amplitude = 0.3
            # 用 episode 级固定相位偏移，避免每步重采样破坏正弦波形
            action = joint_pos + amplitude * np.sin(freq * t + self._abs_phase_offset)
        elif self.action_mode == "delta":
            t = step / self.fps
            freq = 2.0 * np.pi * 0.5
            amplitude = 0.02
            action = amplitude * np.sin(freq * t + np.arange(num_joints) * 0.2)
        elif self.action_mode == "relative_eef":
            action = self.rng.randn(num_joints).astype(np.float32) * 0.01
        else:
            raise ValueError(f"未知动作模式: {self.action_mode}")

        return action.astype(np.float32)

    def _save_video(self, frames: list, path: str, fps: Optional[int] = None):
        if not frames:
            return

        fps = fps or self.fps
        # frames 现为单相机的 ndarray 帧列表
        images = [f for f in frames if isinstance(f, np.ndarray)]
        if not images:
            return

        h, w = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)

        for img in images:
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            writer.write(bgr)

        writer.release()

    def _save_metadata(self, output_path: Path, stats: dict):
        import json

        meta = {
            "robot": self.robot,
            "task": self.task,
            "action_mode": self.action_mode,
            "num_episodes": self.num_episodes,
            "episode_length": self.episode_length,
            "fps": self.fps,
            "image_size": list(self.image_size),
            "state_dim": self.config["state_dim"],
            "action_dim": self.config["action_dim"],
            "camera_names": self.config["camera_names"],
            "episodes": stats["episodes"],
            "total_steps": stats["total_steps"],
        }

        meta_path = output_path / "collection_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


class _MockEnv:
    """模拟环境（无 MJLab 时用于测试数据采集流程）。

    注意：此为测试桩，保留在 src/ 仅为兼容旧测试。新代码应使用 tests/ 下的 fixture。
    """

    def __init__(self, config: dict, rng: np.random.RandomState,
                 episode_length: int = 300, image_size: tuple = (224, 224)):
        self.config = config
        self.rng = rng
        self.step_count = 0
        self.episode_length = episode_length
        self.image_size = image_size

    def reset(self) -> dict:
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple:
        self.step_count += 1
        reward = float(self.rng.randn() * 0.1)
        done = self.step_count >= self.episode_length
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> dict:
        num_joints = self.config["num_joints"]
        h, w = self.image_size
        return {
            "qpos": self.rng.randn(num_joints).astype(np.float32) * 0.5,
            "qvel": self.rng.randn(num_joints).astype(np.float32) * 0.1,
            "base_pos": np.array([0, 0, 0.8], dtype=np.float32),
            "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "base_lin_vel": self.rng.randn(3).astype(np.float32) * 0.05,
            "base_ang_vel": self.rng.randn(3).astype(np.float32) * 0.05,
            "images": {
                cam: self.rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
                for cam in self.config["camera_names"]
            },
        }


def main():
    parser = argparse.ArgumentParser(description="MJLab 仿真数据采集")
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"],
                        help="机器人类型")
    parser.add_argument("--task", type=str, default=None,
                        help="MJLab 任务名（默认从配置取）")
    parser.add_argument("--action-mode", type=str, default="absolute",
                        choices=["absolute", "delta", "relative_eef"],
                        help="动作模式")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="采集 episode 数量")
    parser.add_argument("--episode-length", type=int, default=300,
                        help="每 episode 步数")
    parser.add_argument("--fps", type=int, default=30,
                        help="视频帧率")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                        help="图像尺寸 H W")
    parser.add_argument("--output-dir", type=str, default="g1_raw",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    collector = DataCollector(
        robot=args.robot,
        task=args.task,
        action_mode=args.action_mode,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        fps=args.fps,
        image_size=tuple(args.image_size),
        seed=args.seed,
    )

    collector.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
