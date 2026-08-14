"""
open_loop_eval.py — 本地闭环验证（模拟 Isaac-GR00T open_loop_eval.py）。

用 LeRobotEpisodeLoader 加载转换后的数据集，用 ObservationBuilder 构建符合
check_observation 契约的观测，通过 GR00TClient 发给云端 Policy Server 获取预测动作，
对比 GT 动作计算 MSE/MAE，验证数据链路与观测格式是否正确。

用法:
    # 1. 云端启动 Policy Server（微调后的 checkpoint）
    bash scripts/01_start_server.sh /path/to/checkpoint NEW_EMBODIMENT 5555

    # 2. 本地建立 SSH 隧道
    bash scripts/02_local_tunnel.sh

    # 3. 本地运行闭环验证
    python -m src.open_loop_eval \
        --dataset output/g1_lerobot \
        --robot g1 \
        --host 127.0.0.1 --port 5555 \
        --traj-ids 0 1 \
        --action-horizon 16

依赖: pyzmq msgpack msgpack-numpy numpy opencv-python pandas
（无需 torch / Isaac-GR00T，纯本地 + 云端 Server）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.policy_client import GR00TClient
from src.observation_builder import ObservationBuilder, state_slices_from_config
from src.lerobot_loader import LeRobotEpisodeLoader


def evaluate_episode(
    client: GR00TClient,
    obs_builder: ObservationBuilder,
    dataset: LeRobotEpisodeLoader,
    traj_id: int,
    action_horizon: int,
    action_key: str = "joint_pos",
) -> dict:
    """对单个 episode 做开环评估：每 action_horizon 步推理一次，消费 chunk 中多步，对比 GT。

    对齐官方 gr00t/eval/open_loop_eval.py 的 evaluate_single_trajectory：
      - 每 execution_horizon 步推理一次（而非每步推理）
      - 取预测 chunk 的前 action_horizon 步与 GT 逐步比较
      - 开环语义：在第 t 步用 GT 观测预测，不将预测反馈到状态
    """
    episode = dataset[traj_id]
    n = len(episode)
    if n == 0:
        return {"traj_id": traj_id, "steps": 0, "mse": float("nan"), "mae": float("nan")}

    # 加载该 episode 的任务描述
    task_desc = episode.get_task_description(0)

    preds, gts = [], []
    # 每 action_horizon 步推理一次，消费 chunk 中 action_horizon 步
    for t in range(0, n, action_horizon):
        frame = episode.get_frame(t)
        images = frame["images"]
        state = frame["state"]

        # 构建符合 check_observation 契约的观测
        obs = obs_builder.build(images=images, state=state, language=task_desc)

        try:
            result = client.get_action(obs)
        except Exception as e:
            print(f"    ⚠️  Step {t} 推理失败: {e}")
            continue

        # 从 action dict 提取预测动作 chunk
        action_data = result[0] if isinstance(result, tuple) else result
        if isinstance(action_data, dict):
            pred_chunk = action_data.get(action_key, list(action_data.values())[0])
        else:
            pred_chunk = action_data
        pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
        # pred_chunk 形状 (B, T, D) 或 (T, D) 或 (D,)，去掉 batch 维
        while pred_chunk.ndim > 2:
            pred_chunk = pred_chunk[0]

        # 消费 chunk 中的前 action_horizon 步（或剩余步）
        steps_to_take = min(action_horizon, n - t)
        for j in range(steps_to_take):
            if pred_chunk.ndim == 2 and j < pred_chunk.shape[0]:
                pred = pred_chunk[j]
            elif pred_chunk.ndim == 1:
                pred = pred_chunk  # 单步
            else:
                pred = pred_chunk[0] if pred_chunk.ndim >= 1 else pred_chunk

            gt_frame = episode.get_frame(t + j)
            gt = gt_frame["gt_action"]
            preds.append(pred)
            gts.append(gt)

    if not preds:
        return {"traj_id": traj_id, "steps": 0, "mse": float("nan"), "mae": float("nan")}

    preds = np.stack(preds)
    gts = np.stack(gts)
    # 对齐维度（action_dim 可能不一致时截断）
    d = min(preds.shape[-1], gts.shape[-1])
    preds = preds[..., :d]
    gts = gts[..., :d]

    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))

    return {
        "traj_id": traj_id,
        "steps": n,
        "mse": mse,
        "mae": mae,
        "pred_shape": list(preds.shape),
        "gt_shape": list(gts.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="本地开环评估（验证数据链路与观测格式）")
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot v2 数据集路径")
    parser.add_argument("--robot", type=str, default="g1",
                        choices=["g1", "h1", "h1_with_hand", "h1_2", "h2", "go2"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0],
                        help="评估的 episode 索引列表")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-key", type=str, default="joint_pos",
                        help="从 action dict 提取的 key（与 modality_config.action.modality_keys 一致）")
    parser.add_argument("--language", type=str, default=None,
                        help="覆盖任务描述（默认从 tasks.jsonl 读取）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 数据集不存在: {dataset_path}")
        sys.exit(1)

    print(f"📊 加载数据集: {dataset_path}")
    dataset = LeRobotEpisodeLoader(dataset_path=str(dataset_path))
    print(f"   Episodes: {len(dataset)}")

    # 连接 Policy Server
    print(f"\n🔌 连接 Policy Server ({args.host}:{args.port})...")
    client = GR00TClient(host=args.host, port=args.port)
    if not client.ping():
        print("❌ 无法连接 Policy Server，请确认云端已启动且 SSH 隧道已建立")
        sys.exit(1)
    print("   ✅ 连接成功")

    # 从服务端获取 modality config，构建 ObservationBuilder
    modality_config = client.get_modality_config()
    video_keys = ["front", "wrist"]
    state_keys = []
    language_key = "annotation.human.task_description"
    num_obs_steps = 1
    if isinstance(modality_config, dict):
        video_cfg = modality_config.get("video", {})
        state_cfg = modality_config.get("state", {})
        lang_cfg = modality_config.get("language", {})
        if isinstance(video_cfg, dict):
            mk = video_cfg.get("modality_keys")
            if isinstance(mk, list) and mk:
                video_keys = mk
            di = video_cfg.get("delta_indices")
            if isinstance(di, list) and di:
                num_obs_steps = len(di)
        if isinstance(state_cfg, dict):
            mk = state_cfg.get("modality_keys")
            if isinstance(mk, list) and mk:
                state_keys = mk
        if isinstance(lang_cfg, dict):
            mk = lang_cfg.get("modality_keys")
            if isinstance(mk, list) and mk:
                language_key = mk[0]

    # state_slices：优先用机器人配置，回退到按服务端 state_keys 重排
    state_slices = state_slices_from_config(args.robot)
    if state_keys and list(state_slices.keys()) != state_keys:
        from src.viz.viser_infer import _reorder_slices
        state_slices = _reorder_slices(state_keys, state_slices)

    state_dim = sum(e - s for s, e in state_slices.values())
    obs_builder = ObservationBuilder(
        camera_keys=video_keys,
        state_dim=state_dim,
        state_slices=state_slices,
        language_key=language_key,
        num_obs_steps=num_obs_steps,
        language_instruction=args.language or "perform the task",
    )

    print(f"\n🧪 开环评估 (action_horizon={args.action_horizon}, action_key='{args.action_key}')")
    print(f"   video_keys={video_keys}, state_keys={list(state_slices.keys())}, num_obs_steps={num_obs_steps}")
    print(f"   traj_ids={args.traj_ids}")
    print("")

    results = []
    for tid in args.traj_ids:
        if tid >= len(dataset):
            print(f"  ⚠️  traj {tid} 超出范围 [0, {len(dataset)})，跳过")
            continue
        print(f"  ▶️  Trajectory {tid}...")
        r = evaluate_episode(
            client, obs_builder, dataset, tid,
            action_horizon=args.action_horizon,
            action_key=args.action_key,
        )
        results.append(r)
        print(f"     steps={r['steps']}, MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")

    # 汇总
    valid = [r for r in results if r["steps"] > 0 and not np.isnan(r["mse"])]
    if valid:
        avg_mse = float(np.mean([r["mse"] for r in valid]))
        avg_mae = float(np.mean([r["mae"] for r in valid]))
        print(f"\n📈 平均指标 (over {len(valid)} trajs):")
        print(f"   Average MSE = {avg_mse:.4f}")
        print(f"   Average MAE = {avg_mae:.4f}")
    else:
        print("\n⚠️  无有效评估结果")

    # 保存结果
    out_path = dataset_path.parent / f"{args.robot}_open_loop_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "robot": args.robot,
            "dataset": str(dataset_path),
            "action_horizon": args.action_horizon,
            "action_key": args.action_key,
            "video_keys": video_keys,
            "state_keys": list(state_slices.keys()),
            "num_obs_steps": num_obs_steps,
            "results": results,
            "avg_mse": avg_mse if valid else None,
            "avg_mae": avg_mae if valid else None,
        }, f, indent=2)
    print(f"\n💾 结果已保存: {out_path}")

    client.close()


if __name__ == "__main__":
    main()
