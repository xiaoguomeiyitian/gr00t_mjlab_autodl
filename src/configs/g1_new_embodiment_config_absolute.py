"""G1 modality config (absolute action) — registers for `NEW_EMBODIMENT`.

Same as `g1_new_embodiment_config.py` but with `ABSOLUTE` action representation.
Use this when collecting data with `--action-mode absolute` (joint targets as
absolute angles, not deltas from current position).

For locomotion, `RELATIVE` (delta mode) is usually preferred because it
generalizes better and is less sensitive to bias drift in the joint position
estimator. `ABSOLUTE` is provided for completeness / ablation.

The action key name in the parquet is still `joint_position_delta` (the schema
field name is fixed; only the *semantic meaning* — absolute target vs delta —
changes via the `ActionConfig.rep` field).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


NUM_JOINTS = 29
ACTION_HORIZON = 16


g1_absolute_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_pos",
            "joint_vel",
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
        ],
    ),
    # ── 关键区别: rep=ABSOLUTE, 模型学的是关节目标绝对位置 ──────
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "joint_position_delta",  # 在 absolute 模式下实际存的是 joint_position_target
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="joint_pos",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_absolute_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
