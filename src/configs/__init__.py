"""gr00t_mjlab_autodl 机器人 + 模态配置子包.

本包包含:

- ``g1_config.py`` / ``go2_config.py``  — 关节名、默认姿态、HOME_KEYFRAME、
  LeRobot v2 ``modality.json`` schema 生成函数 (state/action start-end 切片)。

- ``g1_new_embodiment_config.py``  — G1 三角洲 ``register_modality_config`` for
  ``NEW_EMBODIMENT``  (RELATIVE + NON_EEF action)。

- ``g1_new_embodiment_config_absolute.py`` — G1 绝对量 variant (ABSOLUTE)。

- ``go2_new_embodiment_config.py`` — Go2 三角洲 variant。

- ``go2_new_embodiment_config_absolute.py`` — Go2 绝对量 variant。

Isaac-GR00T 的 ``launch_finetune.py`` 和 ``gr00t/data/stats.py`` 都通过
``--modality-config-path <file.py>`` 参数以 importlib 副作用导入上述文件。
训练/推理时只需选一个与采集 ``--action-mode`` 一致的文件。

例 (云端 03 脚本):

    python3 gr00t/experiment/launch_finetune.py \\
        --modality-config-path /path/to/gr00t_mjlab_autodl/src/configs/g1_new_embodiment_config.py \\
        --embodiment-tag NEW_EMBODIMENT \\
        ...
"""

