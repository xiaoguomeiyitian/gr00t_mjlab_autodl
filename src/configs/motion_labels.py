"""
motion_labels.py — 动作文件名到语言标签的映射。
"""

import re
from pathlib import Path


# 顺序敏感：更长/更具体的 keyword 放前面，避免 "walk" 吞掉 "moonwalk"
LABEL_MAP = [
    ("moonwalk", "slide feet backward"),
    ("high_knee", "march with high knees"),
    ("side_step", "side step"),
    ("backpedal", "walk backward"),
    ("walk", "walk forward"),
    ("run", "run forward"),
    ("sprint", "sprint forward"),
    ("dance", "perform dancing motion"),
    ("fight", "perform fighting motion"),
    ("jump", "jump up repeatedly"),
    ("fall", "fall and get up"),
    ("grab", "grab object from ground"),
    ("kick", "kick forward"),
    ("punch", "punch forward"),
    ("turn", "turn around"),
    ("step", "step in place"),
    ("climb", "climb stairs"),
    ("crouch", "crouch down"),
    ("push", "push object"),
    ("pull", "pull object"),
    ("throw", "throw object"),
    ("catch", "catch object"),
    ("wave", "wave hand"),
    ("point", "point forward"),
    ("balance", "balance on one foot"),
    ("tiptoe", "walk on tiptoe"),
    ("shuffle", "shuffle feet"),
    ("twist", "twist body"),
    ("bend", "bend down"),
    ("stretch", "stretch arms"),
    ("sit", "sit down"),
    ("stand", "stand up"),
    ("lie", "lie down"),
    ("crawl", "crawl forward"),
    ("roll", "roll on ground"),
    ("flip", "do a flip"),
    ("spin", "spin around"),
    ("lunge", "lunge forward"),
    ("squat", "squat down"),
    ("high_knee", "lift knees high"),
    ("side_step", "step sideways"),
    ("backpedal", "walk backward"),
    ("zigzag", "walk in zigzag pattern"),
    ("moonwalk", "slide feet backward"),
    ("box", "perform boxing motion"),
    ("yoga", "perform yoga pose"),
    ("tai_chi", "perform tai chi motion"),
    ("golf", "perform golf swing"),
    ("tennis", "perform tennis swing"),
    ("basketball", "perform basketball dribble"),
    ("swim", "perform swimming motion"),
]

def get_motion_label(filename: str, default: str = "perform the locomotion task") -> str:
    # 先取 stem（去扩展名），避免 .csv 后缀干扰正则锚点
    name_lower = Path(filename).stem.lower()
    # 去掉后缀标记（_g1/_h2/_h1/_origin/_retargeted/_from_...）
    name_lower = re.sub(r"(_from_.*|_g1|_h2|_h1|_origin|_retargeted)", "", name_lower)
    # 去掉数字编号后缀
    name_lower = re.sub(r"_\d{3}_\d{3}_\d{3}", "", name_lower)

    for keyword, label in LABEL_MAP:
        if keyword in name_lower:
            return label

    return default


if __name__ == "__main__":
    test_files = [
        "dance1_subject2.csv",
        "walk3_subject1.csv",
        "run1_subject5.csv",
        "fight1_subject3.csv",
        "jumps1_subject2.csv",
        "fallAndGetUp1_subject1.csv",
        "sprint1_subject4.csv",
        "grab_walk_ff_180_001__A550_M.csv",
        "Form_1_stageii_g1.csv",
        "unknown_motion.csv",
    ]

    print("动作标签映射测试：")
    print("-" * 60)
    for f in test_files:
        label = get_motion_label(f)
        print(f"  {f:40s} → {label}")
