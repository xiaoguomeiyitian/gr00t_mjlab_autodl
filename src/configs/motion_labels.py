"""
motion_labels.py — 动作文件名到语言标签的映射。

从 robot_retargeter 的动作文件名自动推断 GR00T 训练所需的语言描述。

用法:
    from src.configs.motion_labels import get_motion_label, LABEL_MAP
    
    label = get_motion_label("dance1_subject2.csv")  # → "perform dancing motion"
"""

import re
from typing import Optional

# ─── 动作关键词 → 语言标签映射 ───
# 按优先级排序（先匹配的先返回）
LABEL_MAP = [
    # (关键词, 语言标签)
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
    """
    从动作文件名推断语言标签。

    Args:
        filename: 动作文件名（如 "dance1_subject2.csv"）
        default: 默认标签（无法匹配时返回）

    Returns:
        语言描述字符串

    Examples:
        >>> get_motion_label("dance1_subject2.csv")
        'perform dancing motion'
        >>> get_motion_label("walk3_subject1.csv")
        'walk forward'
        >>> get_motion_label("unknown_motion.csv")
        'perform the locomotion task'
    """
    # 提取文件名（不含路径和扩展名）
    name = filename.split("/")[-1].split("\\")[-1]
    name_lower = name.lower()

    # 移除常见前缀/后缀
    name_lower = re.sub(r"(_from_.*|_g1|_h2|_h1|_origin|_retargeted)$", "", name_lower)
    name_lower = re.sub(r"_\d{3}_\d{3}_\d{3}$", "", name_lower)  # 移除 _001_001_001 后缀

    # 按优先级匹配
    for keyword, label in LABEL_MAP:
        if keyword in name_lower:
            return label

    return default


def get_all_labels() -> dict:
    """返回所有已注册的动作标签映射（关键词 → 标签）。"""
    return dict(LABEL_MAP)


def add_custom_label(keyword: str, label: str) -> None:
    """
    添加自定义动作标签映射。

    Args:
        keyword: 关键词（匹配文件名）
        label: 对应的语言描述
    """
    LABEL_MAP.insert(0, (keyword.lower(), label))  # 插入到最前面，优先级最高


# ─── 测试 ───
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
