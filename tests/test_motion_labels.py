"""测试 motion_labels.get_motion_label — 文件名到语言标签映射。"""

from pathlib import Path

import pytest

from src.configs.motion_labels import get_motion_label


class TestGetMotionLabel:
    """get_motion_label 测试。"""

    def test_known_keyword_walk(self):
        assert get_motion_label("walk1_subject1.csv") == "walk forward"

    def test_moonwalk_priority_over_walk(self):
        """LABEL_MAP 中 moonwalk 在 walk 之前，应优先命中。"""
        assert get_motion_label("moonwalk1_subject2.csv") == "slide feet backward"

    def test_strip_g1_retargeted_suffix(self):
        """_g1 / _retargeted 后缀应被剥离后仍命中 walk。"""
        assert get_motion_label("walk_g1_retargeted.csv") == "walk forward"

    def test_csv_extension_with_numeric_suffix(self):
        """带 .csv 扩展名 + 数字后缀的文件应命中某个 keyword。"""
        label = get_motion_label("grab_walk_ff_180_001__A550_M.csv")
        # 文件名含 walk，命中 walk forward
        assert label == "walk forward"

    def test_unknown_file_returns_default(self):
        assert get_motion_label("unknown_motion.csv") == "perform the locomotion task"

    def test_custom_default(self):
        label = get_motion_label("unknown_motion.csv", default="do something")
        assert label == "do something"

    def test_run_keyword(self):
        assert get_motion_label("run1_subject5.csv") == "run forward"

    def test_dance_keyword(self):
        assert get_motion_label("dance1_subject2.csv") == "perform dancing motion"

    def test_fight_keyword(self):
        assert get_motion_label("fight1_subject3.csv") == "perform fighting motion"

    def test_high_knee_priority(self):
        """high_knee 在 LABEL_MAP 中位于 walk 之前，应优先命中。"""
        assert get_motion_label("high_knee1_subject1.csv") == "march with high knees"

    def test_backpedal_keyword(self):
        assert get_motion_label("backpedal2_subject3.csv") == "walk backward"

    def test_path_object_input(self):
        """get_motion_label 应能处理 Path 对象（通过 Path.stem）。"""
        assert get_motion_label(Path("walk1_subject1.csv")) == "walk forward"
