# Test package for gr00t_mjlab_autodl.

import os
import sys
from pathlib import Path

# 让 `import src.collect_data` 等不需要 src/ 在 cwd 也能工作
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 让 new_embodiment_config / GR00T 相关 import 能找到 Isaac-GR00T
ISAAC_GR00T = PROJECT_ROOT.parent / "Isaac-GR00T"
if ISAAC_GR00T.exists() and str(ISAAC_GR00T) not in sys.path:
    sys.path.insert(0, str(ISAAC_GR00T))