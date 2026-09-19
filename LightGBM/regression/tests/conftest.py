"""Thiết lập đường dẫn cho pytest.

Thêm repo root vào ``sys.path`` để các test có thể import
``regression.lightgbm_regression`` và các package ``regression.*``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
