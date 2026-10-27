"""Pytest-фикстуры: проектный корень в sys.path (как в check_*.py)."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
