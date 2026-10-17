"""Проверка модуля window_context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.window_context import get_active_window, get_window_context  # noqa: E402

print("=== Window context ===")

w = get_active_window()
print("get_active_window() =", repr(w))

ctx = get_window_context()
print("get_window_context() =", repr(ctx))

print("OK" if w else "FAIL")
