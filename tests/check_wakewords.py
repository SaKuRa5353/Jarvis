"""Проверка wake words на реальных фразах из логов."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import WAKE_WORDS  # noqa: E402

HEARD = [
    "весь",
    "да",
    "гараж",
    "это",
    "она на то",
    "открыла",
    "откровенно на то",
    "гага гага",
    "джарвис",
    "гараж открой код",
    "джервис какой час",
]

for t in HEARD:
    hit = any(w in t for w in WAKE_WORDS)
    print(f"{t!r:25} -> {'WAKE' if hit else '-'}")
