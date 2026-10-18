"""Регрессия fuzzy.correct: исправление ошибок распознавания.

Запуск:
    python tests/check_fuzzy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fuzzy import correct  # noqa: E402

ok = True

print("=== Базовые замены ===")
for src, want in [
    ("тонкость выше", "громкость выше"),
    ("тонкость тише", "громкость тише"),
    ("вэ ка", "вк"),
    ("века", "вк"),
]:
    got = correct(src)
    good = got == want
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] {src!r:25} -> {got!r}")

print("")
print("Известные слова не трогаем ===")
for src, want in [
    ("громкость выше", "громкость выше"),
    ("громче", "громче"),
    ("включи", "включи"),
    ("выключи", "выключи"),
    ("открой вк", "открой вк"),
    ("включи звук", "включи звук"),
    ("привет", "привет"),
]:
    got = correct(src)
    good = got == want
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] {src!r:25} -> {got!r}")

print("")
print("VSYO ZELYONOYE" if ok else "EST PADENIYA")
sys.exit(0 if ok else 1)
