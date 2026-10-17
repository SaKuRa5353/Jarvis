"""Проверка путей готовых фраз и поисковых паттернов «загугли»."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commands.web import _extract_search_query  # noqa: E402
from src.phrases import get_phrase_path  # noqa: E402

print("=== Phrase paths ===")
ok = True
for key in ("welcome", "opening", "closing", "done", "llm_thinking", "not_listened"):
    path = get_phrase_path(key)
    exists = path is not None and path.exists()
    if not exists:
        ok = False
    print(f"  [{'OK ' if exists else 'FAIL'}] {key}: {path}")

print("")
print("=== Search patterns (Vosk variants of 'загугли') ===")
CASES = [
    ("за гугле топ двадцать ноутбуков", "топ двадцать ноутбуков"),
    ("загугли python", "python"),
    ("за гугли python", "python"),
    ("загули котов", "котов"),
    ("погугли погоду", "погоду"),
    ("найди в гугле котиков", "котиков"),
    ("открой ютуб", None),
]
for text, want in CASES:
    got = _extract_search_query(text)
    mark = "OK " if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"  [{mark}] {text!r:40} -> {got!r} (want {want!r})")

print("")
print("VSYO ZELYONOYE" if ok else "EST PADENIYA")
