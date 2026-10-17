"""Проверка Silero TTS: очистка текста, готовые фразы, синтез и воспроизведение.

Запуск:
    python tests/check_tts.py            # с воспроизведением
    python tests/check_tts.py --no-play  # только проверка файлов и текста
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phrases import PHRASES, get_phrase_path  # noqa: E402
from src.tts import _fix_text, speak  # noqa: E402

play = "--no-play" not in sys.argv
ok = True

print("=== Очистка текста (_fix_text) ===")
FIX_CASES = [
    ("Джарвис готов", "Дж+арвис готов"),
    ("50% и 20%", "50 процентов и 20 процентов"),
    ("путь/к/файлу", "путь к файлу"),
    ("a  b", "a b"),
]
for src, want in FIX_CASES:
    got = _fix_text(src)
    good = got == want
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] {src!r:20} -> {got!r}")

print("")
print(f"=== Готовые фразы ({len(PHRASES)}) ===")
for key in PHRASES:
    path = get_phrase_path(key)
    exists = path is not None and path.exists()
    ok = ok and exists
    print(f"  [{'OK ' if exists else 'FAIL'}] {key}: {path.name if path else None}")

if play:
    print("")
    print("=== Синтез через Silero ===")
    text = "Джарвис готов к работе. Все системы в норме."
    print(f"  Текст: {text!r}")
    speak(text)
    print("  Если услышал голос — TTS работает")

print("")
print("VSYO ZELYONOYE" if ok else "EST PADENIYA")
sys.exit(0 if ok else 1)
