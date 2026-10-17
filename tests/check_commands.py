"""Регрессия правил plan_command: громкость, ютуб, поиск, время."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commands import LLM_MARKER, plan_command  # noqa: E402

CASES = [
    # (вход, ожидаемое начало response или LLM_MARKER, ожидаемая phrase_key)
    ("громкость выше", "Делаю громче", "volume_up"),
    ("громкость тише", "Делаю тише", "volume_down"),
    ("сделай погромче", "Делаю громче", "volume_up"),
    ("убавь", "Делаю тише", "volume_down"),
    ("добавь звук", "Делаю громче", "volume_up"),
    ("ютуб", "Открываю", "opening"),
    ("открой ютуб", "Открываю", "opening"),
    ("открой вк", "Открываю", "opening"),
    ("открой века", "Открываю", "opening"),
    ("про века", LLM_MARKER, None),        # омоним — не ссылка
    ("включи звук", "Включаю звук", "volume_unmute"),
    ("выключи звук", "Выключаю звук", "volume_mute"),
    ("открой почту", "Открываю", "opening"),
    ("открой дипсик", "Открываю", "opening"),
    ("за гугле погода", "Ищу погода", None),
    ("который час", "Сейчас", None),
    ("какая погода", None, None),          # weather — свой ответ
    ("расскажи анекдот", LLM_MARKER, None),
    ("привет", LLM_MARKER, None),
]

ok = True
for text, want_prefix, want_phrase in CASES:
    response, action, phrase = plan_command(text, llm=None)
    if want_prefix == LLM_MARKER:
        good = response == LLM_MARKER
    elif want_prefix is None:
        good = response is not None and response != LLM_MARKER
    else:
        good = response is not None and response.startswith(want_prefix)
    if want_phrase is not None:
        good = good and phrase == want_phrase
    if not good:
        ok = False
    mark = "OK " if good else "FAIL"
    print(f"  [{mark}] {text!r:25} -> {response!r} / {phrase!r}")

print("VSYO ZELYONOYE" if ok else "EST PADENIYA")
sys.exit(0 if ok else 1)
