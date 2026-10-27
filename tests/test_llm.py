"""Unit-тесты LLM-ответов: мусорные ответы и fallback-фраза."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import junk_phrase  # noqa: E402


def test_junk_answers_map_to_phrase_key():
    assert junk_phrase("Не расслышал") == "not_listened"
    assert junk_phrase("не придумывай факты") == "not_listened"


def test_real_answers_are_not_junk():
    assert junk_phrase("Сколько будет четыре") is None
    assert junk_phrase("Привет! Чем помочь?") is None
