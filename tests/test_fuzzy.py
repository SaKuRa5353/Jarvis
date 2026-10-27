"""Unit-тесты fuzzy.correct: базовые замены и «все исправления сразу»."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fuzzy import correct  # noqa: E402


def test_simple_correction():
    assert correct("тонкость выше") == "громкость выше"
    assert correct("тонкость тише") == "громкость тише"


def test_vosk_variants():
    assert correct("вэ ка") == "вк"
    assert correct("века") == "вк"
    assert correct("дип сик") == "дипсик"


def test_all_fixes_applied_not_only_first():
    """Регрессия Phase 3: цикл замен не должен выходим после первой найденной."""
    assert correct("тонкость за гугли") == "громкость загугли"


def test_known_words_are_not_touched():
    assert correct("включи звук") == "включи звук"
    assert correct("выключи свет") == "выключи свет"


def test_garbage_is_stable():
    assert correct("джарвис") == "джарвис"


def test_known_words_are_not_mangled():
    """«процессора» — полноценное слово, досоверно не должно исправляться."""
    assert correct("температура процессора") == "температура процессора"
