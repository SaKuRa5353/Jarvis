"""Unit-тесты fuzzy.correct с движком WDL (combined_distance)."""

import src.fuzzy as fuzzy_module
from src.fuzzy import correct


def test_two_vosk_fixes_in_one_phrase():
    """Несколько искажений в одной фразе исправляются все подряд."""
    assert correct("тонкость выше вэ ка") == "громкость выше вк"


def test_wdl_engine_corrects_unknown_word():
    """WDL-движок исправляет неизвестное слово через combined_distance."""
    assert correct("громкот", engine="wdl") == "громкость"


def test_difflib_engine_keeps_previous_behavior():
    """engine="difflib" воспроизводит прежнее поведение."""
    assert correct("громкот", engine="difflib") == "громкость"


def test_wdl_engine_threshold_respected():
    """Порог similarity: при threshold выше максимума исправление не происходит."""
    assert correct("громкот", engine="wdl", threshold=0.99) == "громкот"


def test_stopwords_are_protected(monkeypatch):
    """Стоп-слова не должны уходить в fuzzy-коррекцию."""
    monkeypatch.setattr(fuzzy_module, "STOPWORDS_CORRECT", frozenset(("громкот",)))
    assert correct("громкот") == "громкот"


def test_config_engine_switches_default(monkeypatch):
    """Значение FUZZY_ENGINE из config влияет на движок по умолчанию."""
    monkeypatch.setattr(fuzzy_module, "FUZZY_ENGINE", "difflib")
    assert correct("громкот") == "громкость"


def test_temperatura_processora_stays_stable():
    """Стоп-лист «температура процессора» не должен портиться."""
    assert correct("температура процессора") == "температура процессора"
