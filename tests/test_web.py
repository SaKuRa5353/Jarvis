"""Unit-тесты поиска в гугле (_extract_search_query)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commands.web import _extract_search_query  # noqa: E402


def test_search_simple():
    assert _extract_search_query("загугли python") == "python"


def test_search_with_word_order():
    assert _extract_search_query("за гугле топ двадцать ноутбуков") == "топ двадцать ноутбуков"


def test_search_explicit():
    assert _extract_search_query("найди в гугле котиков") == "котиков"


def test_open_command_is_not_search():
    assert _extract_search_query("открой ютуб") is None
