"""Unit-тесты plan_command: правила команд (без реального LLM/железа)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commands import LLM_MARKER, plan_command  # noqa: E402


def test_volume_up():
    response, action, phrase = plan_command("громкость выше", llm=None)
    assert response == "Делаю громче"
    assert phrase == "volume_up"


def test_volume_down():
    response, action, phrase = plan_command("сделай погромче", llm=None)
    assert response == "Делаю громче"
    assert phrase == "volume_up"


def test_quick_link_open():
    response, action, phrase = plan_command("открой ютуб", llm=None)
    assert response == "Открываю ютуб"
    assert phrase == "opening"


def test_bare_quick_link():
    response, action, phrase = plan_command("музыка", llm=None)
    assert response.startswith("Открываю")
    assert phrase == "opening"


def test_llm_marker_for_chat():
    assert plan_command("расскажи анекдот", llm=None)[0] == LLM_MARKER


def test_time_command():
    response, action, phrase = plan_command("который час", llm=None)
    assert response.startswith("Сейчас")
