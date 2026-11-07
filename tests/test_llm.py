"""Unit-тесты LLM-ответов и маршрутизации function calling."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import junk_phrase  # noqa: E402


def _tool_response(expression: str):
    call = SimpleNamespace(
        function=SimpleNamespace(name="calculate", arguments={"expression": expression}),
    )
    message = SimpleNamespace(content=None, tool_calls=[call], reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


def test_junk_answers_map_to_phrase_key():
    assert junk_phrase("Не расслышал") == "not_listened"
    assert junk_phrase("не придумывай факты") == "not_listened"


def test_real_answers_are_not_junk():
    assert junk_phrase("Сколько будет четыре") is None
    assert junk_phrase("Привет! Чем помочь?") is None


def test_natural_language_arithmetic_routes_to_calculate(monkeypatch) -> None:
    """Фраза пользователя приводит к tool call и точному числовому ответу."""
    from src.llm import JarvisLLM

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: _tool_response("sin(pi/2)"),
            )
        )
    )
    llm = object.__new__(JarvisLLM)
    llm.client = fake_client
    llm.model = "test"
    llm.history = []

    result = llm.ask("сколько будет синус пи пополам")

    assert result == ("1", None)
