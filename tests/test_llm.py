"""Unit-тесты LLM-ответов и маршрутизации function calling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from src.llm import junk_phrase  # noqa: E402


def _tool_response(expression: str) -> SimpleNamespace:
    call = SimpleNamespace(
        id="call_calc",
        function=SimpleNamespace(
            name="calculate",
            arguments='{"expression":"' + expression + '"}',
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[call], reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


def test_junk_answers_map_to_phrase_key() -> None:
    assert junk_phrase("Не расслышал") == "not_listened"
    assert junk_phrase("не придумывай факты") == "not_listened"


def test_real_answers_are_not_junk() -> None:
    assert junk_phrase("Сколько будет четыре") is None
    assert junk_phrase("Привет! Чем помочь?") is None


def test_natural_language_arithmetic_routes_to_calculate(monkeypatch: object) -> None:
    """Фраза пользователя приводит к tool call и точному числовому ответу."""
    del monkeypatch
    from src.llm import JarvisLLM

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: _tool_response("sin(pi/2)"),
            )
        )
    )
    llm = object.__new__(JarvisLLM)
    llm.model = "test"
    llm.history = []
    llm.client = cast(OpenAI, fake_client)

    result = llm.ask("сколько будет синус пи пополам")
    assert result == ("1", None)


def test_raw_json_in_text_is_not_spoken_or_executed() -> None:
    """Модель прислала JSON в виде обычного текста: проверяем что он
    НЕ исполняется как инструмент и НЕ попадает в on_delta/TTS."""
    from src.llm import JarvisLLM

    raw_json = '{"name": "calculate", "arguments": {"expression": "2+2"}}'
    message = SimpleNamespace(content=raw_json, tool_calls=None, reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")

    def make_response(**kwargs: object) -> SimpleNamespace | list[object]:
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[], delta=SimpleNamespace(content="")),
                SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", delta=None)]),
            ]
        return SimpleNamespace(choices=[choice])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=make_response))
    )

    llm = object.__new__(JarvisLLM)
    llm.model = "test"
    llm.history = []
    llm.client = cast(OpenAI, fake_client)

    delta_chunks: list[str] = []

    def on_delta(chunk: str) -> None:
        delta_chunks.append(chunk)

    # non-streaming path — JSON остаётся обычным текстом
    with patch.object(llm, "client", fake_client, create=True):
        result = llm.ask("вычисли два плюс два")
        assert result is not None
        answer, phrase_key = result
        assert answer != "4"
        assert delta_chunks == []

    # streaming path — чанки content не должны превратиться в tool call
    with patch.object(llm, "client", fake_client, create=True):
        delta_chunks.clear()
        result = llm.ask("вычисли два плюс два", on_delta=on_delta)
        assert result is not None
        answer, phrase_key = result
        assert answer != "4"


def test_structured_tool_call_still_works() -> None:
    """Структурированный tool_call (через SDK) обрабатывается корректно."""
    from src.llm import JarvisLLM

    call_ns = SimpleNamespace(
        id="call_1",
        index=0,
        function=SimpleNamespace(name="calculate", arguments='{"expression":"3*4"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[call_ns], reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")

    def make_response(**kwargs: object) -> SimpleNamespace | list[object]:
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[], delta=SimpleNamespace(content="")),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[call_ns]))]
                ),
                SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", delta=None)]),
            ]
        return SimpleNamespace(choices=[choice])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=make_response))
    )

    llm = object.__new__(JarvisLLM)
    llm.model = "test"
    llm.history = []
    llm.client = cast(OpenAI, fake_client)

    delta_chunks: list[str] = []

    def on_delta(chunk: str) -> None:
        delta_chunks.append(chunk)

    with patch.object(llm, "client", fake_client, create=True):
        result = llm.ask("вычисли три умножить на четыре", on_delta=on_delta)

    assert result == ("12", None)
