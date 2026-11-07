"""Тесты обработки пустых и повторных ответов погодного сервиса."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.commands import weather


def test_weather_retries_after_empty_response(monkeypatch) -> None:
    responses = [
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(returncode=0, stdout=json.dumps({"current_condition": []})),
    ]

    def run(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(weather.subprocess, "run", run)

    assert weather._fetch_weather("Петрозаводск") == {"current_condition": []}
    assert not responses


def test_weather_returns_none_after_two_empty_responses(monkeypatch) -> None:
    monkeypatch.setattr(
        weather.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    assert weather._fetch_weather("Петрозаводск") is None