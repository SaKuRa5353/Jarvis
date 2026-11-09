"""Тесты обработки пустых и повторных ответов погодного сервиса."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from src.commands import weather


def _mk_response(returncode: int, stdout: str, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _make_fetch_response(monkeypatch: pytest.MonkeyPatch, data_dict: dict | None) -> None:
    """Ставит заглушку subprocess.run, всегда возвращающую ``data_dict`` как JSON."""
    if data_dict is None:
        def stub_empty(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return _mk_response(0, "")
        monkeypatch.setattr(weather.subprocess, "run", stub_empty)
        return
    payload = json.dumps(data_dict, ensure_ascii=False)

    def stub_payload(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return _mk_response(0, payload)

    monkeypatch.setattr(weather.subprocess, "run", stub_payload)


def test_weather_retries_after_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Первый вызов subprocess.run возвращает пустой stdout, второй — валидный JSON."""
    responses = [
        _mk_response(0, ""),
        _mk_response(0, json.dumps({"current_condition": []})),
    ]

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return responses.pop(0)

    monkeypatch.setattr(weather.subprocess, "run", run)

    assert weather._fetch_weather("Петрозаводск") == {"current_condition": []}
    assert not responses


def test_weather_returns_none_after_two_empty_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Два подряд пустых ответа приводят к возврату None."""

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return _mk_response(0, "")

    monkeypatch.setattr(weather.subprocess, "run", run)

    assert weather._fetch_weather("Петрозаводск") is None


class TestTryHandleEmptyResponseGuard:
    """Регрессия: try_handle не должен озвучивать обрывки при пустом ответе."""

    def test_none_fetch_returns_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Полный промах -- нет интернета или сервис недоступен
        _make_fetch_response(monkeypatch, None)
        result = weather.try_handle("какая погода")
        assert result is not None
        response, action, phrase = result
        assert response == "Не могу получить данные о погоде. Проверь интернет."
        assert action is None
        assert phrase is None

    def test_truncated_current_missing_feels_and_humidity(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Минимальный валидный JSON вернёт неполную но осмысленную строку
        _make_fetch_response(
            monkeypatch,
            {
                "nearest_area": [{"areaName": [{"value": "Москва"}]}],
                "current_condition": [
                    {"temp_C": "5", "FeelsLikeC": "2", "windspeedKmph": "10", "humidity": "80"},
                ],
            },
        )
        result = weather.try_handle("что за погодой")
        assert result is not None
        response, action, phrase = result
        assert isinstance(response, str)
        assert len(response.split()) >= 3
        assert "Москва" in response
        assert action is None
        assert phrase is None

    def test_valid_today_and_tomorrow_return_full_strings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        good_data = {
            "nearest_area": [{"areaName": [{"value": "Самара"}]}],
            "current_condition": [
                {"temp_C": "6", "FeelsLikeC": "3", "windspeedKmph": "14", "humidity": "79"},
            ],
            "weather": [
                {},
                {
                    "mintempC": "-2",
                    "maxtempC": "2",
                    "hourly": [{} for _ in range(8)],
                },
            ],
        }
        _make_fetch_response(monkeypatch, good_data)
        cur = weather.try_handle("погода самара")
        assert cur is not None
        assert isinstance(cur[0], str) and len(cur[0].split()) >= 5

        tom = weather.try_handle("прогноз на завтра")
        assert tom is not None
        assert isinstance(tom[0], str) and len(tom[0].split()) >= 5 and "завтра" in tom[0].lower()

    def test_completely_malformed_json_is_guared_by_fetch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ошибка парсинга JSON -- fetch вернёт None, try_handle отдаст отказ
        def stub(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return _mk_response(0, "{bad json}")

        monkeypatch.setattr(weather.subprocess, "run", stub)
        result = weather.try_handle("какая погода")
        assert result is not None
        assert result[0] == "Не могу получить данные о погоде. Проверь интернет."

    def test_empty_response_returns_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Пустая строка stdout на первой попытке -- повтор, потом отказ
        def stub(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return _mk_response(0, "")

        monkeypatch.setattr(weather.subprocess, "run", stub)
        result = weather.try_handle("погода в москве")
        assert result is not None
        assert result[0] == "Не могу получить данные о погоде. Проверь интернет."

    def test_whitespace_only_response_returns_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Только пробелы/переносы -- считается пустым, возвращает отказ
        def stub(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return _mk_response(0, "\n   \t\n")

        monkeypatch.setattr(weather.subprocess, "run", stub)
        result = weather.try_handle("погода")
        assert result is not None
        assert result[0] == "Не могу получить данные о погоде. Проверь интернет."

    def test_timeout_returns_none_then_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Таймаут subprocess -- обработка через except, retry, затем отказ
        def stub(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            raise subprocess.TimeoutExpired(cmd=b"curl", timeout=12)

        monkeypatch.setattr(weather.subprocess, "run", stub)
        result = weather.try_handle("погода")
        assert result is not None
        assert result[0] == "Не могу получить данные о погоде. Проверь интернет."

    def test_nonzero_returncode_returns_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # curl завершился с ненулевым кодом -- повтор, затем отказ
        def stub(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return _mk_response(1, "", "curl error")

        monkeypatch.setattr(weather.subprocess, "run", stub)
        result = weather.try_handle("погода")
        assert result is not None
        assert result[0] == "Не могу получить данные о погоде. Проверь интернет."


def test_city_normalization_and_url_encoding() -> None:
    """Проверка нормализации склоняемых городов и экранирования URL."""
    assert weather.normalize_city_name("москве") == "Москва"
    assert weather.normalize_city_name("питере") == "Санкт-Петербург"
    assert weather.normalize_city_name("самаре") == "Самара"
    assert weather._extract_city("какая погода в Москве") == "Москва"
    assert weather._extract_city("погода в питере") == "Санкт-Петербург"
