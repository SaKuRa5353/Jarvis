"""Unit-тесты интентов: Bayes-классификатор намерений и решение action_allowed."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import config
from src.intent import CATALOG, Intent, IntentResult, classify, fit_catalog, supports


def test_classify_open_vk():
    result = classify("открой вк")
    assert result.intent == "web.vk"
    assert result.action_allowed is True
    assert result.fallback is False


@pytest.mark.parametrize(
    "text",
    [
        "ярость открой века",
        "тяжесть",
        "смотрят моя история",
        "принадлежит валя",
        "об этом не узнает",
        "казака",
        "до киева экран",
    ],
)
def test_garbage_does_not_allow_action(text):
    result = classify(text)
    assert result.action_allowed is False
    assert result.fallback is True


def test_intent_result_fields_types():
    result = classify("который час")
    assert isinstance(result, IntentResult)
    assert isinstance(result.intent, str)
    assert 0.0 <= result.prob <= 1.0
    assert 0.0 <= result.margin <= 1.0


def test_fit_catalog_returns_model():
    model = fit_catalog(CATALOG)
    assert model is not None
    proba = model.predict_proba("открой вк")
    assert sum(proba.values()) > 0.0


def test_intent_disabled_allows_nothing(monkeypatch):
    monkeypatch.setattr(config, "INTENT_ENABLED", False)
    result = classify("открой вк")
    assert result.action_allowed is False
    assert result.fallback is True
    assert result.intent is None


def test_catalog_intents_have_handlers_and_patterns():
    assert len(CATALOG) > 0
    for intent in CATALOG:
        assert isinstance(intent, Intent)
        assert intent.name
        assert isinstance(intent.patterns, tuple)
        assert len(intent.patterns) > 0
        assert isinstance(intent.dangerous, bool)
        assert intent.handler is not None


def test_supports_rejects_unknown_context() -> None:
    """Поддержка S(q,c) требует командного контекста, а не совпадения подстроки."""
    vk = next(item for item in CATALOG if item.name == "web.vk")
    assert supports("открой вк", vk)
    assert not supports("ярость открой века", vk)
    assert not supports("про века", vk)
    assert not supports("включи свет", vk)


@pytest.mark.parametrize(
    "text",
    [
        "ярость открой века",
        "тяжесть",
        "смотрят моя история",
        "принадлежит валя",
        "об этом не узнает",
        "казака",
        "до киева экран",
    ],
)
def test_garbage_has_no_action_in_pipeline(text: str) -> None:
    from src.commands import LLM_MARKER, plan_command

    response, action, _ = plan_command(text, llm=None)
    assert action is None
    assert response == LLM_MARKER


def _command_cases() -> list[tuple[str, str | None, str | None]]:
    """Читает CASES через AST: C = литерал, без исполнения проверочного скрипта."""
    tree = ast.parse(Path(__file__).with_name("check_commands.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets
        ):
            rows = []
            for row in node.value.elts:
                rows.append(
                    tuple(
                        "__LLM__" if isinstance(cell, ast.Name) else ast.literal_eval(cell)
                        for cell in row.elts
                    )
                )
            return rows
    raise ValueError("В проверочном скрипте отсутствует CASES")


@pytest.mark.parametrize("text,prefix,phrase", _command_cases())
def test_command_cases_match_legacy(text, prefix, phrase, monkeypatch) -> None:
    """Все C из CASES сохраняют ответ, наличие действия и ключ прежнего плана."""
    from src.commands import plan_command, weather

    monkeypatch.setattr(weather, "_fetch_weather", lambda city: None)
    monkeypatch.setattr(config, "INTENT_ENABLED", False)
    legacy = plan_command(text, llm=None)
    monkeypatch.setattr(config, "INTENT_ENABLED", True)
    current = plan_command(text, llm=None)
    assert current[0] == legacy[0]
    assert (current[1] is not None) == (legacy[1] is not None)
    assert current[2] == legacy[2]
    expected = {
        "громкость выше": "system.volume_up",
        "громкость тише": "system.volume_down",
        "сделай погромче": "system.volume_up",
        "убавь": "system.volume_down",
        "добавь звук": "system.volume_up",
        "ютуб": "web.youtube",
        "открой ютуб": "web.youtube",
        "открой вк": "web.vk",
        "открой века": "web.vk",
        "про века": None,
        "включи звук": "system.volume_unmute",
        "выключи звук": "system.volume_mute",
        "открой почту": "web.mail",
        "открой дипсик": "web.deepseek",
        "открой музыку": "web.music",
        "музыка": "web.music",
        "за гугле погода": "web.search",
        "который час": "system.time",
        "какая погода": "weather.current",
        "расскажи анекдот": None,
        "привет": None,
    }
    assert classify(text).intent == expected[text]


def test_disabled_bypasses_classifier(monkeypatch) -> None:
    from src import commands

    monkeypatch.setattr(config, "INTENT_ENABLED", False)

    def fail(text: str) -> None:
        raise AssertionError("Выключенный классификатор вызван")

    monkeypatch.setattr(commands, "classify", fail)
    response, action, phrase = commands.plan_command("ярость открой века", llm=None)
    assert response.startswith("Открываю")
    assert action is not None
    assert phrase == "opening"


@pytest.mark.parametrize(
    "text,name",
    [
        ("открой дипсик", "web.deepseek"),
        ("открой расписание", "web.schedule"),
        ("какая погода", "weather.current"),
        ("прогноз на завтра", "weather.forecast"),
        ("температура процессора", "sensor.cpu"),
        ("добавь звук", "system.volume_up"),
    ],
)
def test_catalog_routes_actual_commands(text: str, name: str) -> None:
    assert classify(text).intent == name


def test_thresholds_read_from_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "INTENT_TAU_DANGEROUS", 1.1)
    assert not classify("открой вк").action_allowed
