"""Проверки маршрутизации отклонённых интентов."""

import pytest

from src import assistant, commands
from src.intent import IntentResult


def test_rejected_intent_does_not_start_llm(monkeypatch):
    """Отклонённая команда не должна попадать в LLM."""
    monkeypatch.setattr(assistant, "correct", lambda text: text)
    monkeypatch.setattr(
        commands,
        "classify",
        lambda _text: IntentResult(
            intent="web.vk",
            prob=0.10,
            margin=0.0,
            action_allowed=False,
            fallback=True,
        ),
    )
    monkeypatch.setattr(commands, "_SIMPLE_HANDLERS", [])
    monkeypatch.setattr(commands, "_LLM_HANDLERS", [])

    def fail_if_llm_called(*_args, **_kwargs):
        pytest.fail("Отклонённая классификатором команда попала в LLM")

    assistant._process_command(
        text="открой вк",
        llm=object(),
        detector=object(),
        plan_command_func=commands.plan_command,
        _ask_fn=fail_if_llm_called,
        _speak_fn=lambda *_args, **_kwargs: None,
        _get_window_ctx=lambda: None,
    )


def test_text_to_safe_action_uses_production_route_without_external_call(monkeypatch):
    """Разрешённая фраза проходит весь маршрут, но внешний запуск перехвачен."""
    opened_urls: list[str] = []
    spoken: list[tuple[str, str | None]] = []

    monkeypatch.setattr(assistant, "correct", lambda text: text)
    monkeypatch.setattr(
        "src.commands.web.open_url",
        lambda url: opened_urls.append(url),
    )

    def fail_if_llm_called(*_args, **_kwargs):
        pytest.fail("Разрешённая команда неожиданно попала в LLM")

    assistant._process_command(
        text="открой ютуб",
        llm=object(),
        detector=object(),
        plan_command_func=commands.plan_command,
        _ask_fn=fail_if_llm_called,
        _speak_fn=lambda _detector, response, phrase: spoken.append((response, phrase)),
        _get_window_ctx=lambda: None,
    )

    assert opened_urls == ["https://www.youtube.com/"]
    assert spoken == [("Открываю ютуб", "opening")]


def test_rejected_production_route_does_not_execute_action_or_start_llm(monkeypatch):
    """Отклонённая production-команда не должна выполнять подготовленное действие."""
    opened_urls: list[str] = []
    llm_calls: list[object] = []

    monkeypatch.setattr(assistant, "correct", lambda text: text)
    monkeypatch.setattr(
        "src.commands.web.open_url",
        lambda url: opened_urls.append(url),
    )
    monkeypatch.setattr(
        commands,
        "classify",
        lambda _text: IntentResult("web.youtube", 0.1, 0.0, False, True),
    )

    def fail_if_llm_called(*args, **_kwargs):
        llm_calls.append(args)
        pytest.fail("Отклонённая команда попала в LLM")

    assistant._process_command(
        text="открой ютуб",
        llm=object(),
        detector=object(),
        plan_command_func=commands.plan_command,
        _ask_fn=fail_if_llm_called,
        _speak_fn=lambda *_args, **_kwargs: None,
        _get_window_ctx=lambda: None,
    )

    assert opened_urls == []
    assert llm_calls == []


@pytest.mark.parametrize("text", ["сделай что-нибудь", "включи свет"])
def test_unknown_or_ambiguous_text_has_no_direct_action(text):
    """Неизвестная или неоднозначная фраза не должна получить действие напрямую."""
    response, action, phrase = commands.plan_command(text, llm=None)

    assert response == commands.LLM_MARKER
    assert action is None
    assert phrase is None


def test_dangerous_command_without_confirmation_is_terminal(monkeypatch):
    """Опасная команда без разрешения не должна попасть ни в handler, ни в LLM."""
    executed: list[str] = []
    llm_calls: list[object] = []

    monkeypatch.setattr(assistant, "correct", lambda text: text)
    monkeypatch.setattr(
        commands,
        "classify",
        lambda _text: IntentResult("system.lock", 0.9, 0.8, False, True),
    )
    monkeypatch.setattr(
        "src.commands.system.run",
        lambda args: executed.append(" ".join(args)),
    )

    def fail_if_llm_called(*args, **_kwargs):
        llm_calls.append(args)
        pytest.fail("Опасная команда без подтверждения попала в LLM")

    assistant._process_command(
        text="заблокируй экран",
        llm=object(),
        detector=object(),
        plan_command_func=commands.plan_command,
        _ask_fn=fail_if_llm_called,
        _speak_fn=lambda *_args, **_kwargs: None,
        _get_window_ctx=lambda: None,
    )

    assert executed == []
    assert llm_calls == []