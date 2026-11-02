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