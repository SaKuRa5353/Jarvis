"""Пакет команд. Собирает все модули и экспортирует plan_command."""

from __future__ import annotations

from typing import Any

import config
from src.commands import apps, hardware, sensors, system, weather, web
from src.intent import CATALOG, classify, supports
from src.llm import JarvisLLM
from src.logger import get_logger
from src.metrics import timed

logger = get_logger("commands")

# Команды, которым НЕ нужен llm.
# sensors стоит раньше weather: «температура процессора» не должна
# уйти в погодный обработчик по слову «температур».
_SIMPLE_HANDLERS = [apps, system, sensors, web, weather]

# Команды, которым НУЖЕН llm (передаём вторым аргументом)
_LLM_HANDLERS = [hardware]

LLM_MARKER = "__LLM__"

# Модуль-ключ -> (функция try_handle, требует llm).
# Используется интент-веткой: intent.action_allowed -> вызов модуля.
_INTENT_HANDLER_REGISTRY: dict[str, tuple[Any, bool]] = {
    "apps": (apps.try_handle, False),
    "system": (system.try_handle, False),
    "sensors": (sensors.try_handle, False),
    "web": (web.try_handle, False),
    "weather": (weather.try_handle, False),
    "hardware": (hardware.try_handle, True),
}

_INTENT_MODULE_BY_NAME: dict[str, str] = {
    intent.name: intent.handler for intent in CATALOG if intent.handler is not None
}


def _run_intent(intent_name: str, text: str, llm: JarvisLLM) -> Any:
    """Делегирует действие модулю из каталога. Возвращает результат или None."""
    module_key = _INTENT_MODULE_BY_NAME.get(intent_name)
    entry = _INTENT_HANDLER_REGISTRY.get(module_key) if module_key else None
    if entry is None:
        return None
    handler, needs_llm = entry
    return handler(text, llm) if needs_llm else handler(text)


@timed("commands.plan_command")
def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable, phrase_key) или (None, None, None).

    Если правила не сработали — возвращает (LLM_MARKER, None, None).
    LLM-вызов делает assistant.py (для прерывания).
    """
    t = text.lower().replace("джарвис", "").strip()

    # Интент-цепочка: Bayes-классификатор решает, выполнять ли действие.
    if config.INTENT_ENABLED:
        intent_result = classify(t)
        if intent_result.action_allowed and intent_result.intent is not None:
            handler_result = _run_intent(intent_result.intent, t, llm)
            if handler_result is not None:
                return handler_result

    for handler in _SIMPLE_HANDLERS:
        result = handler.try_handle(t)
        if result is not None:
            # Отказ классификатора не должен обходиться через прежние правила.
            if (
                config.INTENT_ENABLED
                and result[1] is not None
                and not any(
                    supports(t, item)
                    for item in CATALOG
                    if item.handler == handler.__name__.rsplit(".", 1)[-1] and item.dangerous
                )
            ):
                return LLM_MARKER, None, None
            return result

    for handler in _LLM_HANDLERS:
        result = handler.try_handle(t, llm)
        if result is not None:
            return result

    # Ничего не сработало — маркер для LLM
    return LLM_MARKER, None, None
