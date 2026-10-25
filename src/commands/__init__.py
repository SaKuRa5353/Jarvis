"""Пакет команд. Собирает все модули и экспортирует plan_command."""

from src.llm import JarvisLLM
from src.logger import get_logger
from src.metrics import timed

from src.commands import apps, hardware, sensors, system, weather, web


logger = get_logger("commands")

# Команды, которым НЕ нужен llm.
# sensors стоит раньше weather: «температура процессора» не должна
# уйти в погодный обработчик по слову «температур».
_SIMPLE_HANDLERS = [apps, system, sensors, web, weather]

# Команды, которым НУЖЕН llm (передаём вторым аргументом)
_LLM_HANDLERS = [hardware]


LLM_MARKER = "__LLM__"

@timed("commands.plan_command")
def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable, phrase_key) или (None, None, None).

    Если правила не сработали — возвращает (LLM_MARKER, None, None).
    LLM-вызов делает assistant.py (для прерывания).
    """
    t = text.lower().replace("джарвис", "").strip()

    for handler in _SIMPLE_HANDLERS:
        result = handler.try_handle(t)
        if result is not None:
            return result

    for handler in _LLM_HANDLERS:
        result = handler.try_handle(t, llm)
        if result is not None:
            return result

    # Ничего не сработало — маркер для LLM
    return LLM_MARKER, None, None