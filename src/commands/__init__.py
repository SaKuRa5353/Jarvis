"""Пакет команд. Собирает все модули и экспортирует plan_command."""

from src.llm import JarvisLLM
from src.logger import get_logger

from src.commands import apps, hardware, system, web


logger = get_logger("commands")

# Команды, которым НЕ нужен llm
_SIMPLE_HANDLERS = [apps, system, web]

# Команды, которым НУЖЕН llm (передаём вторым аргументом)
_LLM_HANDLERS = [hardware]


def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable) или (None, None)."""
    t = text.lower().replace("джарвис", "").strip()

    # Сначала простые команды
    for handler in _SIMPLE_HANDLERS:
        result = handler.try_handle(t)
        if result is not None:
            return result

    # Потом команды, которым нужна LLM
    for handler in _LLM_HANDLERS:
        result = handler.try_handle(t, llm)
        if result is not None:
            return result

    # Fallback: LLM
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None