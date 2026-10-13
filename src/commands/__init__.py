"""Пакет команд. Собирает все модули и экспортирует plan_command."""

from src.llm import JarvisLLM
from src.logger import get_logger

from src.commands import apps, system, web


logger = get_logger("commands")

_HANDLERS = [apps, system, web]


def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable) или (None, None)."""
    t = text.lower().replace("джарвис", "").strip()

    for handler in _HANDLERS:
        result = handler.try_handle(t)
        if result is not None:
            return result

    # Fallback: LLM
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None
