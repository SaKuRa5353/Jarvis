"""Роутер команд с fallback на LLM."""
from src.logger import get_logger

logger = get_logger("router")

import subprocess

from src.llm import JarvisLLM


def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable) или (None, None). Ничего не выполняет."""
    t = text.lower()
    t = t.replace("джарвис", "").strip()

    if any(w in t for w in ["проводник", "файлы", "документы", "папк"]):
        return (
            "Открываю проводник",
            lambda: subprocess.Popen(
                ["dolphin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    if any(w in t for w in [
        "vscode", "вс код", "вс-код", "vs code", "visual studio",
        "вс студия", "код",
    ]):
        return (
            "Запускаю редактор кода",
            lambda: subprocess.Popen(
                ["code"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    if any(w in t for w in ["браузер", "brave", "брейв"]):
        return (
            "Открываю браузер",
            lambda: subprocess.Popen(
                ["brave"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    # Fallback: LLM
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None
