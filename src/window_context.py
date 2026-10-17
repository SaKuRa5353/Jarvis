"""Получение имени активного окна через kdotool.

Используется для контекста LLM: «Ты сейчас в окне: VSCode».
Не для управления — только чтение.

Caching: результат кэшируется на 2 секунды, чтобы не спамить
KWin DBus при каждом запросе.
"""

import subprocess
import time
from typing import Optional

from config import (
    KDTOOL_PATH,
    WINDOW_CONTEXT_CACHE_SEC,
    WINDOW_CONTEXT_ENABLED,
    WINDOW_CONTEXT_TIMEOUT,
)

from src.logger import get_logger

logger = get_logger("window_context")

UNKNOWN = "неизвестно"


# ──────────────────────── СОСТОЯНИЕ ────────────────────────

_cache: str | None = None
_cache_time: float = 0.0


def get_active_window() -> str:
    """Возвращает имя активного окна, кэшируя на WINDOW_CONTEXT_CACHE_SEC.

    Пример: "README.md - Джарвис - Visual Studio Code"
    При недоступности kdotool возвращает "неизвестно".
    """
    global _cache, _cache_time

    if not WINDOW_CONTEXT_ENABLED:
        return UNKNOWN

    now = time.monotonic()
    if _cache is not None and now - _cache_time < WINDOW_CONTEXT_CACHE_SEC:
        return _cache

    name = _query(["getactivewindow", "getwindowname"]) or _query(
        ["getactivewindow", "getwindowclassname"]
    ) or UNKNOWN

    _cache = name
    _cache_time = now
    return name


def get_window_context() -> str | None:
    """Строка контекста для системного промпта LLM, либо None."""
    window = get_active_window()
    if window == UNKNOWN:
        return None
    return f"Пользователь сейчас работает в окне: {window}"


def _query(args: list[str]) -> str | None:
    """Один вызов kdotool. None при любой ошибке."""
    try:
        result = subprocess.run(
            [KDTOOL_PATH] + args,
            capture_output=True,
            text=True,
            timeout=WINDOW_CONTEXT_TIMEOUT,
        )
    except FileNotFoundError:
        logger.warning("%s не найден в PATH", KDTOOL_PATH)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("kdotool %s: timeout %.1fs", args, WINDOW_CONTEXT_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("kdotool %s: %s", args, e)
        return None

    return result.stdout.strip() or None
