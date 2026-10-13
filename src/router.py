"""Роутер команд с fallback на LLM."""

import datetime
import subprocess

from src.llm import JarvisLLM
from src.logger import get_logger

logger = get_logger("router")


# ──────────────────────── СЛОВАРИ ────────────────────────

MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


# ──────────────────────── ФОРМАТИРОВАНИЕ ────────────────────────

def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русская плюрализация: 1 час, 2 часа, 5 часов."""
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 19:
        return many
    n_abs %= 10
    if n_abs == 1:
        return one
    if 2 <= n_abs <= 4:
        return few
    return many


def _format_time(now: datetime.datetime) -> str:
    h = now.hour
    m = now.minute
    hour_word = _plural(h, "час", "часа", "часов")
    minute_word = _plural(m, "минута", "минуты", "минут")
    return f"Сейчас {h} {hour_word} {m} {minute_word}"


def _format_date(now: datetime.datetime) -> str:
    month = MONTHS_GENITIVE[now.month - 1]
    return f"Сегодня {now.day} {month} {now.year} года"


# ──────────────────────── РОУТЕР ────────────────────────

def plan_command(text: str, llm: JarvisLLM):
    """Возвращает (response_text, action_callable) или (None, None). Ничего не выполняет."""
    t = text.lower()
    t = t.replace("джарвис", "").strip()

    # ── Проводник ──
    if any(w in t for w in ["проводник", "файлы", "документы", "папк"]):
        return (
            "Открываю проводник",
            lambda: subprocess.Popen(
                ["dolphin"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    # ── VSCode ──
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

    # ── Браузер ──
    if any(w in t for w in ["браузер", "brave", "брейв"]):
        return (
            "Открываю браузер",
            lambda: subprocess.Popen(
                ["brave"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    # ── Время ──
    if any(w in t for w in ["который час", "сколько времени", "время сейчас", "сколько время", "текущее время"]):
        return _format_time(datetime.datetime.now()), None

    # ── Дата ──
    if any(w in t for w in [
        "какое сегодня число", "какая дата", "какое число",
        "какое сегодня", "какой сегодня день", "какая сегодня дата"
    ]):
        return _format_date(datetime.datetime.now()), None

    # ── Блокировка экрана ──
    if any(w in t for w in ["заблокируй", "блокировка экрана", "заблокировать экран"]):
        return (
            "Блокирую экран",
            lambda: subprocess.Popen(
                ["loginctl", "lock-session"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

    # ── Fallback: LLM ──
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None