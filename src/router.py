"""Роутер команд с fallback на LLM."""

import datetime
import subprocess
from pathlib import Path

from src.llm import JarvisLLM
from src.logger import get_logger

logger = get_logger("router")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

SCREENSHOT_DIR = Path.home() / "Pictures" / "Screenshots"


# ──────────────────────── СЛОВАРИ ────────────────────────

MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


# ──────────────────────── ФОРМАТИРОВАНИЕ ────────────────────────

def _plural(n: int, one: str, few: str, many: str) -> str:
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
    h, m = now.hour, now.minute
    hour_word = _plural(h, "час", "часа", "часов")
    minute_word = _plural(m, "минута", "минуты", "минут")
    return f"Сейчас {h} {hour_word} {m} {minute_word}"


def _format_date(now: datetime.datetime) -> str:
    month = MONTHS_GENITIVE[now.month - 1]
    return f"Сегодня {now.day} {month} {now.year} года"


# ──────────────────────── ДЕЙСТВИЯ ────────────────────────

def _screenshot_path() -> Path:
    """Возвращает путь для сохранения скриншота с таймстампом."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return SCREENSHOT_DIR / f"screenshot_{timestamp}.png"


def _make_screenshot():
    path = _screenshot_path()
    subprocess.Popen(
        ["spectacle", "-b", "-n", "-o", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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

    # ── Терминал ──
    if any(w in t for w in ["терминал", "консоль", "konsole"]):
        return (
            "Открываю терминал",
            lambda: subprocess.Popen(
                ["konsole"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ),
        )

    # ── Время ──
    if any(w in t for w in [
        "который час", "который сейчас час",
        "сколько времени", "сколько сейчас времени",
        "время сейчас", "сколько время",
    ]):
        return _format_time(datetime.datetime.now()), None

    # ── Дата ──
    if any(w in t for w in [
        "какое сегодня число", "какая дата", "какое число",
        "какое сегодня", "какой сегодня день", "какая сегодня дата",
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

    # ── Громкость: mute ──
    if any(w in t for w in ["выключи звук", "выключи зву", "без звука"]):
        return (
            "Выключаю звук",
            lambda: subprocess.Popen(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

    if any(w in t for w in ["включи звук", "верни звук"]):
        return (
            "Включаю звук",
            lambda: subprocess.Popen(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

    # ── Громкость: тише / громче ──
    if any(w in t for w in ["тише", "потише", "убавь звук", "сделай тише"]):
        return (
            "Делаю тише",
            lambda: subprocess.Popen(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

    if any(w in t for w in ["громче", "погромче", "прибавь звук", "сделай громче"]):
        return (
            "Делаю громче",
            lambda: subprocess.Popen(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

    # ── Скриншот ──
    if any(w in t for w in ["скриншот", "снимок экрана", "скрин"]):
        return (
            "Делаю скриншот",
            _make_screenshot,
        )

    # ── Fallback: LLM ──
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None