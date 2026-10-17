"""Системные команды: время, дата, блокировка, громкость, скриншот."""

import datetime
from pathlib import Path

from src.commands.utils import plural, run


SCREENSHOT_DIR = Path.home() / "Pictures" / "Screenshots"

MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _format_time() -> str:
    now = datetime.datetime.now()
    h, m = now.hour, now.minute
    hour_word = plural(h, "час", "часа", "часов")
    minute_word = plural(m, "минута", "минуты", "минут")
    return f"Сейчас {h} {hour_word} {m} {minute_word}"


def _format_date() -> str:
    now = datetime.datetime.now()
    month = MONTHS_GENITIVE[now.month - 1]
    return f"Сегодня {now.day} {month} {now.year} года"


def _make_screenshot():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
    run(["spectacle", "-b", "-n", "-o", str(path)])


def try_handle(text: str):
    # ── Время ──
    if any(w in text for w in [
        "который час", "который сейчас час",
        "сколько времени", "сколько сейчас времени",
        "время сейчас", "сколько время",
    ]):
        return _format_time(), None, None

    # ── Дата ──
    if any(w in text for w in [
        "какое сегодня число", "какая дата", "какое число",
        "какое сегодня", "какой сегодня день", "какая сегодня дата",
    ]):
        return _format_date(), None, None

    # ── Блокировка ──
    if any(w in text for w in ["заблокируй", "блокировка экрана", "заблокировать экран"]):
        return (
            "Блокирую экран",
            lambda: run(["loginctl", "lock-session"]),
            "lock_screen",
        )

    # ── Mute ──
    if any(w in text for w in ["выключи звук", "выключи зву", "без звука"]):
        return (
            "Выключаю звук",
            lambda: run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"]),
            "volume_mute",
        )

    if any(w in text for w in ["включи звук", "верни звук"]):
        return (
            "Включаю звук",
            lambda: run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"]),
            "volume_unmute",
        )

    # ── Громкость ──
    # «выше/ниже» сами по себе опасны («температура выше вчерашней»),
    # поэтому они работают только рядом со словом про звук/громкость.
    volume_ctx = any(w in text for w in ["громк", "звук", "громкость"])
    if any(w in text for w in ["тише", "потише", "убавь"]) or (
        volume_ctx and "ниже" in text
    ):
        return (
            "Делаю тише",
            lambda: run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"]),
            "volume_down",
        )

    if any(w in text for w in ["громче", "погромче", "прибавь", "добавь"]) or (
        volume_ctx and any(w in text for w in ["выше", "больше"])
    ):
        return (
            "Делаю громче",
            lambda: run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"]),
            "volume_up",
        )

    # ── Скриншот ──
    if any(w in text for w in ["скриншот", "снимок экрана", "скрин"]):
        return "Делаю скриншот", _make_screenshot, "screenshot"

    return None
