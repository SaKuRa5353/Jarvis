"""Роутер команд с fallback на LLM."""

import datetime
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from src.llm import JarvisLLM
from src.logger import get_logger

logger = get_logger("router")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

SCREENSHOT_DIR = Path.home() / "Pictures" / "Screenshots"

QUICK_LINKS = {
    "расписание": "https://petrsu.ru/schedule/term?group=22404",
    "почта": "https://mail.google.com/mail/u/0/#inbox",
    "почту": "https://mail.google.com/mail/u/0/#inbox",
    "youtube": "https://www.youtube.com/",
    "ютуб": "https://www.youtube.com/",
    "вк": "https://vk.com/feed",
    "vk": "https://vk.com/feed",
    "дипсик": "https://chat.deepseek.com/",
    "дипсика": "https://chat.deepseek.com/",
    "дип сик": "https://chat.deepseek.com/",
    "дикси": "https://chat.deepseek.com/",
    "deepseek": "https://chat.deepseek.com/",
    "deep sick": "https://chat.deepseek.com/",
    "депсик": "https://chat.deepseek.com/",
}


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


def _open_url(url: str):
    subprocess.Popen(
        ["brave", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _extract_search_query(text: str) -> str | None:
    """Извлекает поисковый запрос из фразы."""
    patterns = [
        r"найди в гугле\s+(.+)",
        r"загугли\s+(.+)",
        r"найди\s+(.+)в гугле",
        r"поищи\s+(.+)",
        r"погугли\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            query = match.group(1).strip()
            query = re.sub(r"\s+в (гугле|интернете|сети)$", "", query)
            if query:
                return query
    return None


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
        return "Делаю скриншот", _make_screenshot

    # ── Telegram (десктопное приложение) ──
    if any(w in t for w in ["телеграм", "telegram", "тг", "телега"]):
        return (
            "Открываю Telegram",
            lambda: subprocess.Popen(
                ["Telegram"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ),
        )

        # ── Быстрые ссылки ──
    if "открой" in t or "открыть" in t:
        for keyword, url in QUICK_LINKS.items():
            if keyword in t:
                response = f"Открываю"
                return response, lambda u=url: _open_url(u)

    # ── Поиск в гугле ──
    query = _extract_search_query(t)
    if query:
        url = f"https://www.google.com/search?q={quote(query)}"
        return None, lambda u=url: _open_url(u)

    # ── Fallback: LLM ──
    logger.info("🤔 Думаю (LLM)...")
    answer = llm.ask(text)
    if answer:
        return answer, None

    return None, None