"""Веб-команды: быстрые ссылки, поиск в гугле."""

import re
from urllib.parse import quote

from src.commands.utils import open_url


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


_SEARCH_PATTERNS = [
    r"найди в гугле\s+(.+)",
    r"загугли\s+(.+)",
    r"найди\s+(.+)в гугле",
    r"поищи\s+(.+)",
    r"погугли\s+(.+)",
]


def _extract_search_query(text: str) -> str | None:
    for pattern in _SEARCH_PATTERNS:
        match = re.search(pattern, text)
        if match:
            query = match.group(1).strip()
            query = re.sub(r"\s+в (гугле|интернете|сети)$", "", query)
            if query:
                return query
    return None


def try_handle(text: str):
    # ── Быстрые ссылки ──
    if "открой" in text or "открыть" in text:
        for keyword, url in QUICK_LINKS.items():
            if keyword in text:
                return f"Открываю {keyword}", lambda u=url: open_url(u)

    # ── Поиск в гугле ──
    query = _extract_search_query(text)
    if query:
        url = f"https://www.google.com/search?q={quote(query)}"
        return f"Ищу {query}", lambda u=url: open_url(u)

    return None
