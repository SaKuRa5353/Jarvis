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
    "музыка": "https://music.yandex.ru/",
    "музыку": "https://music.yandex.ru/",
    "вк": "https://vk.com/feed",
    "vk": "https://vk.com/feed",
    # Vosk слышит «ВК» как «вэ ка» / «века»
    "вэ ка": "https://vk.com/feed",
    "века": "https://vk.com/feed",
    "дипсик": "https://chat.deepseek.com/",
    "дипсика": "https://chat.deepseek.com/",
    "дип сик": "https://chat.deepseek.com/",
    "дикси": "https://chat.deepseek.com/",
    "deepseek": "https://chat.deepseek.com/",
    "deep sick": "https://chat.deepseek.com/",
    "депсик": "https://chat.deepseek.com/",
}


# Vosk small-ru слышит «загугли» по-разному: «загугли», «за гугли»,
# «за гугле», «загули» — поэтому вариант с пробелом тоже здесь.
_SEARCH_PATTERNS = [
    r"найди в гугле\s+(.+)",
    r"найди в интернете\s+(.+)",
    r"(?:загугли|за\s?гуг(?:ли|ле|лить)|загули|погугли|загуглить)\s+(.+)",
    r"найди\s+(.+)в гугле",
    r"поищи\s+(.+)",
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


# Глаголы, после которых ключевое слово точно означает ссылку.
_OPEN_WORDS = ("открой", "открыть", "запусти", "включи", "иди на", "перейди")

# Ключи-омонимы («века» = возраст слова «век») — только с глаголом.
_NEED_VERB = {"века", "вэ ка"}


def _try_quick_link(text: str):
    for keyword, url in QUICK_LINKS.items():
        if keyword not in text:
            continue
        # «открой ютуб» — явная команда;
        # просто «ютуб» (1-2 слова) — пользователь имел в виду то же самое.
        has_open_verb = any(w in text for w in _OPEN_WORDS)
        is_bare = len(text.split()) <= 2 and keyword not in _NEED_VERB
        if has_open_verb or is_bare:
            return f"Открываю {keyword}", lambda u=url: open_url(u), "opening"
    return None


def try_handle(text: str):
    # ── Быстрые ссылки ──
    result = _try_quick_link(text)
    if result is not None:
        return result

    # ── Поиск в гугле ──
    query = _extract_search_query(text)
    if query:
        url = f"https://www.google.com/search?q={quote(query)}"
        return f"Ищу {query}", lambda u=url: open_url(u), None

    return None
