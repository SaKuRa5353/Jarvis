"""LLM tools: function calling для qwen2.5:3b-instruct.

Принцип (как в Irene): модель НИЧЕГО не выполняет сама — она только
выбирает функцию и аргументы. Выполнение на стороне Python, строго по
белым спискам. qwen2.5:3b иногда игнорирует enum (например
«открой языка» → app="языка") — валидация обязательна.

execute_tool() возвращает (answer_text, phrase_key) либо None, и
запускает действие сразу (fire-and-forget subprocess). Разговорные
ответы модель даёт обычным текстом (без tool) — их можно стримить.
"""

import json
from urllib.parse import quote

from src.actions import open_url, run
from src.logger import get_logger

logger = get_logger("llm_tools")

# ── Белые списки: canonical name -> (действие, озвучка, фраза) ──

_SITES = {
    "vk": ("https://vk.com/feed", "вконтакт"),
    "youtube": ("https://www.youtube.com/", "ютуб"),
    "deepseek": ("https://chat.deepseek.com/", "дипсик"),
    "avito": ("https://www.avito.ru/", "авито"),
    "github": ("https://github.com/", "гитхаб"),
    "music": ("https://music.yandex.ru/", "музыку"),
    "schedule": ("https://petrsu.ru/schedule/term?group=22404", "расписание"),
    "mail": ("https://mail.google.com/mail/u/0/#inbox", "почту"),
    "moodle": ("https://moodle2.petrsu.ru/", "мудл"),
}

_APPS = {
    "проводник": (["dolphin"], "проводник"),
    "vscode": (["code"], "редактор кода"),
    "браузер": (["brave"], "браузер"),
    "терминал": (["konsole"], "терминал"),
    "telegram": (["Telegram"], "телеграм"),
}

_VOLUME = {
    "up": (["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], "Делаю громче", "volume_up"),
    "down": (["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], "Делаю тише", "volume_down"),
    "mute": (["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], "Выключаю звук", "volume_mute"),
    "unmute": (["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], "Включаю звук", "volume_unmute"),
}

# ── Схемы для OpenAI-compatible API (Ollama) ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_site",
            "description": "Открыть сайт в браузере: вк, ютуб, дипсик, авито, гитхаб, музыка, расписание, почта, мудл.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "enum": list(_SITES.keys()),
                        "description": "Какой сайт открыть",
                    },
                },
                "required": ["site"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Запустить приложение на компьютере.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "enum": list(_APPS.keys()),
                        "description": "Какое приложение запустить",
                    },
                },
                "required": ["app"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Искать что-то в Google, когда просят найти или загуглить.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Что искать (только предмет поиска, без глаголов)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_action",
            "description": "Действие с системой: скриншот, блокировка экрана, громкость.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["screenshot", "lock", *_VOLUME.keys()],
                        "description": "Какое действие выполнить",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


def execute_tool(name: str, raw_args) -> tuple[str, str | None] | None:
    """Выполняет выбранный моделью tool. None — аргументы невалидны."""
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            logger.warning("[tool] битый JSON: %s", raw_args)
            return None
    else:
        args = raw_args or {}

    if name == "open_site":
        site = str(args.get("site", "")).lower().strip()
        if site not in _SITES:
            return None
        url, keyword = _SITES[site]
        open_url(url)
        return f"Открываю {keyword}", "opening"

    if name == "open_app":
        app = str(args.get("app", "")).lower().strip()
        if app not in _APPS:
            return None
        cmd, keyword = _APPS[app]
        run(cmd)
        return f"Открываю {keyword}", "opening"

    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if len(query) < 2:
            return None
        open_url(f"https://www.google.com/search?q={quote(query)}")
        return f"Ищу {query}", None

    if name == "system_action":
        action = str(args.get("action", "")).lower().strip()
        if action == "screenshot":
            # Ленивый импорт: src.commands.system тянет commands/__init__,
            # который импортирует llm — круг. Здесь импорт легален.
            from src.commands.system import _make_screenshot
            _make_screenshot()
            return "Делаю скриншот", "screenshot"
        if action == "lock":
            run(["loginctl", "lock-session"])
            return "Блокирую экран", "lock_screen"
        if action in _VOLUME:
            cmd, answer, phrase = _VOLUME[action]
            run(cmd)
            return answer, phrase
        return None

    logger.warning("[tool] неизвестный tool: %s", name)
    return None
