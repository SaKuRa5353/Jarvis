"""Команды запуска приложений."""

from src.commands.utils import run


def try_handle(text: str):
    # ── Проводник ──
    if any(w in text for w in ["проводник", "файлы", "документы", "папк"]):
        return (
            "Открываю проводник",
            lambda: run(["dolphin"]),
            "opening",
        )

    # ── VSCode ──
    if any(w in text for w in [
        "vscode", "вс код", "вс-код", "vs code", "visual studio",
        "вс студия", "код",
    ]):
        return (
            "Запускаю редактор кода",
            lambda: run(["code"]),
            "opening_app",
        )

    # ── Браузер ──
    if any(w in text for w in ["браузер", "brave", "брейв"]):
        return (
            "Открываю браузер",
            lambda: run(["brave"]),
            "opening",
        )

    # ── Терминал ──
    if any(w in text for w in ["терминал", "консоль", "konsole"]):
        return (
            "Открываю терминал",
            lambda: run(["konsole"]),
            "opening",
        )

    # ── Telegram ──
    if any(w in text for w in ["телеграм", "telegram", "тг", "телега"]):
        return (
            "Открываю Telegram",
            lambda: run(["Telegram"]),
            "opening",
        )

    return None
