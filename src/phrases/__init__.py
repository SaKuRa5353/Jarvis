"""Словарь готовых фраз.

Каждый ключ — это «категория» ответа.
При вызове speak(text, phrase=...) сначала проверяется словарь:
если ключ есть — играет mpv (0.1 сек), иначе — Silero TTS (1.5 сек).
"""

from pathlib import Path

# WAV-файлы лежат рядом с этим __init__.py — в src/phrases/
PHRASES_DIR = Path(__file__).resolve().parent

# (ключ, путь к WAV, текст для логирования)
PHRASES = {
    # ── Приветствия ──
    "welcome": ("welcome.wav", "К вашим услугам"),

    # ── Выполнение команд ──
    "opening": ("opening.wav", "Открываю"),
    "opening_app": ("opening_app.wav", "Запускаю"),
    "closing": ("closing.wav", "Закрываю"),
    "volume_up": ("volume_up.wav", "Делаю громче"),
    "volume_down": ("volume_down.wav", "Делаю тише"),
    "volume_mute": ("volume_mute.wav", "Выключаю звук"),
    "volume_unmute": ("volume_unmute.wav", "Включаю звук"),
    "screenshot": ("screenshot.wav", "Делаю скриншот"),
    "lock_screen": ("lock_screen.wav", "Блокирую экран"),

    # ── Завершение ──
    "done": ("done.wav", "Готово"),

    # ── Ошибки ──
    "not_listened": ("not_listened.wav", "Не расслышал"),
    "not_understood": ("not_understood.wav", "Не понимаю"),
    "error": ("error.wav", "Ошибка"),

    # ── LLM ──
    "llm_thinking": ("llm_thinking.wav", "Думаю"),

    # ── Earcons: короткие «динь» вместо речевых фраз ──
    # Включаются CONFIRM_SOUND="earcon" в config.py (или JARVIS_CONFIRM_SOUND).
    # Создать файлы: python tests/generate_earcons.py
    "earcon_welcome": ("earcon_welcome.wav", "[динь]"),
    "earcon_thinking": ("earcon_thinking.wav", "[динь-думает]"),
    "earcon_done": ("earcon_done.wav", "[динь-готово]"),
}

# Какие фразы считаются «подтверждением» и заменяются earcon.
# not_listened / error / volume_* — содержательные, их не трогаем.
CONFIRM_KEYS = {
    "welcome": "earcon_welcome",
    "llm_thinking": "earcon_thinking",
    "done": "earcon_done",
}


def get_phrase_path(key: str) -> Path | None:
    """Возвращает путь к WAV-файлу или None."""
    entry = PHRASES.get(key)
    if entry is None:
        return None
    return PHRASES_DIR / entry[0]


def resolve_phrase_key(key: str) -> str | None:
    """Ключ фразы с учётом CONFIRM_SOUND; None — если звук выключен.

    Если earcon-файла ещё нет (не сгенерирован), возвращает исходный ключ:
    ассистент не онемеет из-за отсутствующего WAV.
    """
    from config import CONFIRM_SOUND  # локально: config не должен зависеть от нас

    mode = (CONFIRM_SOUND or "voice").lower()
    if mode == "off" and key in CONFIRM_KEYS:
        return None
    if mode == "earcon" and key in CONFIRM_KEYS:
        earcon = CONFIRM_KEYS[key]
        if (PHRASES_DIR / PHRASES[earcon][0]).exists():
            return earcon
    return key
