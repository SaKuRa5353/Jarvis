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
}


def get_phrase_path(key: str) -> Path | None:
    """Возвращает путь к WAV-файлу или None."""
    entry = PHRASES.get(key)
    if entry is None:
        return None
    return PHRASES_DIR / entry[0]
