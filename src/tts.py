"""Silero TTS (локальный) + mpv с возможностью прерывания."""

import subprocess
import threading
import re
from pathlib import Path
from silero_tts.silero_tts import SileroTTS

from config import SILERO_MODEL, SILERO_SAMPLE_RATE, SILERO_SPEAKER, TTS_WAV

from src.logger import get_logger
from src.phrases import get_phrase_path, PHRASES

logger = get_logger("tts")


# ──────────────────────── СОСТОЯНИЕ ────────────────────────

_tts = None
_current_mpv: subprocess.Popen | None = None
_mpv_lock = threading.Lock()


def _get_voice() -> SileroTTS:
    global _tts
    if _tts is None:
        logger.info("[TTS] Загружаю Silero (%s)...", SILERO_SPEAKER)
        _tts = SileroTTS(
            model_id=SILERO_MODEL,
            language="ru",
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_SAMPLE_RATE,
            device="cpu",
            put_accent=True,
            put_yo=True,
        )
        logger.info("[TTS] Silero загружен")
    return _tts


def stop_speaking():
    """Прерывает текущее воспроизведение mpv, если оно идёт."""
    global _current_mpv
    with _mpv_lock:
        if _current_mpv is not None and _current_mpv.poll() is None:
            logger.info("[TTS] Прерываю воспроизведение")
            _current_mpv.terminate()
            try:
                _current_mpv.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _current_mpv.kill()
        _current_mpv = None


# ──────────────────────── ОЧИСТКА ТЕКСТА ────────────────────────

def _fix_text(text: str) -> str:
    """Исправляет ударения и чистит символы для Silero."""
    # Ударения (Silero понимает + перед ударной гласной)
    stress_fixes = {
        "Джарвис": "Дж+арвис",
        "джарвис": "Дж+арвис",
    }
    for wrong, right in stress_fixes.items():
        text = text.replace(wrong, right)

    # Чистка символов, которые Silero читает плохо
    symbol_fixes = {
        "/": " ",           # косая черта → пробел
        "_": " ",           # подчёркивание → пробел
        "~": " ",           # тильда → пробел
        "%": " процентов ",
        "&": " и ",
        "@": " собака ",
        "#": " номер ",
        "№": " номер ",
        "°C": " градусов Цельсия",
        "°": " градусов ",
    }
    for wrong, right in symbol_fixes.items():
        text = text.replace(wrong, right)
    # Убираем двойные пробелы после замен
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────── ФРАЗЫ (0 сек синтеза) ────────────────────────

def speak_phrase(key: str) -> bool:
    """Проигрывает готовую фразу из src/phrases/. True если файл найден."""
    path = get_phrase_path(key)
    if path is None or not path.exists():
        return False

    text = PHRASES.get(key, (None, "?"))[1]
    logger.info("🔊 Джарвис (фраза): %s", text)

    global _current_mpv
    with _mpv_lock:
        _current_mpv = subprocess.Popen(
            ["mpv", "--no-terminal", "--really-quiet", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mpv = _current_mpv

    mpv.wait()

    with _mpv_lock:
        if _current_mpv is mpv:
            _current_mpv = None

    return True


# ──────────────────────── СИНТЕЗ + ВОСПРОИЗВЕДЕНИЕ ────────────────────────

def speak(text: str, phrase_key: str | None = None):
    """Блокирующий синтез + воспроизведение. Прерывается через stop_speaking().

    Если phrase_key задан и WAV-файл существует — играет mpv за 0 сек.
    Иначе — Silero TTS за ~1.5 сек.
    """
    if phrase_key and speak_phrase(phrase_key):
        return

    # Fallback — синтез через Silero
    global _current_mpv
    text = _fix_text(text)
    logger.info("🔊 Джарвис: %s", text)
    try:
        tts = _get_voice()
        tts.tts(text, str(TTS_WAV))

        with _mpv_lock:
            _current_mpv = subprocess.Popen(
                ["mpv", "--no-terminal", "--really-quiet", str(TTS_WAV)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mpv = _current_mpv

        mpv.wait()

        with _mpv_lock:
            if _current_mpv is mpv:
                _current_mpv = None
    except Exception as e:
        logger.error("[TTS error] %s", e)