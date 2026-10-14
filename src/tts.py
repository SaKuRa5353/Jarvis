"""Silero TTS (локальный) + mpv с возможностью прерывания."""

import subprocess
import threading

from silero_tts.silero_tts import SileroTTS

from src.logger import get_logger

logger = get_logger("tts")


# ──────────────────────── КОНСТАНТЫ ────────────────────────

SILERO_MODEL = "v5_ru"
SILERO_SPEAKER = "eugene"
SILERO_SAMPLE_RATE = 48000
TTS_WAV = "/tmp/jarvis_tts.wav"


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

    return text


# ──────────────────────── СИНТЕЗ + ВОСПРОИЗВЕДЕНИЕ ────────────────────────

def speak(text: str):
    """Блокирующий синтез + воспроизведение. Прерывается через stop_speaking()."""
    global _current_mpv
    text = _fix_text(text)
    logger.info("🔊 Джарвис: %s", text)
    try:
        tts = _get_voice()
        tts.tts(text, TTS_WAV)

        with _mpv_lock:
            _current_mpv = subprocess.Popen(
                ["mpv", "--no-terminal", "--really-quiet", TTS_WAV],
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