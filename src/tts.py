"""Piper TTS (локальный) + mpv с возможностью прерывания."""

import subprocess
import threading
import wave

from piper import PiperVoice

from config import PIPER_MODEL, TTS_WAV

from src.logger import get_logger

logger = get_logger("tts")


_voice = None
_current_mpv: subprocess.Popen | None = None
_mpv_lock = threading.Lock()


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        logger.info("[TTS] Загружаю Piper...")
        _voice = PiperVoice.load(PIPER_MODEL)
    return _voice


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


def speak(text: str):
    """Блокирующий синтез + воспроизведение. Прерывается через stop_speaking()."""
    global _current_mpv
    logger.info("🔊 Джарвис: %s", text)
    try:
        voice = _get_voice()
        chunks = list(voice.synthesize(text))
        if not chunks:
            logger.warning("[TTS] нет аудио")
            return

        first = chunks[0]
        with wave.open(str(TTS_WAV), "wb") as wav_file:
            wav_file.setnchannels(first.sample_channels)
            wav_file.setsampwidth(first.sample_width)
            wav_file.setframerate(first.sample_rate)
            wav_file.writeframes(b"".join(c.audio_int16_bytes for c in chunks))

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