"""Piper TTS (локальный) + mpv."""
from src.logger import get_logger

logger = get_logger("tts")
import subprocess
import wave

from piper import PiperVoice

from config import PIPER_MODEL, TTS_WAV


_voice = None


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        logger.info("[TTS] Загружаю Piper...")
        _voice = PiperVoice.load(PIPER_MODEL)
    return _voice


def speak(text: str):
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

        subprocess.run(
            ["mpv", "--no-terminal", "--really-quiet", str(TTS_WAV)],
            check=False,
        )
    except Exception as e:
        logger.error("[TTS error] %s", e)
