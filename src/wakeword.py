"""Wakeword 'Джарвис' через Vosk (keyword spotting)."""

import json

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from config import DEVICE, SAMPLE_RATE, VOSK_MODEL

from src.logger import get_logger

logger = get_logger("wakeword")


# ──────────────────────── МОДЕЛЬ ────────────────────────

logger.info("Загружаю Vosk модель...")
_model = Model(VOSK_MODEL)
logger.info("Vosk модель загружена")


# ──────────────────────── WAKE WORD ────────────────────────

_WAKE_WORDS = (
    "джарвис",
    "jarvis",
)


def wait_for_wakeword():
    """Блокирующе ждёт, пока не услышит 'Джарвис'."""
    logger.info("💤 Жду 'Джарвис'...")

    rec = KaldiRecognizer(_model, SAMPLE_RATE)
    rec.SetWords(True)

    block_size = 4000   # 250 мс при 16 kHz

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=block_size,
        device=DEVICE,
        dtype="int16",
        channels=1,
    ) as stream:
        while True:
            data, _ = stream.read(block_size)
            if not rec.AcceptWaveform(bytes(data)):
                continue

            # Полный результат — только по нему принимаем решение
            result = json.loads(rec.Result())
            text = result.get("text", "").lower().strip()

            if text:
                logger.info("Vosk услышал: %s", text)

            if any(word in text for word in _WAKE_WORDS):
                logger.info("🎯 Услышал wakeword: %s", text)
                return

def check_wakeword_or_done(is_done, timeout: float = 30.0) -> bool:
    """
    Слушает микрофон до timeout секунд.

    Возвращает:
        True  — услышал wakeword (нужно прервать операцию)
        False — is_done() стал True (операция завершена сама, не прерываем)
    """
    import time as _time

    rec = KaldiRecognizer(_model, SAMPLE_RATE)
    block_size = 4000
    deadline = _time.time() + timeout

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=block_size,
        device=DEVICE,
        dtype="int16",
        channels=1,
    ) as stream:
        while _time.time() < deadline:
            # Проверяем, не завершилась ли операция (LLM/TTS)
            if is_done():
                return False

            data, _ = stream.read(block_size)
            if not rec.AcceptWaveform(bytes(data)):
                continue

            result = json.loads(rec.Result())
            text = result.get("text", "").lower().strip()
            if any(word in text for word in _WAKE_WORDS):
                logger.info("🛑 Прерывание по wakeword: %s", text)
                return True

    return False