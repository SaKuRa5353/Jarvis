"""Vosk Speech-to-Text (полностью офлайн)."""

import json

import numpy as np
from vosk import KaldiRecognizer, Model

from config import SAMPLE_RATE

from src.logger import get_logger
from src.vosk_model import get_vosk_model

logger = get_logger("stt")


class VoskSTT:
    """Распознавание речи через Vosk."""

    def __init__(self, model: Model | None = None, sample_rate: int = SAMPLE_RATE):
        self.model = model or get_vosk_model()
        self.sample_rate = sample_rate

    def recognize(self, audio_int16: np.ndarray) -> str | None:
        """Распознаёт текст из аудио (int16 PCM моно)."""
        if len(audio_int16) == 0:
            return None

        logger.info("🧠 Распознаю голос...")

        rec = KaldiRecognizer(self.model, self.sample_rate)

        # Разбиваем на чанки 4000 байт (250 мс при 16 kHz)
        data = audio_int16.tobytes()
        chunk_size = 4000

        for i in range(0, len(data), chunk_size):
            rec.AcceptWaveform(data[i : i + chunk_size])

        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        if text:
            logger.info("📝 Ты: %s", text)
        else:
            logger.warning("❌ Не разобрал (пустой результат Vosk)")

        return text if text else None
