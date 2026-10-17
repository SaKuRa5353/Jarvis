"""Wakeword 'Джарвис' через Vosk (keyword spotting)."""

import json
import time

from vosk import KaldiRecognizer, Model

from config import SAMPLE_RATE, WAKE_WORDS

from src.audio import Microphone
from src.logger import get_logger
from src.vosk_model import get_vosk_model

logger = get_logger("wakeword")


class WakewordDetector:
    """Слушает общий микрофон и ловит обращение 'Джарвис'."""

    def __init__(
        self,
        mic: Microphone,
        model: Model | None = None,
        wake_words: tuple[str, ...] = WAKE_WORDS,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.mic = mic
        self.model = model or get_vosk_model()
        self.wake_words = wake_words
        self.sample_rate = sample_rate
        self.block_size = mic.block_size

    def _match(self, text: str) -> bool:
        return any(word in text for word in self.wake_words)

    def _new_recognizer(self) -> KaldiRecognizer:
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        return rec

    def wait(self) -> str:
        """Блокирующе ждёт wakeword. Возвращает распознанную фразу целиком."""
        logger.info("💤 Жду 'Джарвис'...")
        self.mic.drain()
        rec = self._new_recognizer()

        while True:
            frame = self.mic.read(self.block_size)
            if not rec.AcceptWaveform(frame.tobytes()):
                continue

            text = _text_of(rec.Result())
            if not text:
                continue

            logger.info("Vosk услышал: %s", text)

            if self._match(text):
                logger.info("🎯 Услышал wakeword: %s", text)
                return text

    def strip_wakeword(self, text: str) -> str:
        """Убирает wakeword из фразы, возвращает хвост команды."""
        t = text.lower()
        for ww in self.wake_words:
            t = t.replace(ww, "").strip()
        return t

    def wait_or_done(self, is_done, timeout: float = 30.0) -> bool:
        """
        Слушает микрофон до timeout секунд.

        Возвращает:
            True  — услышал wakeword (нужно прервать операцию)
            False — is_done() стал True (операция завершилась сама)
        """
        self.mic.drain()
        rec = self._new_recognizer()
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if is_done():
                return False

            try:
                frame = self.mic.read(self.block_size, timeout=0.5)
            except TimeoutError:
                continue

            if not rec.AcceptWaveform(frame.tobytes()):
                continue

            text = _text_of(rec.Result())
            if text and self._match(text):
                logger.info("🛑 Прерывание по wakeword: %s", text)
                return True

        return False


def _text_of(raw_json: str) -> str:
    try:
        return json.loads(raw_json).get("text", "").lower().strip()
    except (ValueError, AttributeError):
        return ""
