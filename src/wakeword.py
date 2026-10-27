"""Wakeword 'Джарвис' через Vosk с VAD-гейтом.

В тишине крутится только дешёвый Silero VAD (~0.3% CPU). Тяжёлый
KaldiRecognizer включается на речь и получает pre-roll (~0.4 сек
предыдущих кадров), чтобы первое слово не обрезалось.
"""

import time
from collections import deque
from collections.abc import Callable, Iterator

from vosk import KaldiRecognizer, Model

from config import (
    SAMPLE_RATE,
    VAD_FRAME,
    VAD_MIN_SILENCE_SEC,
    VAD_SPEECH_THRESHOLD,
    WAKE_WORDS,
    WW_MAX_UTTERANCE_SEC,
    WW_PREROLL_SEC,
)

from src.audio import Subscriber
from src.logger import get_logger
from src.silero_vad import SileroVAD
from src.vosk_model import get_vosk_model, text_of

logger = get_logger("wakeword")


class WakewordDetector:
    """Слушает подписку микрофона и ловит обращение 'Джарвис'."""

    def __init__(
        self,
        sub: Subscriber,
        model: Model | None = None,
        wake_words: tuple[str, ...] = WAKE_WORDS,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.sub = sub
        self.model = model or get_vosk_model()
        self.wake_words = wake_words
        self.sample_rate = sample_rate
        self.vad = SileroVAD(sample_rate)

    def is_match(self, text: str) -> bool:
        return any(word in text for word in self.wake_words)

    def _new_recognizer(self) -> KaldiRecognizer:
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        return rec

    # ──────────────────── VAD-ГЕЙТ ────────────────────

    def phrases(
        self,
        should_stop: Callable[[], bool] | None = None,
        deadline: float | None = None,
    ) -> Iterator[str]:
        """Генератор текста каждой произнесённой фразы.

        Тишину считает только Silero VAD. Как только prob > порога —
        включаем KaldiRecognizer, скармливаем ему pre-roll и дальше все
        кадры. После VAD_MIN_SILENCE_SEC тишины — FinalResult и yield.
        """
        frame = VAD_FRAME
        sps = self.sample_rate / frame  # кадров в секунду
        pre_roll_max = max(1, round(WW_PREROLL_SEC * sps))
        min_silence = max(1, round(VAD_MIN_SILENCE_SEC * sps))
        max_frames = round(WW_MAX_UTTERANCE_SEC * sps)

        self.vad.reset()
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_max)
        rec: KaldiRecognizer | None = None
        silence = 0
        frames = 0

        while True:
            if should_stop is not None and should_stop():
                return
            if deadline is not None and time.monotonic() > deadline:
                return

            self.sub.check()
            try:
                chunk = self.sub.read(frame, timeout=0.5)
            except TimeoutError:
                continue

            audio = chunk.tobytes()
            prob = self.vad.prob(chunk)

            if rec is None:
                # ── Спи режим: только VAD ──
                pre_roll.append(audio)
                if prob > VAD_SPEECH_THRESHOLD:
                    rec = self._new_recognizer()
                    for old in pre_roll:
                        rec.AcceptWaveform(old)
                    pre_roll.clear()
                    silence = 0
                    frames = 0
                continue

            # ── Бодрствующий режим: VAD + Vosk ──
            rec.AcceptWaveform(audio)
            frames += 1
            if prob > VAD_SPEECH_THRESHOLD:
                silence = 0
            else:
                silence += 1

            if silence >= min_silence or frames >= max_frames:
                text = text_of(rec.FinalResult())
                rec = None
                if text:
                    yield text

    # ──────────────────── ПУБЛИЧНОЕ ────────────────────

    def wait(self) -> str:
        """Блокирующе ждёт wakeword. Возвращает распознанную фразу целиком."""
        logger.info("💤 Жду 'Джарвис'...")
        self.sub.drain()

        for text in self.phrases():
            logger.info("Vosk услышал: %s", text)
            if self.is_match(text):
                logger.info("🎯 Услышал wakeword: %s", text)
                return text

        return ""  # недостижимо: без should_stop/deadline генератор вечен

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
        self.sub.drain()
        deadline = time.monotonic() + timeout

        for text in self.phrases(should_stop=is_done, deadline=deadline):
            if self.is_match(text):
                logger.info("🛑 Прерывание по wakeword: %s", text)
                return True

        return False
