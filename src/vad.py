"""Silero VAD + запись команды до тишины.

Речь распознаётся параллельно с записью: каждый блок аудио идёт и в VAD,
и в KaldiRecognizer. Когда VAD решает, что команда окончена, текст уже
готов — экономия ~1 секунды на отдельном проходе распознавания.
"""

from __future__ import annotations

import numpy as np
from vosk import KaldiRecognizer

import config
from config import (
    SAMPLE_RATE,
    VAD_FRAME,
    VAD_MAX_RECORD_SEC,
    VAD_MIN_SILENCE_SEC,
    VAD_MIN_SPEECH_SEC,
    VAD_SPEECH_THRESHOLD,
)
from src.audio import Subscriber
from src.logger import get_logger
from src.mathcore.viterbi import HMM
from src.metrics import timed
from src.silero_vad import SileroVAD
from src.vosk_model import get_vosk_model, text_of

logger = get_logger("vad")


class SpeechRecorder:
    """Режет речь из общего потока микрофона по Silero VAD.

    Одновременно кормит тем же аудио Vosk-распознаватель, поэтому по
    окончании записи текст уже готов.
    """

    def __init__(
        self,
        sub: Subscriber,
        sample_rate: int = SAMPLE_RATE,
        frame: int = VAD_FRAME,
    ):
        self.sub = sub
        self.sample_rate = sample_rate
        self.frame = frame
        self.hmm: HMM | None = None
        if config.VAD_DECODER not in ("threshold", "viterbi"):
            raise ValueError("VAD_DECODER должен быть threshold или viterbi")
        if config.VAD_DECODER == "viterbi":
            # Формат NPZ: A (2,2), pi (2,), emissions (2,2); 0 — тишина, 1 — речь.
            with np.load(config.VAD_HMM_MODEL, allow_pickle=False) as model:
                self.hmm = HMM(2, model["A"], model["pi"], model["emissions"].tolist())
        self.vad = SileroVAD(sample_rate)

    @timed("vad.record")
    def record(self) -> tuple[np.ndarray, str]:
        """Читает из общего микрофона, пока не наступит тишина.

        Возвращает (аудио, распознанный текст). При слишком короткой речи
        или ошибке — (пустой массив, "").
        """
        self.sub.drain()
        self.vad.reset()

        rec = KaldiRecognizer(get_vosk_model(), self.sample_rate)

        buffer: list[np.ndarray] = []
        speech_started = False
        silence_frames = 0
        speech_frames = 0
        probabilities: list[float] = []

        min_silence_frames = int(VAD_MIN_SILENCE_SEC * self.sample_rate / self.frame)
        max_frames = int(VAD_MAX_RECORD_SEC * self.sample_rate / self.frame)
        min_speech_frames = int(VAD_MIN_SPEECH_SEC * self.sample_rate / self.frame)

        logger.info("🎤 Слушаю команду...")

        for _ in range(max_frames):
            self.sub.check()
            frame = self.sub.read(self.frame)
            rec.AcceptWaveform(frame.tobytes())
            prob = self.vad.prob(frame)
            if self.hmm is not None:
                probabilities.append(prob)
                path, _ = self.hmm.viterbi(probabilities)
                # Последнее состояние оптимального пути заменяет только решение
                # о кадре; счётчики и критерии конца записи остаются прежними.
                prob = float(path[-1])

            if (prob == 1.0 if self.hmm is not None else prob > VAD_SPEECH_THRESHOLD):
                speech_started = True
                silence_frames = 0
                speech_frames += 1
                buffer.append(frame)
            elif speech_started:
                silence_frames += 1
                buffer.append(frame)
                if silence_frames >= min_silence_frames:
                    break

        if speech_frames < min_speech_frames:
            logger.warning("⚠️ Слишком короткая речь")
            return np.array([], dtype=np.int16), ""

        audio = np.concatenate(buffer)
        logger.info("✅ Записано %.1f сек", len(audio) / self.sample_rate)

        text = text_of(rec.FinalResult())
        if text:
            logger.info("📝 Ты: %s", text)
        else:
            logger.warning("❌ Не разобрал (пустой результат Vosk)")

        return audio, text
