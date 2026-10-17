"""Silero VAD + запись команды до тишины."""

import numpy as np
import onnxruntime as ort

from config import (
    SAMPLE_RATE,
    VAD_FRAME,
    VAD_MAX_RECORD_SEC,
    VAD_MIN_SILENCE_SEC,
    VAD_MIN_SPEECH_SEC,
    VAD_MODEL,
    VAD_SPEECH_THRESHOLD,
)

from src.audio import Microphone
from src.logger import get_logger

logger = get_logger("vad")


# ──────────────────────── МОДЕЛЬ ────────────────────────

logger.info("Загружаю Silero VAD...")
_vad_session = ort.InferenceSession(VAD_MODEL, providers=["CPUExecutionProvider"])
logger.info("Silero VAD загружен")


class SpeechRecorder:
    """Режет речь из общего потока микрофона по Silero VAD."""

    def __init__(
        self,
        mic: Microphone,
        session: ort.InferenceSession | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame: int = VAD_FRAME,
    ):
        self.mic = mic
        self.session = session or _vad_session
        self.sample_rate = sample_rate
        self.frame = frame
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _prob(self, frame_int16: np.ndarray) -> float:
        x = frame_int16.astype(np.float32) / 32768.0
        x = x.reshape(1, -1)
        out, self._h, self._c = self.session.run(
            None,
            {
                "input": x,
                "sr": np.array(self.sample_rate, dtype=np.int64),
                "h": self._h,
                "c": self._c,
            },
        )
        return float(out[0][0])

    def record(self) -> np.ndarray:
        """Читает из общего микрофона, пока не наступит тишина."""
        self.mic.drain()
        self._reset()

        buffer: list[np.ndarray] = []
        speech_started = False
        silence_frames = 0
        speech_frames = 0

        min_silence_frames = int(VAD_MIN_SILENCE_SEC * self.sample_rate / self.frame)
        max_frames = int(VAD_MAX_RECORD_SEC * self.sample_rate / self.frame)
        min_speech_frames = int(VAD_MIN_SPEECH_SEC * self.sample_rate / self.frame)

        logger.info("🎤 Слушаю команду...")

        for _ in range(max_frames):
            frame = self.mic.read(self.frame)
            prob = self._prob(frame)

            if prob > VAD_SPEECH_THRESHOLD:
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
            return np.array([], dtype=np.int16)

        audio = np.concatenate(buffer)
        logger.info("✅ Записано %.1f сек", len(audio) / self.sample_rate)
        return audio
