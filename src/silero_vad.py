"""Silero VAD — дешёвый детектор речи (ONNX, ~0.3% CPU).

Сессия onnxruntime одна на всех потребителей (run() потокобезопасен),
но состояние LSTM (h/c) у каждого экземпляра своё.
"""

from typing import Any

import numpy as np

from config import SAMPLE_RATE, VAD_MODEL
from src.logger import get_logger

logger = get_logger("vad")

_session: Any | None = None


def get_vad_session() -> Any:
    """Ленивый singleton сессии ONNX."""
    global _session
    if _session is None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            if exc.name != "onnxruntime":
                raise
            raise RuntimeError(
                "Для запуска Silero VAD установите зависимость onnxruntime"
            ) from exc
        logger.info("Загружаю Silero VAD...")
        _session = ort.InferenceSession(
            VAD_MODEL, providers=["CPUExecutionProvider"]
        )
        logger.info("Silero VAD загружен")
    return _session


class SileroVAD:
    """Одна линия VAD: своя LSTM-память, общая ONNX-сессия."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.session = get_vad_session()
        self.sample_rate = sample_rate
        self.reset()

    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, frame_int16: np.ndarray) -> float:
        """Вероятность речи в кадре int16 (для Silero нужен 32 мс при 16 kHz)."""
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
