"""Silero VAD + запись до тишины."""
from src.logger import get_logger

logger = get_logger("vad")
import numpy as np
import onnxruntime as ort
import sounddevice as sd

from config import (
    DEVICE,
    SAMPLE_RATE,
    VAD_FRAME,
    VAD_MAX_RECORD_SEC,
    VAD_MIN_SILENCE_SEC,
    VAD_MIN_SPEECH_SEC,
    VAD_MODEL,
    VAD_SILENCE_THRESHOLD,
    VAD_SPEECH_THRESHOLD,
)


_session = ort.InferenceSession(VAD_MODEL, providers=["CPUExecutionProvider"])

_h = np.zeros((2, 1, 64), dtype=np.float32)
_c = np.zeros((2, 1, 64), dtype=np.float32)


def _reset():
    global _h, _c
    _h = np.zeros((2, 1, 64), dtype=np.float32)
    _c = np.zeros((2, 1, 64), dtype=np.float32)


def _prob(frame_int16: np.ndarray) -> float:
    global _h, _c
    x = frame_int16.astype(np.float32) / 32768.0
    x = x.reshape(1, -1)
    out, _h, _c = _session.run(
        None,
        {
            "input": x,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            "h": _h,
            "c": _c,
        },
    )
    return float(out[0][0])


def record_until_silence() -> np.ndarray:
    """Записывает аудио с микрофона, пока не наступит тишина."""
    _reset()

    buffer = []
    speech_started = False
    silence_frames = 0
    min_silence_frames = int(VAD_MIN_SILENCE_SEC * SAMPLE_RATE / VAD_FRAME)
    max_frames = int(VAD_MAX_RECORD_SEC * SAMPLE_RATE / VAD_FRAME)
    min_speech_frames = int(VAD_MIN_SPEECH_SEC * SAMPLE_RATE / VAD_FRAME)
    speech_frames = 0

    logger.info("🎤 Слушаю команду...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_FRAME,
        device=DEVICE,
    ) as stream:
        for _ in range(max_frames):
            frame, _ = stream.read(VAD_FRAME)
            frame = frame[:, 0].copy()
            prob = _prob(frame)

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
    logger.info("✅ Записано %.1f сек", len(audio) / SAMPLE_RATE)
    return audio
