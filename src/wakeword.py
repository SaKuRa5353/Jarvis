"""Wakeword 'Hey Jarvis' через openWakeWord."""
from src.logger import get_logger

logger = get_logger("wakeword")
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from config import (
    DEVICE,
    SAMPLE_RATE,
    WAKE_BLOCK_SIZE,
    WAKE_COOLDOWN,
    WAKE_MODEL,
    WAKE_THRESHOLD,
)


_model = Model(wakeword_model_paths=[WAKE_MODEL])

_triggered = False
_last_time = 0.0


def _callback(indata, frames, time_info, status):
    global _triggered, _last_time
    if status:
        logger.warning("[wake status] %s", status)
    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = _model.predict(audio)
    score = prediction.get("hey_jarvis_v0.1", 0.0)
    now = time.time()
    if score > WAKE_THRESHOLD and (now - _last_time) > WAKE_COOLDOWN:
        _last_time = now
        _triggered = True


def wait_for_wakeword():
    """Блокирующе ждёт, пока не услышит 'Hey Jarvis'."""
    global _triggered
    _triggered = False
    logger.info("💤 Жду 'Hey Jarvis'...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=WAKE_BLOCK_SIZE,
        device=DEVICE,
        callback=_callback,
    ):
        while not _triggered:
            sd.sleep(100)

    logger.info("🎯 Услышал wakeword!")
