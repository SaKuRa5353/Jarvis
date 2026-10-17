"""Проверка микрофона: запись 3 секунды и пиковая амплитуда.

Запуск:
    python tests/check_mic.py
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEVICE, SAMPLE_RATE  # noqa: E402

DURATION = 3

print(f"Говори что-нибудь {DURATION} секунды... (device {DEVICE})")
rec = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
    device=DEVICE,
)
sd.wait()

amplitude = np.abs(rec).max()
print(f"Записано. Max amplitude: {amplitude}")
if amplitude > 500:
    print("Микрофон работает и слышит тебя.")
else:
    print("Слишком тихо — подними усиление в pavucontrol -> Input Devices.")
    sys.exit(1)
