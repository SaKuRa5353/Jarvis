"""Проверка STT через Vosk (офлайн, без интернета)."""

import sounddevice as sd
import numpy as np
from config import DEVICE, SAMPLE_RATE  # noqa: E402
from src.stt import VoskSTT

DURATION = 5
SR = SAMPLE_RATE  # из config, а не литералом

print(f"Говори {DURATION} секунд... (microphone #{DEVICE})")
rec = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='int16', device=DEVICE)
sd.wait()

audio = rec.flatten()
print(f"Записано {len(audio)/SR:.1f} сек. Amplitude: {np.abs(rec).max()}")

if np.abs(rec).max() < 500:
    print("⚠️ Слишком тихо! Подними усиление в pavucontrol.")
    exit(1)

stt = VoskSTT()
text = stt.recognize(audio)
print(f"Распознано: {text if text else '(пусто)'}")

