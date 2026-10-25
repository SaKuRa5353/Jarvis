"""Генерация коротких звуков подтверждения (earcons) в src/phrases/.

    python tests/generate_earcons.py

Зачем: earcon 180 мс вместо речевой фразы ~0.6-1.1 с = минус ~0.5 с на каждом
цикле и заметно меньше эха в микрофон (реже самосрабатывание wakeword на
собственный TTS). Синтез они НЕ экономят — готовые WAV-фразы уже есть.

Без сторонних зависимостей: wave + math + struct.
Файлы попадают в git (.gitignore содержит !src/phrases/*.wav).
"""

from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SILERO_SAMPLE_RATE  # noqa: E402
from src.phrases import PHRASES_DIR  # noqa: E402

# имя файла -> (частоты по сегментам, длительность мс, громкость 0..1)
TONES: dict[str, tuple[tuple[float, ...], int, float]] = {
    "earcon_welcome.wav": ((880.0,), 180, 0.35),
    "earcon_thinking.wav": ((660.0, 990.0), 220, 0.30),
    "earcon_done.wav": ((990.0, 1320.0), 200, 0.35),
}


def make_tone(freqs: tuple[float, ...], ms: int, gain: float, sr: int) -> bytes:
    n = int(sr * ms / 1000)
    seg = max(1, n // len(freqs))
    fade = int(sr * 0.010)  # 10 мс на фронты — иначе щёлкает
    frames = bytearray()
    for i in range(n):
        f = freqs[min(i // seg, len(freqs) - 1)]
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > n - fade:
            env = max(0.0, (n - i) / fade)
        val = int(gain * env * 32767 * math.sin(2 * math.pi * f * i / sr))
        frames += struct.pack("<h", max(-32768, min(32767, val)))
    return bytes(frames)


def main() -> int:
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)
    sr = int(SILERO_SAMPLE_RATE)
    for name, (freqs, ms, gain) in TONES.items():
        path = PHRASES_DIR / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(make_tone(freqs, ms, gain, sr))
        print(f"  ✅ {path}  {path.stat().st_size / 1024:.1f} КБ  {ms} мс  {freqs}")
    print('\nВключить: CONFIRM_SOUND = "earcon" в config.py')
    print("         (или JARVIS_CONFIRM_SOUND=earcon в .env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
