#!/usr/bin/env python3
"""Запись эталонов «Джарвис» для DTW-детектора (требует микрофон, маркер hw).

Записывает N произнесений по одному, сохраняет MFCC-последовательности в
data/wakeword/ref_*.npy. Запуск вручную: python scripts/record_wakeword_refs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.mathcore.features import mfcc_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5, help="сколько произнесений записать")
    parser.add_argument("--sec", type=float, default=1.5, help="длительность одной записи")
    parser.add_argument("--device", type=int, default=config.DEVICE)
    args = parser.parse_args()

    import sounddevice as sd

    for index in range(args.count):
        input(f"Эталон {index + 1}/{args.count}: Enter, затем скажи «Джарвис»... ")
        audio = sd.rec(
            int(args.sec * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=args.device,
        )
        sd.wait()
        flat = audio.flatten()
        if np.abs(flat).max() < 500:
            print("⚠️ Слишком тихо, запись отброшена")
            continue
        sequence = mfcc_sequence(flat, config.SAMPLE_RATE)
        out_dir = config.DATA_WAKEWORD_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"ref_{len(list(out_dir.glob('ref_*.npy'))):03d}.npy"
        np.save(path, sequence)
        print(f"✅ Сохранён {path} ({len(sequence)} кадров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())