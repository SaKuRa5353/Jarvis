#!/usr/bin/env python3
"""Отдельное измерение экспериментального MFCC/DTW wakeword-детектора."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.mathcore.features import dtw, mfcc_sequence  # noqa: E402
from src.mathcore.thresholds import auc, roc_curve  # noqa: E402

HOP_SAMPLES = 160
MFCC_COEFFICIENTS = 13


def _read_wav(path: Path, sample_rate: int = config.SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Читает только mono 16 kHz PCM signed 16-bit WAV."""
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            rate = stream.getframerate()
            frames = stream.getnframes()
            if channels != 1 or width != 2 or rate != sample_rate:
                raise ValueError(
                    f"{path}: ожидается mono PCM16 {sample_rate} Hz, "
                    f"получено channels={channels}, width={width * 8}, rate={rate}"
                )
            raw = stream.readframes(frames)
    except (OSError, wave.Error) as exc:
        raise ValueError(f"{path}: не удалось прочитать WAV: {exc}") from exc
    audio = np.frombuffer(raw, dtype=np.int16).copy()
    if len(audio) < 512:
        raise ValueError(f"{path}: запись слишком короткая")
    return audio, len(audio) / sample_rate


def _load_references(directory: Path) -> tuple[list[np.ndarray], list[str]]:
    """Загружает только MFCC .npy из отдельного reference-каталога."""
    references: list[np.ndarray] = []
    names: list[str] = []
    for path in sorted(directory.glob("*.npy")):
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] != MFCC_COEFFICIENTS or not len(matrix):
            raise ValueError(f"{path}: ожидается непустая MFCC-матрица (T, 13)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path}: MFCC содержит NaN или Inf")
        references.append(matrix.astype(np.float64))
        names.append(path.name)
    if not references:
        raise ValueError(f"в {directory} нет MFCC-эталонов .npy")
    return references, names


def _similarity(audio: np.ndarray, references: list[np.ndarray]) -> float:
    sequence = mfcc_sequence(audio, config.SAMPLE_RATE)
    if not len(sequence):
        raise ValueError("окно слишком короткое для MFCC")
    distance = min(dtw(sequence, reference, window=config.WAKEWORD_DTW_WINDOW)[0]
                  for reference in references)
    return 1.0 / (1.0 + distance) if np.isfinite(distance) else 0.0


def _negative_windows(audio: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    """Режет фон на окна длительностью типичного эталона с 50% перекрытием."""
    step = max(1, frame_samples // 2)
    if len(audio) < frame_samples:
        raise ValueError("отрицательный фон короче окна типичного эталона")
    return [audio[start:start + frame_samples]
            for start in range(0, len(audio) - frame_samples + 1, step)]


def measure(references_dir: Path, positives: Path, noise: Path) -> dict[str, Any]:
    references, reference_names = _load_references(references_dir)
    positive_paths = sorted(positives.glob("*.wav"))
    if not positive_paths:
        raise ValueError("в --positives нет WAV-файлов")
    positive_audio = [_read_wav(path) for path in positive_paths]
    reference_frames = int(round(median(len(reference) for reference in references)))
    window_samples = max(512, reference_frames * HOP_SAMPLES)
    positive_scores = [
        {"file": path.name, "score": _similarity(audio, references), "status": "ok"}
        for path, (audio, _) in zip(positive_paths, positive_audio, strict=False)
    ]
    noise_audio, noise_duration = _read_wav(noise)
    windows = _negative_windows(noise_audio, window_samples)
    negative_scores = [_similarity(window, references) for window in windows]
    scores = [item["score"] for item in positive_scores] + negative_scores
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    fpr, tpr, _ = roc_curve(labels, [float(s) for s in scores])  # type: ignore[arg-type]
    return {
        "kind": "wakeword_dtw",
        "status": "ok",
        "detector": "MFCC + DTW (экспериментальный, не production Vosk)",
        "reference_files": reference_names,
        "reference_count": len(references),
        "positive_count": len(positive_scores),
        "negative_window_count": len(negative_scores),
        "noise_duration_sec": noise_duration,
        "window_duration_sec": window_samples / config.SAMPLE_RATE,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "ROC_AUC": float(auc(list(fpr), list(tpr))),
        "threshold": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = measure(args.references, args.positives, args.noise)
    except (OSError, ValueError):
        raise
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("positive_scores", "negative_scores")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
