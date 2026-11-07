#!/usr/bin/env python3
"""Измерение акустического детектора обращения на WAV-выборках."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mathcore.features import dtw, mfcc_sequence  # noqa: E402
from src.mathcore.thresholds import auc, fah, np_threshold, roc_curve  # noqa: E402


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"{path}: нужен моно WAV PCM int16")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).copy(), stream.getframerate()


def _write_wav(path: Path, audio: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(audio.astype(np.int16).tobytes())


def _score(audio: np.ndarray, references: list[np.ndarray], sample_rate: int) -> float:
    sequence = mfcc_sequence(audio, sample_rate)
    if len(sequence) == 0:
        return 0.0
    distance = min(dtw(sequence, reference, window=5)[0] for reference in references)
    return 1.0 / (1.0 + distance) if np.isfinite(distance) else 0.0


def measure(positives: Path, noise: Path, frame_sec: float) -> dict[str, Any]:
    positive_paths = sorted(positives.glob("*.wav"))
    if not positive_paths:
        raise ValueError("в --positives нет WAV-файлов")
    positive_audio = [_read_wav(path) for path in positive_paths]
    sample_rate = positive_audio[0][1]
    references = [mfcc_sequence(audio, sample_rate) for audio, _ in positive_audio]
    references = [reference for reference in references if len(reference)]
    if not references:
        raise ValueError("не удалось получить MFCC эталонов")
    positive_scores = [_score(audio, references, rate) for audio, rate in positive_audio]
    noise_audio, noise_rate = _read_wav(noise)
    chunk = max(1, int(frame_sec * noise_rate))
    noise_scores = [_score(noise_audio[start : start + chunk], references, noise_rate)
                    for start in range(0, len(noise_audio), chunk)]
    scores = positive_scores + noise_scores
    labels = [1] * len(positive_scores) + [0] * len(noise_scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    alpha0 = frame_sec / 3600.0
    recommended = np_threshold(noise_scores, alpha0)
    index = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - recommended))
    return {
        "kind": "wakeword",
        "sample_rate": sample_rate,
        "positive_count": len(positive_scores),
        "negative_count": len(noise_scores),
        "TPR": float(tpr[index]),
        "FPR": float(fpr[index]),
        "FAH": fah(float(fpr[index]), frame_sec),
        "ROC_AUC": auc(fpr, tpr),
        "neyman_threshold": recommended,
        "positive_scores": positive_scores,
        "negative_scores": noise_scores,
    }


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        positives = root / "positives"
        positives.mkdir()
        t = np.arange(16000) / 16000.0
        for index in range(3):
            _write_wav(positives / f"positive_{index}.wav", (12000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16))
        noise = root / "noise.wav"
        _write_wav(noise, np.zeros(16000, dtype=np.int16))
        return measure(positives, noise, 0.032)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path)
    parser.add_argument("--noise", type=Path)
    parser.add_argument("--record-noise", type=float, help="зарезервировано для аппаратной записи")
    parser.add_argument("--json", type=Path, default=Path("wakeword_results.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test()
    else:
        if args.positives is None or args.noise is None:
            parser.error("нужны --positives и --noise, либо --self-test")
        result = measure(args.positives, args.noise, 0.032)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("positive_scores", "negative_scores")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
