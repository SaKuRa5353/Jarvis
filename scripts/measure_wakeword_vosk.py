#!/usr/bin/env python3
"""Измерение production wakeword-пути Vosk на заранее записанных WAV."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.vosk_model import get_vosk_model  # noqa: E402
from src.wakeword import Subscriber, WakewordDetector  # noqa: E402


class ReplaySubscriber(Subscriber):
    """Offline-реализация подписчика для прогона WAV без микрофона."""

    def __init__(self, audio: np.ndarray, frame: int, tail_frames: int) -> None:
        tail = np.zeros(frame * tail_frames, dtype=np.int16)
        self._audio = np.concatenate((audio, tail))
        self._position = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        """True после передачи всей записи и искусственного тишинного хвоста."""
        return self._finished

    def read(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """Возвращает очередной полный production-кадр."""
        del timeout
        if self._position >= len(self._audio):
            self._finished = True
            return np.zeros(n_samples, dtype=np.int16)
        end = self._position + n_samples
        chunk = self._audio[self._position:end]
        self._position = end
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        if self._position >= len(self._audio):
            self._finished = True
        return chunk.astype(np.int16, copy=False)

    def check(self) -> None:
        """Совместимость с Subscriber; в offline replay проверка не нужна."""

    def drain(self) -> None:
        """Совместимость с Subscriber; replay начинается с нулевой позиции."""
        self._position = 0
        self._finished = False


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
    if len(audio) < config.VAD_FRAME:
        raise ValueError(f"{path}: запись слишком короткая, нужно минимум {config.VAD_FRAME} samples")
    return audio, len(audio) / sample_rate


def replay_detector(
    audio: np.ndarray,
    model: Any,
    sample_rate: int = config.SAMPLE_RATE,
) -> tuple[list[str], list[str]]:
    """Прогоняет аудио через WakewordDetector и возвращает (фразы, wakeword-фразы)."""
    frame = config.VAD_FRAME
    tail_frames = max(1, round(config.VAD_MIN_SILENCE_SEC * sample_rate / frame) + 1)
    subscriber = ReplaySubscriber(audio, frame, tail_frames)
    detector = WakewordDetector(subscriber, model=model, sample_rate=sample_rate)
    phrases = list(detector.phrases(should_stop=lambda: subscriber.finished))
    matches = [text for text in phrases if detector.is_match(text)]
    return phrases, matches


def _load_model(model_path: Path | None) -> Any:
    if model_path is None:
        return get_vosk_model()
    try:
        from vosk import Model, SetLogLevel  # type: ignore[import-untyped]

    except ModuleNotFoundError as exc:
        raise RuntimeError("Vosk не установлен") from exc
    if not model_path.is_dir():
        raise FileNotFoundError(f"Vosk-модель не найдена: {model_path}")
    SetLogLevel(-1)
    return Model(str(model_path))


def measure(positives: Path, noise: Path, model_path: Path | None) -> dict[str, Any]:
    positive_paths = sorted(positives.glob("*.wav"))
    if not positive_paths:
        raise ValueError("в --positives нет WAV-файлов")
    positive_results = []
    model = _load_model(model_path)
    for path in positive_paths:
        audio, duration = _read_wav(path)
        phrases, matches = replay_detector(audio, model)
        positive_results.append({
            "file": path.name,
            "duration_sec": duration,
            "phrases": phrases,
            "wakeword_matches": matches,
            "detected": bool(matches),
            "status": "ok",
        })
    noise_audio, noise_duration = _read_wav(noise)
    noise_phrases, noise_matches = replay_detector(noise_audio, model)
    false_events = len(noise_matches)
    return {
        "kind": "wakeword_vosk",
        "status": "ok",
        "detector": "src.wakeword.WakewordDetector",
        "positive_count": len(positive_results),
        "positive_detected": sum(bool(item["detected"]) for item in positive_results),
        "TPR": float(sum(bool(item["detected"]) for item in positive_results)) / len(positive_results),
        "noise_file": noise.name,
        "noise_duration_sec": noise_duration,
        "false_wakeword_events": false_events,
        "FAH": false_events / (noise_duration / 3600.0),
        "noise_phrases": noise_phrases,
        "noise_wakeword_matches": noise_matches,
        "ROC_AUC": None,
        "threshold": None,
        "positives": positive_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = measure(args.positives, args.noise, args.model)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        result = {"kind": "wakeword_vosk", "status": "unavailable", "error": str(exc)}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "positives"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
