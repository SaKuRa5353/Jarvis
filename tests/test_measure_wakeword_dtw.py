"""Offline-тесты раздельного экспериментального DTW-измерения."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from scripts import measure_wakeword_dtw as dtw_measure
from src.mathcore.features import mfcc_sequence


def _write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(audio.astype(np.int16).tobytes())


def test_reference_and_positive_sets_are_separate(tmp_path: Path) -> None:
    audio = (10000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.int16)
    references = tmp_path / "references"
    positives = tmp_path / "positives"
    references.mkdir()
    positives.mkdir()
    np.save(references / "ref_000.npy", mfcc_sequence(audio))
    _write_wav(positives / "positive_000.wav", audio)
    noise = tmp_path / "noise.wav"
    _write_wav(noise, np.zeros(32000, dtype=np.int16))

    result = dtw_measure.measure(references, positives, noise)

    assert result["reference_files"] == ["ref_000.npy"]
    assert result["reference_count"] == 1
    assert result["positive_count"] == 1
    assert result["positive_scores"][0]["file"] == "positive_000.wav"
    assert result["negative_window_count"] > 0
