"""Offline-тесты E2 Vosk replay без микрофона и модели."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts import measure_wakeword_vosk as e2


def _write_wav(path: Path, audio: np.ndarray, rate: int = 16000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(audio.astype(np.int16).tobytes())


def test_read_wav_accepts_required_format(tmp_path: Path) -> None:
    path = tmp_path / "ok.wav"
    _write_wav(path, np.arange(1024, dtype=np.int16))
    audio, duration = e2._read_wav(path)
    assert len(audio) == 1024
    assert duration == pytest.approx(1024 / 16000)


@pytest.mark.parametrize(
    ("rate", "channels", "samples"),
    [(8000, 1, 1024), (16000, 2, 2048), (16000, 1, 100)],
)
def test_read_wav_rejects_invalid_format(
    tmp_path: Path, rate: int, channels: int, samples: int
) -> None:
    path = tmp_path / "bad.wav"
    _write_wav(path, np.zeros(samples, dtype=np.int16), rate, channels)
    with pytest.raises(ValueError):
        e2._read_wav(path)


def test_replay_subscriber_preserves_audio_and_adds_only_tail() -> None:
    audio = np.arange(1000, dtype=np.int16)
    subscriber = e2.ReplaySubscriber(audio, frame=512, tail_frames=2)
    chunks = [subscriber.read(512), subscriber.read(512), subscriber.read(512), subscriber.read(512)]
    assert np.concatenate(chunks[:2]).tolist()[:1000] == audio.tolist()
    assert np.all(np.concatenate(chunks[2:]) == 0)


def test_production_wakeword_match_is_binary() -> None:
    detector = e2.WakewordDetector.__new__(e2.WakewordDetector)
    detector.wake_words = ("джарвис", "jarvis")
    assert detector.is_match("привет джарвис") is True
    assert detector.is_match("привет") is False


def test_measure_counts_separate_negative_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeDetector:
        def __init__(self, subscriber: object, model: object, sample_rate: object) -> None:
            del subscriber, model, sample_rate

        def phrases(self, should_stop: object) -> object:
            del should_stop
            return iter(("шум", "джарвис", "ещё джарвис"))

        @staticmethod
        def is_match(text: str) -> bool:
            return "джарвис" in text

    monkeypatch.setattr(e2, "WakewordDetector", FakeDetector)
    monkeypatch.setattr(e2, "_load_model", lambda path: object())
    positive_dir = tmp_path / "positive"
    positive_dir.mkdir()
    _write_wav(positive_dir / "001.wav", np.zeros(1024, dtype=np.int16))
    noise = tmp_path / "noise.wav"
    _write_wav(noise, np.zeros(16000, dtype=np.int16))

    result = e2.measure(positive_dir, noise, None)
    assert result["positive_detected"] == 1
    assert result["false_wakeword_events"] == 2
    assert result["FAH"] == pytest.approx(2.0 * 3600.0)


def test_main_reports_unavailable_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "e2.json"
    monkeypatch.setattr(sys, "argv", ["measure_wakeword_vosk.py", "--positives", str(tmp_path), "--noise", str(tmp_path / "noise.wav"), "--model", str(tmp_path / "missing"), "--json", str(output)])
    assert e2.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
