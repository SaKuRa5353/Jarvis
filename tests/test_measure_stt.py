"""Регрессионные тесты передачи полного WAV-потока в Vosk."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from scripts import measure_stt


def test_recognizer_receives_all_samples_once(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[bytes] = []

    class FakeRecognizer:
        def __init__(self, model: object, sample_rate: int) -> None:
            assert model == "model"
            assert sample_rate == 16000

        def AcceptWaveform(self, data: bytes) -> None:
            received.append(data)

        def FinalResult(self) -> str:
            return json.dumps({"text": "тест"})

    monkeypatch.setitem(sys.modules, "vosk", types.SimpleNamespace(KaldiRecognizer=FakeRecognizer))
    monkeypatch.setattr(measure_stt, "_recognize_vosk", measure_stt._recognize_vosk)
    import src.vosk_model

    monkeypatch.setattr(src.vosk_model, "get_vosk_model", lambda: "model")
    audio = np.arange(5001, dtype=np.int16)

    assert measure_stt._recognize_vosk(audio, 16000) == "тест"
    assert b"".join(received) == audio.tobytes()