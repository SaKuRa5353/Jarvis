"""Тесты безопасного сохранения raw WAV без микрофона."""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from scripts import record_wakeword_wav
from scripts.record_wakeword_wav import _write_wav


def test_write_wav_creates_readable_mono_pcm16_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    audio = np.array([-32768, -123, 0, 123, 32767], dtype=np.int16)

    _write_wav(path, audio, 16000)

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 16000
        assert stream.getsampwidth() == 2
        assert stream.getcomptype() == "NONE"
        assert np.array_equal(np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16), audio)


def test_write_wav_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "existing.wav"
    original = b"existing-content"
    path.write_bytes(original)

    with pytest.raises(FileExistsError):
        _write_wav(path, np.zeros(8, dtype=np.int16), 16000)

    assert path.read_bytes() == original


class CountingStream:
    """Базовый пайтовый поток с подсчётом вызовов и флагом закрытия."""

    def __init__(self, frames_per_call: int = 1600, fail_at_call: int | None = None, **kwargs: object) -> None:
        del kwargs
        self.frames_per_call = frames_per_call
        self.fail_at_call = fail_at_call
        self.calls = 0
        self.closed = False

    def __enter__(self) -> CountingStream:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.closed = True
        return False

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        if self.fail_at_call is not None and self.calls == self.fail_at_call:
            raise RuntimeError("ошибка устройства")
        self.calls += 1
        return np.zeros((frames, 1), dtype=np.int16), False


def _fake_sounddevice(monkeypatch: pytest.MonkeyPatch, stream_factory: object) -> types.SimpleNamespace:
    sd = types.SimpleNamespace(
        InputStream=stream_factory,
        rec=lambda *a, **kw: None,
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_close_stream_on_successful_noise_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def factory(**kwargs: object) -> CountingStream:
        return CountingStream(frames_per_call=1600, **kwargs)  # type: ignore[arg-type]
    _fake_sounddevice(monkeypatch, factory)
    monkeypatch.setattr("builtins.input", lambda prompt: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path), "--seconds", "1"],
    )

    assert record_wakeword_wav.main() == 0
    files = sorted(tmp_path.glob("*.wav"))
    assert len(files) == 1
    assert files[0].name == "noise_001.wav"
    partials = list(tmp_path.glob("*.partial"))
    assert partials == [], f"частичных файлов не должно быть, но найден{'' if len(partials)==1 else 'ы'} {partials}"


def test_partial_file_has_non_wav_extension_on_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Прерванная запись сохраняется с расширением .partial и не попадает в поиск *.wav."""
    closed_instances: list[bool] = []

    class InterruptedStream(CountingStream):
        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            if self.calls >= 2:
                raise KeyboardInterrupt()
            return super().read(frames)

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
            closed_instances.append(exc_type is not None)
            return False

    _fake_sounddevice(monkeypatch, InterruptedStream)
    monkeypatch.setattr("builtins.input", lambda prompt: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path), "--seconds", "10"],
    )

    assert record_wakeword_wav.main() == 0
    partials = sorted(tmp_path.glob("*.partial"))
    wav_files = sorted(tmp_path.glob("*.wav"))
    assert len(partials) == 1
    assert wav_files == []
    assert partials[0].name == "noise_partial_001.partial"
    assert closed_instances == [True]


def test_partial_file_created_on_read_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ошибка чтения во время записи также сохраняет частичный файл с .partial."""
    closed_instances: list[bool] = []

    class ErrorAfterCallStream(CountingStream):
        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            if self.calls >= 2:
                raise RuntimeError("ошибка устройства")
            return super().read(frames)

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
            closed_instances.append(exc_type is not None)
            return False

    _fake_sounddevice(monkeypatch, ErrorAfterCallStream)
    monkeypatch.setattr("builtins.input", lambda prompt: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path), "--seconds", "5"],
    )

    assert record_wakeword_wav.main() == 0
    partials = sorted(tmp_path.glob("*.partial"))
    wav_files = sorted(tmp_path.glob("*.wav"))
    assert len(partials) == 1
    assert wav_files == []
    assert partials[0].name == "noise_partial_001.partial"
    assert closed_instances == [True]


def test_noise_mode_writes_one_file_without_wakeword_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []
    prompts: list[str] = []

    class FakeStream:
        def __init__(self, samplerate: object, device: object, channels: object, dtype: object, blocksize: object, **kwargs: object) -> None:
            del samplerate, device, channels, dtype, kwargs
            calls.append(blocksize)  # type: ignore[arg-type]

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
            return False

        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            return np.zeros((frames, 1), dtype=np.int16), False

    sd = types.SimpleNamespace(InputStream=FakeStream, rec=lambda *a, **kw: None, wait=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path), "--seconds", "1"],
    )

    assert record_wakeword_wav.main() == 0
    files = sorted(tmp_path.glob("*.wav"))
    assert [path.name for path in files] == ["noise_001.wav"]
    assert len(calls) == 1
    assert "Джарвис" not in prompts[0]
    with wave.open(str(files[0]), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 16000
        assert stream.getsampwidth() == 2


def test_noise_mode_uses_next_path_without_overwriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []

    class FakeStream:
        def __init__(self, samplerate: object, device: object, channels: object, dtype: object, blocksize: object, **kwargs: object) -> None:
            del samplerate, device, channels, dtype, kwargs
            calls.append(blocksize)  # type: ignore[arg-type]

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
            return False

        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            return np.zeros((frames, 1), dtype=np.int16), False

    sd = types.SimpleNamespace(InputStream=FakeStream, rec=lambda *a, **kw: None, wait=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr("builtins.input", lambda prompt: None)
    existing = tmp_path / "noise_001.wav"
    existing.write_bytes(b"keep")
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path)],
    )

    assert record_wakeword_wav.main() == 0
    assert existing.read_bytes() == b"keep"
    assert (tmp_path / "noise_002.wav").exists()


def test_wakeword_mode_keeps_separate_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []
    prompts: list[str] = []

    class FakeStream:
        def __init__(self, samplerate: object, device: object, channels: object, dtype: object, blocksize: object, **kwargs: object) -> None:
            del samplerate, device, channels, dtype, kwargs
            calls.append(blocksize)  # type: ignore[arg-type]

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
            return False

        def read(self, frames: int) -> tuple[np.ndarray, bool]:
            return np.zeros((frames, 1), dtype=np.int16), False

    sd = types.SimpleNamespace(InputStream=FakeStream, rec=lambda *a, **kw: None, wait=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt))
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--out", str(tmp_path), "--count", "2", "--seconds", "1"],
    )

    assert record_wakeword_wav.main() == 0
    assert sorted(path.name for path in tmp_path.glob("*.wav")) == [
        "wakeword_001.wav",
        "wakeword_002.wav",
    ]
    assert len(calls) == 2
    assert all("Джарвис" in prompt for prompt in prompts)


def test_noise_mode_rejects_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_wakeword_wav.py", "--mode", "noise", "--out", str(tmp_path), "--count", "1"],
    )
    with pytest.raises(SystemExit):
        record_wakeword_wav.main()
