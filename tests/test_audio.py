"""Регрессии контракта фиксированного размера аудиокадров и валидации устройств."""

from __future__ import annotations

import types

import numpy as np
import pytest

from src.audio import BroadcastBuffer, MicrophoneError, validate_input_device


def test_subscriber_splits_large_block_into_exact_frames() -> None:
    broadcast = BroadcastBuffer()
    subscriber = broadcast.subscribe()
    block = np.arange(1280, dtype=np.int16).tobytes()
    broadcast.push(block)

    first = subscriber.read(512, timeout=0.01)
    second = subscriber.read(512, timeout=0.01)
    third = subscriber.read(256, timeout=0.01)

    assert [len(first), len(second), len(third)] == [512, 512, 256]
    np.testing.assert_array_equal(
        np.concatenate([first, second, third]), np.arange(1280, dtype=np.int16)
    )


class FakeSD:
    """Минимальный мак sounddevice для тестирования validate_input_device()."""

    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices
        self.default = types.SimpleNamespace(device=[None])

    def query_devices(self) -> list[dict]:
        return self._devices


def _make_sd(*, valid_indices: list[int], default_idx: int | None = None) -> FakeSD:
    """Создаёт FakeSD с устройствами: заданные индексы имеют входные каналы, остальные — 0."""
    max_idx = max(valid_indices) if valid_indices else 0
    devices = []
    for i in range(max_idx + 1):
        channels = (i + 1) if i in valid_indices else 0
        devices.append({"index": i, "max_input_channels": channels, "name": f"dev_{i}"})
    sd = FakeSD(devices)
    if default_idx is not None:
        sd.default.device = [default_idx]
    else:
        sd.default.device = [None]
    return sd


def test_valid_requested_device_returns_itself() -> None:
    sd = _make_sd(valid_indices=[3], default_idx=None)
    assert validate_input_device(sd, 3) == 3


def test_raises_on_invalid_index(monkeypatch: pytest.MonkeyPatch) -> None:
    sd = _make_sd(valid_indices=[0], default_idx=None)
    monkeypatch.setattr(sd, "default", types.SimpleNamespace(device=[99]))
    with pytest.raises(MicrophoneError, match="не существует"):
        validate_input_device(sd, 99)


def test_raises_when_device_has_no_input_channels() -> None:
    sd = _make_sd(valid_indices=[], default_idx=None)
    with pytest.raises(MicrophoneError, match="Устройство #0.*не имеет входных каналов"):
        validate_input_device(sd, 0)


def test_raises_when_query_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    sd = FakeSD([{"index": 0, "max_input_channels": 2, "name": "ok"}])
    monkeypatch.setattr(sd, "query_devices", lambda: (_ for _ in []).throw(RuntimeError("fail")))
    with pytest.raises(MicrophoneError, match="Не удалось получить список"):
        validate_input_device(sd, 0)


def test_raises_when_query_returns_empty_list() -> None:
    sd = FakeSD([])
    with pytest.raises(MicrophoneError, match="ни одного аудиоустройства"):
        validate_input_device(sd, 0)
