"""Общие pytest-фикстуры проекта и пропуск аппаратных тестов без устройств."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def rng() -> np.random.Generator:
    """Воспроизводимый генератор случайных чисел."""
    return np.random.default_rng(42)


@pytest.fixture
def tmp_cfg(monkeypatch: pytest.MonkeyPatch):
    """Возвращает helper для временной подмены атрибутов ``config``."""
    import config

    def apply(**values: object) -> None:
        for name, value in values.items():
            monkeypatch.setattr(config, name, value)

    return apply


@pytest.fixture
def synthetic_audio():
    """Возвращает генератор синтетического int16-аудио: синус плюс шум."""

    def generate(
        length: int = 16000,
        frequency: float = 440.0,
        sample_rate: int = 16000,
        amplitude: float = 0.5,
        noise: float = 0.0,
    ) -> np.ndarray:
        if length < 0 or sample_rate <= 0 or amplitude < 0 or noise < 0:
            raise ValueError("Некорректные параметры синтетического аудио")
        time = np.arange(length, dtype=np.float64) / sample_rate
        signal = amplitude * np.sin(2.0 * np.pi * frequency * time)
        if noise:
            signal += np.random.default_rng(42).normal(0.0, noise, length)
        return np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16)

    return generate


def _has_input_device() -> bool:
    """Проверяет наличие входного аудиоустройства без падения при headless CI."""
    try:
        import sounddevice as sd

        return any(device.get("max_input_channels", 0) > 0 for device in sd.query_devices())
    except (ImportError, OSError, RuntimeError):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Помечает hw-тесты skip, если микрофон недоступен."""
    if _has_input_device():
        return
    skip_hw = pytest.mark.skip(reason="нет доступного входного аудиоустройства")
    for item in items:
        if "hw" in item.keywords:
            item.add_marker(skip_hw)
