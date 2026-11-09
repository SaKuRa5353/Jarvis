"""Регрессии контракта фиксированного размера аудиокадров и валидации устройств."""

from __future__ import annotations

import types

import numpy as np
import pytest

from src.audio import (
    BroadcastBuffer,
    MicrophoneError,
    select_input_device,
    validate_input_device,
)


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

    def __init__(
        self,
        devices: list[dict],
        supported_rates: dict[int, list[int]] | None = None,
    ) -> None:
        self._devices = devices
        self.default = types.SimpleNamespace(device=[None])
        self._supported_rates = supported_rates or {}

    def query_devices(self) -> list[dict]:
        return self._devices

    def query_hostapis(self) -> list[dict]:
        return [{"name": "ALSA"}, {"name": "JACK"}]

    def check_input_settings(self, device: int, channels: int, samplerate: int, dtype: str) -> None:
        rates = self._supported_rates.get(device, [16000])
        if samplerate not in rates:
            raise RuntimeError(f"Invalid sample rate {samplerate}")


def _make_sd(
    *,
    valid_indices: list[int],
    default_idx: int | None = None,
    rates: dict[int, list[int]] | None = None,
) -> FakeSD:
    """Создаёт FakeSD с устройствами: заданные индексы имеют входные каналы, остальные — 0."""
    max_idx = max(valid_indices) if valid_indices else 0
    devices = []
    for i in range(max_idx + 1):
        channels = (i + 1) if i in valid_indices else 0
        devices.append(
            {
                "index": i,
                "max_input_channels": channels,
                "name": f"dev_{i}",
                "default_samplerate": 44100.0,
            }
        )
    sd = FakeSD(devices, supported_rates=rates)
    if default_idx is not None:
        sd.default.device = [default_idx]
    else:
        sd.default.device = [None]
    return sd


def test_valid_requested_device_returns_itself_and_rate() -> None:
    sd = _make_sd(valid_indices=[3], default_idx=None, rates={3: [16000]})
    assert validate_input_device(sd, 3) == (3, 16000)


def test_device_without_16k_returns_native_rate() -> None:
    sd = _make_sd(valid_indices=[3], default_idx=None, rates={3: [44100]})
    assert validate_input_device(sd, 3) == (3, 44100)


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


def test_name_selection_survives_index_change() -> None:
    sd = FakeSD(
        [
            {"name": "Built-in Mic", "max_input_channels": 2},
            {"name": "USB Audio CODEC (hw:2,0)", "max_input_channels": 1},
        ]
    )
    assert select_input_device(sd, requested_name="USB Audio CODEC (hw:1,0)") == 1


def test_name_selection_can_match_hostapi() -> None:
    sd = FakeSD(
        [
            {"name": "USB Audio CODEC", "max_input_channels": 1, "hostapi": 0},
            {"name": "USB Audio CODEC", "max_input_channels": 1, "hostapi": 1},
        ]
    )
    assert select_input_device(sd, requested_name="USB Audio CODEC", requested_hostapi="JACK") == 1


def test_ambiguous_name_is_rejected() -> None:
    sd = FakeSD(
        [
            {"name": "USB Audio CODEC", "max_input_channels": 1},
            {"name": "USB Audio CODEC", "max_input_channels": 1},
        ]
    )
    with pytest.raises(MicrophoneError, match="неоднозначно"):
        select_input_device(sd, requested_name="USB Audio CODEC")


def test_missing_name_is_an_error_without_fallback() -> None:
    sd = _make_sd(valid_indices=[0], default_idx=0)
    with pytest.raises(MicrophoneError, match="не найдено"):
        select_input_device(sd, requested_name="USB Audio CODEC")


def test_default_input_is_used_without_configuration() -> None:
    sd = _make_sd(valid_indices=[0, 2], default_idx=2)
    assert select_input_device(sd) == 2


def test_default_output_is_not_used_as_microphone() -> None:
    sd = FakeSD(
        [{"name": "HDMI", "max_input_channels": 0}, {"name": "Mic", "max_input_channels": 1}]
    )
    sd.default.device = [0]
    with pytest.raises(MicrophoneError, match="не имеет входных каналов"):
        select_input_device(sd)


def test_legacy_index_remains_supported_after_name_priority() -> None:
    sd = _make_sd(valid_indices=[1], default_idx=1)
    assert select_input_device(sd, requested_device=1) == 1


def test_selected_device_is_validated_at_current_index() -> None:
    sd = FakeSD(
        [{"name": "USB Audio CODEC", "max_input_channels": 1, "default_samplerate": 44100}],
        supported_rates={0: [44100]},
    )
    assert validate_input_device(sd, requested_name="USB Audio CODEC") == (0, 44100)


def test_stream_resampler_preserves_length_and_continuity() -> None:
    """Проверяет длину, индексы и стыки при дробном преобразовании."""
    from src.audio import StreamResampler

    for orig_sr in (44100, 48000):
        samples = int(orig_sr * 0.23) + 37
        t = np.arange(samples, dtype=np.float64) / orig_sr
        signal = (
            7000 * np.sin(2 * np.pi * 440 * t)
            + 2500 * np.sin(2 * np.pi * 2100 * t)
        ).astype(np.int16)
        resampler = StreamResampler(orig_sr, 16000)
        reference = resampler._resample_poly(
            signal.astype(np.float32), resampler.up, resampler.down
        )

        chunks: list[np.ndarray] = []
        block_lengths = (1000, 777, 2047)
        offset = 0
        block_index = 0
        while offset < len(signal):
            length = block_lengths[block_index % len(block_lengths)]
            chunks.append(resampler.resample(signal[offset : offset + length]))
            offset += length
            block_index += 1
        chunks.append(resampler.flush())
        streamed = np.concatenate(chunks)

        expected = (len(signal) * 16000 + orig_sr - 1) // orig_sr
        assert len(streamed) == len(reference) == expected
        np.testing.assert_allclose(streamed, reference, atol=2.0)
        assert resampler.flush().size == 0


def test_stream_resampler_vs_oneshot_consistency() -> None:
    """Сравнивает поток с one-shot по индексам, задержке и фильтровым стыкам."""
    from src.audio import StreamResampler

    for orig_sr in (44100, 48000):
        samples = int(orig_sr * 0.2) + 19
        t = np.arange(samples, dtype=np.float64) / orig_sr
        signal = (
            8000 * np.sin(2 * np.pi * 440 * t)
            + 3000 * np.sin(2 * np.pi * 1800 * t)
        ).astype(np.int16)
        resampler = StreamResampler(orig_sr, 16000)
        one_shot = resampler._resample_poly(
            signal.astype(np.float32), resampler.up, resampler.down
        )

        chunks = [
            resampler.resample(signal[start : start + 1000])
            for start in range(0, len(signal), 1000)
        ]
        chunks.append(resampler.flush())
        streamed = np.concatenate(chunks)

        # Выдача задержана на фильтровую историю, но индексы после flush
        # совпадают с one-shot; сравнение массивов разной длины было бы ошибкой.
        assert len(streamed) == len(one_shot)
        np.testing.assert_allclose(streamed, one_shot, atol=2.0)
        assert sum(map(len, chunks[:-1])) < len(one_shot)


def test_stream_resampler_rejects_aliasing() -> None:
    """Проверяет подавление тона выше выходного Найквиста."""
    from src.audio import StreamResampler

    for orig_sr in (44100, 48000):
        samples = int(orig_sr * 0.25)
        t = np.arange(samples, dtype=np.float64) / orig_sr
        signal = (12000 * np.sin(2 * np.pi * 12000 * t)).astype(np.int16)
        resampler = StreamResampler(orig_sr, 16000)
        chunks = [
            resampler.resample(signal[start : start + 777])
            for start in range(0, len(signal), 777)
        ]
        chunks.append(resampler.flush())
        streamed = np.concatenate(chunks).astype(np.float64)

        edge = int(np.ceil(resampler._future_len * resampler.up / resampler.down))
        interior = streamed[edge:-edge]
        assert np.sqrt(np.mean(interior**2)) < 400


def test_stream_resampler_direct_path() -> None:
    """Частота 16 кГц проходит без ресемплинга и изменения данных."""
    from src.audio import StreamResampler

    signal = np.arange(257, dtype=np.int16)
    result = StreamResampler(16000, 16000).resample(signal)
    np.testing.assert_array_equal(result, signal)
