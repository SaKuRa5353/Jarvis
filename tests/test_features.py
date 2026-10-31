"""Проверки MFCC и DTW на синтетике: только numpy, без устройств и файлов."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.mathcore.features import (
    dct_ii,
    deltas,
    dtw,
    hann,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    mfcc,
    mfcc_sequence,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_hann_symmetry_and_bounds() -> None:
    window = hann(8)
    assert window[0] == 0.0 and window[-1] == 0.0
    assert math.isclose(window[3], 0.5 - 0.5 * math.cos(2 * math.pi * 3 / 7), rel_tol=1e-12)
    assert np.allclose(window, window[::-1])
    assert hann(1) == np.ones(1)


def test_mel_roundtrip() -> None:
    values = np.array([0.0, 100.0, 440.0, 1000.0, 8000.0])
    assert np.allclose(hz_to_mel(mel_to_hz(values)), values, rtol=1e-9, atol=0.0)
    assert math.isclose(float(hz_to_mel(1000.0)), 1000.0, rel_tol=1e-2)


def test_mel_filterbank_shape_and_normalization() -> None:
    bank = mel_filterbank(16000, 512, n_mels=26)
    assert bank.shape == (26, 257)
    assert np.all(bank >= 0)
    for row in bank:
        if row.sum() > 0:
            assert math.isclose(float(row.max()), 1.0, rel_tol=1e-9)


def test_mel_filterbank_invalid() -> None:
    with pytest.raises(ValueError):
        mel_filterbank(16000, 512, f_min=9000.0)
    for args in ((0, 512), (16000, 1), (16000, 512, 0)):
        with pytest.raises(ValueError):
            mel_filterbank(*args)


def test_dct_ii_matches_direct_sum() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    n_out = 4
    expected = np.array([
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * len(x))) for n in range(len(x)))
        for k in range(n_out)
    ])
    assert np.allclose(dct_ii(x, n_out), expected, rtol=1e-12)
    assert len(dct_ii(x, 2)) == 2
    with pytest.raises(ValueError):
        dct_ii(np.ones((2, 2)), 2)


def test_mfcc_concentrates_tone_energy(rng) -> None:
    t = np.arange(512) / 16000
    tone = np.sin(2 * np.pi * 440 * t)
    noise = rng.uniform(-1, 1, 512)
    tone_coeffs = mfcc(tone)
    noise_coeffs = mfcc(noise)
    # У тона спектр сосредоточен в низких мелах, поэтому AC-коэффициенты C1..C3
    # доминируют над C4..C12. У шума мел-энергии ровные, DCT почти нулевой.
    def head_to_tail(coeffs):
        return abs(coeffs[1:4]).sum() / (abs(coeffs[4:]).sum() + 1e-12)

    assert head_to_tail(tone_coeffs) > head_to_tail(noise_coeffs)


def test_mfcc_sequence_shape() -> None:
    audio = np.zeros(2048)
    matrix = mfcc_sequence(audio, frame=512, hop=160)
    assert matrix.shape == (1 + (2048 - 512) // 160, 13)
    assert mfcc_sequence(np.zeros(100)).shape == (0, 13)
    with pytest.raises(ValueError):
        mfcc_sequence(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        mfcc_sequence(np.ones(512), frame=0)
    with pytest.raises(ValueError):
        deltas(np.ones(5), k=0)


def test_dtw_path_structure(rng) -> None:
    """Путь монотонно растёт и покрывает обе последовательности."""
    x, y = rng.uniform(size=(7, 3)), rng.uniform(size=(9, 3))
    _, path = dtw(x, y)
    assert len(path) > 0
    for (i1, j1), (i2, j2) in zip(path, path[1:], strict=False):
        assert i2 >= i1 and j2 >= j1
        assert i2 - i1 <= 1 and j2 - j1 <= 1
    assert all(0 <= i < 7 and 0 <= j < 9 for i, j in path)


def test_deltas_preserves_shape() -> None:
    matrix = np.random.default_rng(1).uniform(size=(10, 13))
    assert deltas(matrix).shape == matrix.shape
    assert deltas(matrix, k=1).shape == matrix.shape
    # Константный ряд имеет нулевые дельты.
    assert np.allclose(deltas(np.ones((5, 3))), 0.0)


def test_dtw_identity_and_symmetry(rng) -> None:
    for _ in range(10):
        x = rng.uniform(size=(15, 3))
        distance, path = dtw(x, x)
        assert math.isclose(distance, 0.0, rel_tol=1e-9)
        assert dtw(x, x)[0] == 0.0
        y = rng.uniform(size=(17, 3))
        assert math.isclose(dtw(x, y)[0], dtw(y, x)[0], rel_tol=1e-12)
        assert path[0] == (0, 0) and path[-1] == (14, 14)


def test_dtw_triangle_inequality(rng) -> None:
    for _ in range(50):
        x, y, z = (rng.uniform(size=(10, 2)) for _ in range(3))
        assert dtw(x, z)[0] <= dtw(x, y)[0] + dtw(y, z)[0] + 1e-9


def test_dtw_stretch_cheaper_than_pointwise(rng) -> None:
    base = rng.uniform(size=(12, 2))
    stretched = np.repeat(base, 2, axis=0)  # каждый кадр продублирован
    other = rng.uniform(size=(12, 2))
    stretched_distance = dtw(base, stretched)[0]
    euclid = float(np.linalg.norm(base - other, axis=1).mean())
    assert stretched_distance < euclid


def test_dtw_window(rng) -> None:
    x, y = rng.uniform(size=(12, 2)), rng.uniform(size=(12, 2))
    assert math.isclose(dtw(x, y, window=11)[0], dtw(x, y)[0], rel_tol=1e-12)
    small = dtw(x, y, window=2)[0]
    assert small >= dtw(x, y)[0] - 1e-12
    shifted = rng.uniform(size=(20, 2))
    distance, path = dtw(x, shifted, window=1)
    assert distance == math.inf and path == []


def test_dtw_empty_and_invalid() -> None:
    assert dtw([], []) == (0.0, [])
    assert dtw([1.0], [])[0] == math.inf
    with pytest.raises(ValueError):
        dtw([1.0], [1.0], window=-1)
    with pytest.raises(ValueError):
        dtw(np.ones((2, 2, 2)), [1.0])


def test_edge_cases() -> None:
    with pytest.raises(ValueError):
        hann(0)
    with pytest.raises(ValueError):
        hz_to_mel(-1.0)
    with pytest.raises(ValueError):
        mel_to_hz(-0.5)
    with pytest.raises(ValueError):
        dct_ii(np.array([1.0]), 0)
    with pytest.raises(ValueError):
        mfcc(np.ones(600), n_fft=512)
    with pytest.raises(ValueError):
        deltas(np.ones(5), k=0)
    with pytest.raises(ValueError):
        mfcc(np.array([]))
    with pytest.raises(ValueError):
        mfcc(np.ones(16), sr=0)
    with pytest.raises(ValueError):
        dct_ii(np.array([]), 1)