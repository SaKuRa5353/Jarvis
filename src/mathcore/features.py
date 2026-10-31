"""Акустические признаки: MFCC и динамическое выравнивание по времени.

Реализация только на numpy: rfft для спектра, матричная DCT-II, DP для DTW.
Модуль не открывает файлов, устройств и сети: только чистые функции.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hann(n: int) -> NDArray[np.float64]:
    """Окно Ханна: w[k] = 0.5 - 0.5*cos(2*pi*k/(n-1)), k = 0..n-1.

    Для n = 1 возвращается [1.0]. Время и память O(n).
    """
    if n < 1:
        raise ValueError("Длина окна должна быть положительной")
    if n == 1:
        return np.ones(1)
    k = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))


def hz_to_mel(f: float | np.ndarray) -> NDArray[np.float64]:
    """Частота в мелы: m = 2595 * log10(1 + f/700). Время O(len(f))."""
    values = np.asarray(f, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Частота должна быть неотрицательной")
    return 2595.0 * np.log10(1.0 + values / 700.0)


def mel_to_hz(m: float | np.ndarray) -> NDArray[np.float64]:
    """Мелы в частоту: f = 700 * (10**(m/2595) - 1). Время O(len(m))."""
    values = np.asarray(m, dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Мелы должны быть неотрицательными")
    return 700.0 * (10.0 ** (values / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 26,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> NDArray[np.float64]:
    """Треугольные мел-фильтры, матрица (n_mels, n_fft//2 + 1).

    Границы фильтров равномерны в мелах от f_min до f_max (по умолчанию
    sr/2). Каждый фильтр нормирован на пик: max = 1.
    Время O(n_mels * (n_fft//2+1)).
    """
    if sr <= 0 or n_fft < 2 or n_mels < 1:
        raise ValueError("Нужны sr > 0, n_fft >= 2, n_mels >= 1")
    top = sr / 2.0 if f_max is None else f_max
    if not 0.0 <= f_min < top:
        raise ValueError("Нужны 0 <= f_min < f_max <= sr/2")
    edges = mel_to_hz(hz_to_mel(np.linspace(f_min, top, n_mels + 2)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bank = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = edges[index], edges[index + 1], edges[index + 2]
        rise = (freqs - left) / max(center - left, 1e-12)
        fall = (right - freqs) / max(right - center, 1e-12)
        triangle = np.clip(np.minimum(rise, fall), 0.0, None)
        peak = triangle.max()
        if peak > 0:
            triangle /= peak
        bank[index] = triangle
    return bank


def dct_ii(x: ArrayLike, n_out: int) -> NDArray[np.float64]:
    """DCT-II: y[k] = sum_n x[n] * cos(pi*k*(2n+1)/(2*N)), k = 0..n_out-1.

    Без нормировочных множителей. Время O(n_out * N), память O(n_out * N).
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Нужен непустой одномерный массив")
    if n_out < 1:
        raise ValueError("n_out должен быть положительным")
    size = values.size
    k = np.arange(n_out, dtype=np.float64)[:, None]
    n = np.arange(size, dtype=np.float64)[None, :]
    return np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * size)) @ values


def mfcc(
    frame: ArrayLike,
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 26,
    n_coef: int = 13,
) -> NDArray[np.float64]:
    """MFCC одного кадра длиной до n_fft отсчётов.

    Кадр умножается на окно Ханна, дополняется нулями до n_fft,
    берётся спектр мощности |rfft|^2, мел-фильтры, логарифм энергии
    (floor 1e-10) и DCT-II. Возвращаются первые n_coef коэффициентов.
    Время O(n_fft log n_fft + n_mels*(n_fft/2) + n_coef*n_mels).
    """
    if sr <= 0 or n_fft < 2 or n_mels < 1 or n_coef < 1:
        raise ValueError("Некорректные параметры MFCC")
    samples = np.asarray(frame, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Кадр должен быть непустой одномерной последовательностью")
    if samples.size > n_fft:
        raise ValueError("Длина кадра не должна превышать n_fft")
    windowed = samples * hann(samples.size)
    padded = np.zeros(n_fft, dtype=np.float64)
    padded[: samples.size] = windowed
    power = np.abs(np.fft.rfft(padded)) ** 2
    energies = mel_filterbank(sr, n_fft, n_mels) @ power
    return dct_ii(np.log(np.maximum(energies, 1e-10)), n_coef)


def mfcc_sequence(
    audio: ArrayLike,
    sr: int = 16000,
    frame: int = 512,
    hop: int = 160,
) -> NDArray[np.float64]:
    """Последовательность MFCC: матрица (T, 13), T = 1 + (len - frame)//hop.

    Кадры без перекрытия за пределами hop берутся со сдвигом hop.
    Для аудио короче frame возвращается пустая матрица (0, 13).
    Время O(T * стоимость mfcc).
    """
    if frame < 1 or hop < 1:
        raise ValueError("Длина кадра и сдвиг должны быть положительными")
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("Аудио должно быть одномерным")
    if samples.size < frame:
        return np.zeros((0, 13), dtype=np.float64)
    total = 1 + (samples.size - frame) // hop
    return np.stack([mfcc(samples[i * hop : i * hop + frame], sr, frame, 26, 13) for i in range(total)])


def deltas(coeffs: ArrayLike, k: int = 2) -> NDArray[np.float64]:
    """Дельта-признаки: d_t = sum_{n=1..k} n*(c_{t+n} - c_{t-n}) / (2*sum n^2).

    Индексы за границами зажимаются к краевым кадрам (mode=edge),
    форма результата равна форме входа. Время O(T*C*k).
    """
    if k < 1:
        raise ValueError("Радиус дельты должен быть >= 1")
    values = np.asarray(coeffs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Нужна матрица коэффициентов (T, C)")
    total, _ = values.shape
    if total == 0:
        return values.copy()
    padded = np.pad(values, ((k, k), (0, 0)), mode="edge")
    result = np.zeros_like(values)
    weight = 2.0 * sum(n * n for n in range(1, k + 1))
    for n in range(1, k + 1):
        result += n * (padded[k + n : k + n + total] - padded[k - n : k - n + total])
    return result / weight


def _prepare(series: ArrayLike) -> NDArray[np.float64]:
    """Приводит ряд к матрице (T, C); одномерный ряд трактуется как (T, 1)."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("Последовательности должны быть одно- или двумерными")
    return values


def dtw(
    x: ArrayLike,
    y: ArrayLike,
    window: int | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Динамическое выравнивание: нормированное расстояние и путь.

    D[i,j] = d(x_i, y_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1]),
    где d — евклидово расстояние между кадрами. Возвращается
    D(n,m)/(n+m) и путь 0-индексов от (0,0) до (n-1,m-1).
    Окно Сако-Чибы: заполняются только клетки |i-j| <= window.
    Если путь недостижим (window < |n-m|) — (inf, []).
    Пустые ряды: оба пустых → (0.0, []), один пустой → (inf, []).
    Время O(n*m) без окна и O(n*window) с окном, память O(n*m).
    """
    first, second = _prepare(x), _prepare(y)
    n, m = len(first), len(second)
    if window is not None and window < 0:
        raise ValueError("Ширина окна должна быть неотрицательной")
    width = max(n, m) if window is None else window
    if n == 0 and m == 0:
        return 0.0, []
    if n == 0 or m == 0:
        return math.inf, []
    if width < abs(n - m):
        return math.inf, []
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - width), min(m, i + width) + 1):
            step = float(np.linalg.norm(first[i - 1] - second[j - 1]))
            acc[i, j] = step + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    if math.isinf(float(acc[n, m])):
        return math.inf, []
    path: list[tuple[int, int]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        path.append((i - 1, j - 1))
        options = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        i, j = min(options, key=lambda cell: acc[cell[0], cell[1]])
    path.reverse()
    return float(acc[n, m]) / (n + m), path