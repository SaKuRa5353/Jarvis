"""Проверки пороговых критериев: ROC, Юден, Нейман-Пирсон, Ципф, M/M/1 и др.

Три функции сверены с независимым расчётом вручную (см. docstring):
    - fah:            FAH = 8.9e-6 · 3600 / 0.032 = 1.00125 ≈ 1.0
    - liu_layland:    U(3) = 3·(2^{1/3} − 1) = 0.77976…
    - mm1_wait:       Wq(2,5) = 2/(5·3) = 0.13333…
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.mathcore.thresholds import (
    alpha_for_fah,
    auc,
    confusion,
    cost_threshold,
    expected_tts_time,
    fah,
    liu_layland_bound,
    mm1_utilization,
    mm1_wait,
    nlms_excess_mse,
    nlms_update,
    np_threshold,
    optimal_first_chunk,
    percentile_ci_median,
    roc_curve,
    schedulable,
    wilcoxon_paired,
    youden_threshold,
    zipf_cache_size,
    zipf_hit_rate,
)


def test_confusion_perfect() -> None:
    assert confusion([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 0.5) == (2, 0, 0, 2)
    assert confusion([1, 0], [0.4, 0.6], 0.5) == (0, 1, 1, 0)


def test_confusion_invalid() -> None:
    with pytest.raises(ValueError):
        confusion([1, 0], [0.5], 0.5)
    with pytest.raises(ValueError):
        confusion([2, 0], [0.5, 0.5], 0.5)


def test_auc_ideal() -> None:
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    fpr, tpr, thrs = roc_curve(y, s)
    assert math.isclose(auc(fpr, tpr), 1.0, rel_tol=1e-9)
    assert isinstance(fpr, np.ndarray)
    assert isinstance(tpr, np.ndarray)
    assert isinstance(thrs, np.ndarray)
    # youden_threshold разделяет классы: порог между 0.2 и 0.8
    th = youden_threshold(y, s)
    assert 0.2 < th <= 0.8


def test_auc_random() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000).tolist()
    s = rng.random(5000).tolist()
    fpr, tpr, _ = roc_curve(y, s)
    assert abs(auc(fpr, tpr) - 0.5) < 0.05


def test_roc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.1, 0.5, 0.9])


def test_cost_threshold_asymmetry() -> None:
    y = [0, 0, 0, 1]
    s = [0.1, 0.3, 0.5, 0.7]
    th_fn = cost_threshold(y, s, c_fp=1.0, c_fn=10.0)
    th_fp = cost_threshold(y, s, c_fp=10.0, c_fn=1.0)
    assert th_fn <= th_fp
    with pytest.raises(ValueError):
        cost_threshold(y, s, c_fp=-1.0, c_fn=1.0)


def test_np_threshold_quantile() -> None:
    neg = [0.05, 0.1, 0.15, 0.9]
    tau = np_threshold(neg, alpha0=0.5)
    # FPR = доля отрицательных с score >= tau, не более 0.5
    fp = sum(1 for s in neg if s >= tau)
    assert fp / len(neg) <= 0.5 + 1e-12
    with pytest.raises(ValueError):
        np_threshold(neg, alpha0=1.5)
    with pytest.raises(ValueError):
        np_threshold([], 0.5)


def test_fah_known_value() -> None:
    # Ручной расчёт: 8.9e-6 · 3600 / 0.032 = 1.00125 ≈ 1.0
    assert math.isclose(fah(8.9e-6, 0.032), 1.0, rel_tol=1e-2)
    assert math.isclose(alpha_for_fah(1.0, 0.032), 8.9e-6, rel_tol=1e-2)


def test_fah_invalid() -> None:
    with pytest.raises(ValueError):
        fah(-0.1, 0.032)
    with pytest.raises(ValueError):
        fah(0.1, -0.032)
    with pytest.raises(ValueError):
        alpha_for_fah(-1.0, 0.032)


def test_zipf_hit_rate_and_cache_size() -> None:
    assert math.isclose(zipf_hit_rate(7, 7, s=1.0), 1.0, rel_tol=1e-9)
    assert zipf_hit_rate(20, 1000, s=1.0) < zipf_hit_rate(200, 1000, s=1.0)
    assert zipf_cache_size(1.0, 100, s=1.0) == 100
    assert zipf_cache_size(0.0, 100) == 0


def test_mm1_formulas() -> None:
    # Ручной расчёт: lam=2, mu=5 → Wq = 2/(5*3) = 0.13333…
    assert math.isclose(mm1_wait(2.0, 5.0), 2.0 / (5.0 * 3.0), rel_tol=1e-9)
    assert math.isclose(mm1_utilization(2.0, 5.0), 0.4, rel_tol=1e-9)
    # при lambda→mu время растёт
    assert mm1_wait(2.9, 3.0) < mm1_wait(2.99, 3.0)
    assert mm1_wait(5.0, 5.0) == math.inf
    assert mm1_wait(6.0, 5.0) == math.inf


def test_expected_tts_time() -> None:
    assert math.isclose(expected_tts_time(1.0, 0.1, 1.5), 0.1, rel_tol=1e-9)
    assert math.isclose(expected_tts_time(0.0, 0.1, 1.5), 1.5, rel_tol=1e-9)
    assert math.isclose(expected_tts_time(0.8, 0.1, 1.5), 0.8 * 0.1 + 0.2 * 1.5, rel_tol=1e-9)


def test_liu_layland_bound() -> None:
    assert math.isclose(liu_layland_bound(1), 1.0, rel_tol=1e-9)
    # Ручной расчёт: 3·(2^{1/3} − 1) = 0.779763…
    assert math.isclose(liu_layland_bound(3), 0.7798, rel_tol=1e-4)
    # убывает
    assert liu_layland_bound(10) < liu_layland_bound(3)
    # стремится к ln 2
    assert math.isclose(liu_layland_bound(10_000), math.log(2), rel_tol=1e-2)


def test_schedulable() -> None:
    # U=0.9 > 0.7798 → False
    tasks_high = [(0.3, 1.0), (0.3, 1.0), (0.3, 1.0)]
    ok, utilization = schedulable(tasks_high)
    assert ok is False
    assert math.isclose(utilization, 0.9, rel_tol=1e-9)
    # U=0.5 → True
    tasks_low = [(1.0, 4.0), (1.0, 4.0)]
    ok, utilization = schedulable(tasks_low)
    assert ok is True
    assert math.isclose(utilization, 0.5, rel_tol=1e-9)
    assert schedulable([]) == (True, 0.0)


def test_optimal_first_chunk() -> None:
    # v_llm=10, t_synth_per_char=0.05, t_fixed=0.5 → L*=10·0.5/(1−0.5)=10
    assert math.isclose(optimal_first_chunk(10.0, 0.05, 0.5), 10.0, rel_tol=1e-9)
    # v_llm·t_synth_per_char >= 1 → inf
    assert optimal_first_chunk(10.0, 0.1, 0.5) == math.inf


def test_nlms_update() -> None:
    h = np.array([1.0, -0.5])
    x = np.array([0.4, 0.9])
    # e == 0 не меняет h
    assert np.allclose(nlms_update(h, x, 0.0, 0.5), h)
    # mu → 0 почти не меняет h
    h_after = nlms_update(h, x, 0.7, 1e-9)
    assert np.allclose(h_after, h, atol=1e-6)
    with pytest.raises(ValueError):
        nlms_update(h, np.array([1.0]), 1.0, 0.1)


def test_nlms_excess_mse() -> None:
    j = nlms_excess_mse(0.5, 0.1, 4)
    assert math.isfinite(j) and j > 0
    assert nlms_excess_mse(2.0, 0.1, 4) == math.inf


def test_wilcoxon_identical_gives_zero() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    w, z = wilcoxon_paired(x, x)
    assert w == 0.0
    assert math.isnan(z)


def test_wilcoxon_shifted_large_z() -> None:
    a = [i * 0.1 for i in range(20)]
    b = [ai + 5.0 for ai in a]
    w, z = wilcoxon_paired(a, b)
    assert not math.isnan(z)
    assert abs(z) > 1.96


def test_wilcoxon_small_sample_returns_nan() -> None:
    # n < 20 → z = nan
    a = [1.0, 2.0, 3.0]
    b = [10.0, 11.0, 12.0]
    _, z = wilcoxon_paired(a, b)
    assert math.isnan(z)


def test_percentile_ci_contains_median() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000).tolist()
    lo, median, hi = percentile_ci_median(data, alpha=0.05)
    assert lo <= median <= hi
    with pytest.raises(ValueError):
        percentile_ci_median([1.0, 2.0], alpha=1.5)
    with pytest.raises(ValueError):
        percentile_ci_median([1.0], alpha=0.05)


def test_validation_branches() -> None:
    """Валидация входных данных во всех функциях — ожидаемые ValueError."""
    # zipf
    with pytest.raises(ValueError):
        zipf_hit_rate(-1, 100)
    with pytest.raises(ValueError):
        zipf_hit_rate(1, 0)
    with pytest.raises(ValueError):
        zipf_hit_rate(1, 100, s=-1.0)
    with pytest.raises(ValueError):
        zipf_cache_size(1.5, 100)
    with pytest.raises(ValueError):
        zipf_cache_size(0.5, 0)
    # mm1
    with pytest.raises(ValueError):
        mm1_utilization(1.0, 0.0)
    with pytest.raises(ValueError):
        mm1_utilization(-1.0, 5.0)
    with pytest.raises(ValueError):
        mm1_wait(-1.0, 5.0)
    with pytest.raises(ValueError):
        mm1_wait(1.0, 0.0)
    # expected_tts_time
    with pytest.raises(ValueError):
        expected_tts_time(1.5, 0.1, 1.0)
    with pytest.raises(ValueError):
        expected_tts_time(0.5, -0.1, 1.0)
    # optimal_first_chunk
    with pytest.raises(ValueError):
        optimal_first_chunk(0.0, 0.05, 0.5)
    with pytest.raises(ValueError):
        optimal_first_chunk(10.0, -0.1, 0.5)
    # nlms
    with pytest.raises(ValueError):
        nlms_update(np.array([1.0]), np.array([1.0]), 1.0, -0.1)
    # nlms_excess_mse
    with pytest.raises(ValueError):
        nlms_excess_mse(-0.1, 0.1, 4)
    with pytest.raises(ValueError):
        nlms_excess_mse(0.5, 0.1, 0)
    # wilcoxon несовпадающие длины
    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_np_threshold_tight_and_loose() -> None:
    """Нейман-Пирсон: FPR строго не превосходит α₀."""
    neg = [0.2, 0.4, 0.6, 0.8]
    # alpha0=0 → ни один не должен сработать → порог выше максимума
    tau = np_threshold(neg, alpha0=0.0)
    assert sum(1 for s in neg if s >= tau) == 0
    # alpha0=1 → срабатывают все → минимальный порог
    tau = np_threshold(neg, alpha0=1.0)
    assert sum(1 for s in neg if s >= tau) == len(neg)