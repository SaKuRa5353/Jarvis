"""Пороговые критерии и прикладная статистика для конвейера ассистента.

Чистые функции без зависимостей от остального ``src/``. Разрешены stdlib и
``numpy``. Каждая функция: формула в docstring, типизация, оценка сложности.

Разделы:
    1. Бинарная классификация, ROC, AUC, пороги (Юден, стоимость, Нейман-Пирсон)
    2. Ложные срабатывания в час (FAH)
    3. Закон Ципфа — модель кэша TTS
    4. Очередь M/M/1 — задержка и загрузка
    5. Liu-Layland — граница выполнимости для реального времени
    6. Стриминг TTS — оптимальный первый чанк
    7. NLMS — адаптивная фильтрация (один шаг и избыточная ошибка)
    8. Непараметрическая статистика — Уилкоксон и CI медианы
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# ───────────────────────── 1. КЛАССИФИКАЦИЯ / ROC ─────────────────────────

def confusion(y_true: Sequence[int], scores: Sequence[float], thr: float) -> tuple[int, int, int, int]:
    """Бинарная матрица ошибок для одного порога ``thr``.

    Предсказание ŷᵢ = 1 если ``scores[i] >= thr``, иначе 0. Формулы:
        TP = Σ 1[yᵢ=1 ∧ ŷᵢ=1],  FP = Σ 1[yᵢ=0 ∧ ŷᵢ=1],
        FN = Σ 1[yᵢ=1 ∧ ŷᵢ=0],  TN = Σ 1[yᵢ=0 ∧ ŷᵢ=0].
    Возвращает кортеж ``(TP, FP, FN, TN)``. Сложность O(n), память O(1).
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true и scores должны иметь одинаковую длину")
    tp = fp = fn = tn = 0
    for y, s in zip(y_true, scores, strict=False):
        if y not in (0, 1):
            raise ValueError("метки должны быть из {0, 1}")
        pred = 1 if s >= thr else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def roc_curve(
    y_true: Sequence[int], scores: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ROC-кривая: (FPR, TPR, пороги) при варьировании порога.

    FPR(τ) = P(score ≥ τ | y=0), TPR(τ) = P(score ≥ τ | y=1). Точки
    дополняются концами (FPR,TPR) = (0,0) и (1,1). Пороги отсортированы
    по убыванию, возвращаемый массив упорядочен по возрастанию FPR.
    Сложность O(n log n) — сортировка.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true и scores должны иметь одинаковую длину")
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("нужны оба класса для ROC-кривой")

    pairs = sorted(zip(scores, y_true, strict=False), key=lambda p: (-p[0], p[1]))
    fpr: list[float] = [0.0]
    tpr: list[float] = [0.0]
    thrs: list[float] = [pairs[0][0] + 1.0]

    tp = fp = 0
    prev: float | None = None
    for score, y in pairs:
        if prev is not None and score != prev:
            fpr.append(fp / n_neg)
            tpr.append(tp / n_pos)
            thrs.append(prev)
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev = score

    fpr.append(fp / n_neg)
    tpr.append(tp / n_pos)
    thrs.append(prev if prev is not None else 0.0)
    fpr.append(1.0)
    tpr.append(1.0)
    thrs.append(-math.inf)
    return (
        np.asarray(fpr, dtype=np.float64),
        np.asarray(tpr, dtype=np.float64),
        np.asarray(thrs, dtype=np.float64),
    )


def auc(fpr: Sequence[float], tpr: Sequence[float]) -> float:
    """Площадь под ROC-кривой по правилу трапеций.

    AUC = Σᵢ (fprᵢ₊₁ − fprᵢ) · (tprᵢ₊₁ + tprᵢ) / 2.
    Ожидается возрастающая ``fpr``. Сложность O(k), k — число точек.
    """
    if len(fpr) != len(tpr) or len(fpr) < 2:
        raise ValueError("нужны как минимум две точки и равные длины массивов")
    area = 0.0
    for i in range(len(fpr) - 1):
        width = fpr[i + 1] - fpr[i]
        height = (tpr[i + 1] + tpr[i]) / 2.0
        area += width * height
    return area


def youden_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Оптимальный порог по критерию Юдена: argmax_c J(τ) = TPR − FPR.

    Возвращает сам порог ``τ``. Сложность O(n log n) из-за roc_curve.
    """
    fpr, tpr, thrs = roc_curve(y_true, scores)
    best = 0
    best_j = -math.inf
    for i in range(len(thrs)):
        j = tpr[i] - fpr[i]
        if j > best_j:
            best_j = j
            best = i
    return float(thrs[best])


def cost_threshold(
    y_true: Sequence[int], scores: Sequence[float], c_fp: float, c_fn: float
) -> float:
    """Порог, минимизирующий стоимость ошибок C(τ) = c_fp·FP + c_fn·FN.

    Возвращает порог ``τ``. Сложность O(n log n) из-за roc_curve.
    """
    if c_fp < 0 or c_fn < 0:
        raise ValueError("стоимости ошибок должны быть неотрицательными")
    fpr, tpr, thrs = roc_curve(y_true, scores)
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    best_t = thrs[0]
    best_c = math.inf
    for i, th in enumerate(thrs):
        fp = n_neg * fpr[i]
        fn = n_pos * (1.0 - tpr[i])
        cost = c_fp * fp + c_fn * fn
        if cost < best_c:
            best_c = cost
            best_t = th
    return float(best_t)


def np_threshold(scores_neg: Sequence[float], alpha0: float) -> float:
    """Порог Неймана-Пирсона на отрицательной выборке (квантиль).

    Возвращает минимальный ``τ``, при котором FPR = |{s ≥ τ}| / n ≤ alpha0.
    Реализация: перебор уникальных значений по возрастанию с бинарным
    поиском числа срабатываний. Сложность O(n log n).
    """
    if not 0.0 <= alpha0 <= 1.0:
        raise ValueError("alpha0 должен быть в [0, 1]")
    if not scores_neg:
        raise ValueError("scores_neg не может быть пустым")
    asc = sorted(scores_neg)
    n = len(asc)
    for tau in sorted(set(asc)):
        fp = n - bisect.bisect_left(asc, tau)
        if fp / n <= alpha0 + 1e-12:
            return float(tau)
    return asc[-1] + 1.0


# ───────────────────────── 2. FAH ─────────────────────────

def fah(alpha: float, frame_sec: float) -> float:
    """Ложные срабатывания в час: FAH = α · 3600 / frame_sec.

    ``alpha`` — вероятность ложного срабатывания на одном кадре,
    ``frame_sec`` — длительность кадра в секундах. Сложность O(1).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha должен быть в [0, 1]")
    if frame_sec <= 0:
        raise ValueError("frame_sec должен быть положительным")
    return alpha * 3600.0 / frame_sec


def alpha_for_fah(fah_target: float, frame_sec: float) -> float:
    """Обратная к FAH: α = FAH · frame_sec / 3600. Сложность O(1)."""
    if fah_target < 0:
        raise ValueError("fah_target должен быть неотрицательным")
    if frame_sec <= 0:
        raise ValueError("frame_sec должен быть положительным")
    return fah_target * frame_sec / 3600.0


# ───────────────────────── 3. ЦИПФ ─────────────────────────

def _zipf_sum(m: int, s: float) -> float:
    """Сумма Σ_{k=1..m} k^{-s} (обобщённое гармоническое число H_m(s))."""
    return sum(k ** (-s) for k in range(1, m + 1))


def zipf_hit_rate(n_cache: int, m_total: int, s: float = 1.0) -> float:
    """Вероятность попадания в кэш размера ``n_cache`` при запросах по Ципфу.

    P(hit) = Σ_{k=1..n} k^{-s} / Σ_{k=1..M} k^{-s}; при n ≥ M — 1.0.
    ``m_total`` — общее число уникальных фраз, ``s`` — показатель Ципфа.
    Сложность O(M).
    """
    if n_cache < 0 or m_total <= 0 or s < 0:
        raise ValueError("нужны n_cache ≥ 0, m_total > 0, s ≥ 0")
    if n_cache >= m_total:
        return 1.0
    return _zipf_sum(n_cache, s) / _zipf_sum(m_total, s)


def zipf_cache_size(target_hit: float, m_total: int, s: float = 1.0) -> int:
    """Минимальный размер кэша для hit rate ≥ ``target_hit``.

    n* = min{n : zipf_hit_rate(n, M, s) ≥ target}. Сложность O(M).
    """
    if not 0.0 <= target_hit <= 1.0:
        raise ValueError("target_hit должен быть в [0, 1]")
    if m_total <= 0 or s < 0:
        raise ValueError("нужны m_total > 0, s ≥ 0")
    if target_hit == 0.0:
        return 0
    denominator = _zipf_sum(m_total, s)
    cum = 0.0
    for n in range(1, m_total + 1):
        cum += n ** (-s)
        if cum / denominator >= target_hit - 1e-12:
            return n
    return m_total


# ───────────────────────── 4. M/M/1 ─────────────────────────

def mm1_utilization(lmbda: float, mu: float) -> float:
    """Коэффициент загрузки ρ = λ/μ очереди M/M/1. Сложность O(1)."""
    if mu <= 0 or lmbda < 0:
        raise ValueError("нужны μ > 0 и λ ≥ 0")
    return lmbda / mu


def mm1_wait(lmbda: float, mu: float) -> float:
    """Среднее время ожидания в очереди M/M/1: Wq = λ / (μ·(μ−λ)).

    При λ ≥ μ очередь нестабильна — возвращает ``math.inf``. Сложность O(1).
    """
    if mu <= 0 or lmbda < 0:
        raise ValueError("нужны μ > 0 и λ ≥ 0")
    if lmbda >= mu:
        return math.inf
    return lmbda / (mu * (mu - lmbda))


def expected_tts_time(h: float, t_wav: float, t_synth: float) -> float:
    """Ожидаемое время ответа TTS при кэшировании.

    Промах кэша (вероятность 1−h) стоит синтез ``t_synth``, попадание
    (вероятность h) — воспроизведение готового WAV ``t_wav``:
        E[T] = h · t_wav + (1 − h) · t_synth.
    ``h`` — hit rate кэша. Сложность O(1).
    """
    if not 0.0 <= h <= 1.0:
        raise ValueError("h должен быть в [0, 1]")
    if t_wav < 0 or t_synth < 0:
        raise ValueError("времена должны быть неотрицательными")
    return h * t_wav + (1.0 - h) * t_synth


# ───────────────────────── 5. LIU-LAYLAND ─────────────────────────

def liu_layland_bound(n: int) -> float:
    """Верхняя граница загрузки CPU по Liu-Layland: U(n) = n·(2^{1/n} − 1).

    При n → ∞ сходится к ln 2 ≈ 0.6931. Сложность O(1).
    """
    if n <= 0:
        raise ValueError("число задач должно быть ≥ 1")
    return n * (2.0 ** (1.0 / n) - 1.0)


def schedulable(tasks: Sequence[tuple[float, float]]) -> tuple[bool, float]:
    """Проверка выполнимости по Liu-Layland.

    ``tasks`` — пары (Cᵢ, Tᵢ): время исполнения и период. Загрузка
    U = Σ Cᵢ/Tᵢ сравнивается с границей U_bound(n). Возвращает
    ``(выполнимо, U)``. Сложность O(n).
    """
    if any(c < 0 or t <= 0 for c, t in tasks):
        raise ValueError("нужны C ≥ 0 и T > 0")
    if not tasks:
        return True, 0.0
    utilization = sum(c / t for c, t in tasks)
    return utilization <= liu_layland_bound(len(tasks)), utilization


# ───────────────────────── 6. СТРИМИНГ TTS ─────────────────────────

def optimal_first_chunk(v_llm: float, t_synth_per_char: float, t_fixed: float) -> float:
    """Оптимальный размер первого чанка для стриминга TTS.

    ``v_llm`` — скорость генерации LLM (символ/с), ``t_synth_per_char`` —
    время синтеза на символ (с), ``t_fixed`` — фиксированная задержка запуска.
    Чтобы к моменту окончания озвучки первого чанка накопился следующий текст:
        L* = v_llm · t_fixed / (1 − v_llm · t_synth_per_char).
    При v_llm·t_synth_per_char ≥ 1 синтез не успевает за генерацией —
    возвращает ``math.inf``. Сложность O(1).
    """
    if v_llm <= 0 or t_synth_per_char < 0 or t_fixed < 0:
        raise ValueError("нужны v_llm > 0, t_synth_per_char ≥ 0, t_fixed ≥ 0")
    denom = 1.0 - v_llm * t_synth_per_char
    if denom <= 0:
        return math.inf
    return v_llm * t_fixed / denom


# ───────────────────────── 7. NLMS ─────────────────────────

def nlms_update(
    h: NDArray[np.float64], x: NDArray[np.float64], e: float, mu: float, eps: float = 1e-8
) -> NDArray[np.float64]:
    """Один шаг нормализованного LMS.

        h' = h + μ · e · x / (xᵀx + ε).
    ``e`` — скалярная ошибка d − hᵀx, ``μ`` — шаг адаптации. Возвращает
    новый вектор, не изменяя входной. Сложность O(d), d = len(h).
    """
    h_arr = np.asarray(h, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if h_arr.shape != x_arr.shape:
        raise ValueError("h и x должны иметь одинаковую форму")
    if mu < 0 or eps < 0:
        raise ValueError("нужны μ ≥ 0 и ε ≥ 0")
    norm = float(x_arr @ x_arr) + eps
    return h_arr + (mu * e / norm) * x_arr


def nlms_excess_mse(mu: float, sigma_e2: float, L: int) -> float:
    """Верхняя оценка избыточной MSE NLMS в установившемся режиме.

    J_ex = (μ / (2 − μ)) · σ_e² · L — доминирующая оценка, линейная по
    числу коэффициентов ``L`` и дисперсии шума ``σ_e²``. При μ ≥ 2 фильтр
    неустойчив — возвращает ``math.inf``. Сложность O(1).
    """
    if mu < 0 or sigma_e2 < 0 or L <= 0:
        raise ValueError("нужны μ ≥ 0, σ_e2 ≥ 0, L ≥ 1")
    if mu >= 2.0:
        return math.inf
    return (mu / (2.0 - mu)) * sigma_e2 * L


# ───────────────────────── 8. СТАТИСТИКА ─────────────────────────

def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Знаковый ранговый тест Уилкоксона для связанных выборок.

    Возвращает ``(W, z)``, где W — сумма рангов положительных разностей,
    z — нормальная аппроксимация z = (W − μ_W)/σ_W:
        μ_W = n(n+1)/4, σ_W² = n(n+1)(2n+1)/24.
    Нулевые разности dᵢ = 0 исключаются из n. При n < 20 точный тест
    необходим, поэтому возвращается ``z = nan``. Связки получают средние
    ранги; поправка на связи в оценке дисперсии не вводится.
    Сложность O(n log n).
    """
    if len(a) != len(b):
        raise ValueError("выборки должны иметь одинаковую длину")
    diffs = [ai - bi for ai, bi in zip(a, b, strict=False) if ai != bi]
    n = len(diffs)
    if n == 0:
        return 0.0, math.nan
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(diffs[order[j]]) == abs(diffs[order[i]]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    if n < 20:
        return w_plus, math.nan
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return w_plus, (w_plus - mean) / sd


def percentile_ci_median(
    x: Sequence[float], alpha: float = 0.05
) -> tuple[float, float, float]:
    """Непараметрический доверительный интервал медианы (sign test).

    Для отсортированной выборки X₍₁₎,…,X₍ₙ₎ границы — порядковые
    статистики [X₍ᵣ₎, X₍ₙ₋ᵣ₊₁₎]. Уровень доверия = 1 − 2·P(B ≤ r−1),
    где B ~ Bin(n, 0.5). Выбирается наибольшее r с P(B ≤ r−1) ≤ α/2.
    Возвращает ``(нижняя граница, медиана, верхняя граница)``.
    Сложность O(n log n).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha должен быть в (0, 1)")
    if len(x) < 2:
        raise ValueError("нужно минимум два наблюдения")
    xs = sorted(x)
    n = len(xs)
    median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0

    target = alpha / 2.0
    denom = 2.0 ** n
    cdf = 0.0
    max_j = -1
    for j in range(n):
        cdf += math.comb(n, j) / denom
        if cdf <= target + 1e-12:
            max_j = j
        else:
            break
    if max_j < 0:
        lo_idx, hi_idx = 0, n - 1
    else:
        r = max_j + 1
        r = min(r, (n + 1) // 2)
        lo_idx = r - 1
        hi_idx = n - r
        if hi_idx < lo_idx:
            hi_idx = lo_idx
    return xs[lo_idx], median, xs[hi_idx]