"""Скрытая марковская модель вероятностей речи: 0 — тишина, 1 — речь.

Декодирование и обучение работают только с переданными массивами. Файлы,
аудиоустройства и модели распознавания этот модуль не открывает.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class HMM:
    """Модель P(s,x) = pi[s₀] ∏ A[sₜ₋₁,sₜ] ∏ Beta(xₜ; alphaₛ, betaₛ).

    A — квадратная стохастическая матрица, pi — начальное распределение,
    emissions — положительные параметры бета-распределений для состояний.
    Массивы копируются, обучение изменяет только данный экземпляр.
    """

    n_states: int
    A: NDArray[np.float64]
    pi: NDArray[np.float64]
    emissions: list[tuple[float, float]]

    def __post_init__(self) -> None:
        """Проверяет Aᵢⱼ ≥ 0, Σⱼ Aᵢⱼ = 1, Σᵢ piᵢ = 1 и alpha,beta > 0."""
        if not isinstance(self.n_states, int) or self.n_states < 1:
            raise ValueError("Число состояний должно быть положительным целым")
        self.A = np.array(self.A, dtype=np.float64, copy=True)
        self.pi = np.array(self.pi, dtype=np.float64, copy=True)
        self.emissions = list(self.emissions)
        if self.A.shape != (self.n_states, self.n_states):
            raise ValueError("Матрица переходов должна иметь форму (n_states, n_states)")
        if self.pi.shape != (self.n_states,):
            raise ValueError("Начальное распределение должно иметь форму (n_states,)")
        for values in (self.A, self.pi):
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Вероятности должны быть конечными и неотрицательными")
        if not np.allclose(self.A.sum(axis=1), 1.0, rtol=1e-12, atol=0.0):
            raise ValueError("Каждая строка матрицы переходов должна суммироваться в 1")
        if not math.isclose(float(self.pi.sum()), 1.0, rel_tol=1e-12):
            raise ValueError("Начальное распределение должно суммироваться в 1")
        params = np.asarray(self.emissions, dtype=np.float64)
        if params.shape != (self.n_states, 2) or not np.all(np.isfinite(params)):
            raise ValueError("Нужна пара конечных параметров эмиссии для каждого состояния")
        if np.any(params <= 0):
            raise ValueError("Параметры бета-распределений должны быть положительными")

    def _labels(self, labels: Sequence[int]) -> NDArray[np.int64]:
        """Проверяет метки: sₜ ∈ {0, …, n_states − 1}."""
        raw = np.asarray(labels)
        if raw.ndim != 1 or not np.all(np.isfinite(raw)):
            raise ValueError("Метки должны быть одномерной последовательностью целых")
        if np.any(raw < 0) or np.any(raw >= self.n_states) or np.any(raw != np.floor(raw)):
            raise ValueError("Метка состояния вне допустимого диапазона")
        return raw.astype(np.int64)

    def fit_transitions(self, labels: Sequence[int]) -> HMM:
        """Обучает Aᵢⱼ = (Nᵢⱼ + 1)/(Σⱼ Nᵢⱼ + |S|) за O(T + |S|²).

        Для одной последовательности piᵢ = (1 + [s₀=i])/(|S|+1).
        Пустая разметка даёт равномерные A и pi. Память O(T + |S|²).
        """
        states = self._labels(labels)
        counts = np.ones((self.n_states, self.n_states), dtype=np.float64)
        np.add.at(counts, (states[:-1], states[1:]), 1.0)
        self.A = counts / counts.sum(axis=1, keepdims=True)
        initial = np.ones(self.n_states, dtype=np.float64)
        if states.size:
            initial[states[0]] += 1.0
        self.pi = initial / initial.sum()
        return self

    def fit_emissions(self, obs: Sequence[float], labels: Sequence[int]) -> HMM:
        """Оценивает alpha,beta по состояниям; при SciPy используется ММП.

        Максимизируется Σ log Beta(xₜ; alphaₛ,betaₛ) при фиксированной области
        (0,1). Без SciPy используется метод моментов, а не ММП:
        k = mean*(1-mean)/var − 1, alpha = mean*k, beta = (1-mean)*k.
        Невозможные по моментам оценки ограничиваются k ≥ 1e-3; k ≤ 1e6
        ограничивает вырожденные оценки. Константная выборка оценивается
        методом моментов, так как конечного ММП в таком случае нет.
        Состояния без наблюдений сохраняют параметры. Границы 0 и 1
        сдвигаются на машинный epsilon. Память O(T), время O(T*I) для I
        итераций оптимизатора, без SciPy — O(T).
        """
        observations = self._observations(obs)
        states = self._labels(labels)
        if len(observations) != len(states):
            raise ValueError("Количество наблюдений и меток должно совпадать")
        try:
            from scipy.stats import beta as beta_distribution  # type: ignore[import-untyped]
        except ImportError:
            beta_distribution = None
        fitted = list(self.emissions)
        for state in range(self.n_states):
            values = observations[states == state]
            if not values.size:
                continue
            mean = float(values.mean())
            variance = float(values.var())
            concentration = min(1e6, max(1e-3, mean * (1.0 - mean) / max(variance, 1e-300) - 1.0))
            alpha, beta = mean * concentration, (1.0 - mean) * concentration
            if beta_distribution is not None and variance > 1e-12:
                alpha, beta, _, _ = beta_distribution.fit(values, floc=0.0, fscale=1.0)
            fitted[state] = (float(alpha), float(beta))
        self.emissions = fitted
        return self

    def _observations(self, obs: Sequence[float]) -> NDArray[np.float64]:
        """Проверяет 0 ≤ xₜ ≤ 1 и ограничивает x диапазоном [epsilon,1−epsilon]."""
        values = np.asarray(obs, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Наблюдения должны быть конечной одномерной последовательностью")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Вероятность речи должна принадлежать [0, 1]")
        eps = np.finfo(np.float64).eps
        return np.clip(values, eps, 1.0 - eps)

    def _emission_logs(self, obs: Sequence[float]) -> NDArray[np.float64]:
        """Вычисляет log f(x|s) = (alpha−1)log x + (beta−1)log(1−x) − log B."""
        values = self._observations(obs)
        logs = np.empty((len(values), self.n_states), dtype=np.float64)
        for state, (alpha, beta) in enumerate(self.emissions):
            log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
            logs[:, state] = (
                (alpha - 1.0) * np.log(values) + (beta - 1.0) * np.log1p(-values) - log_beta
            )
        return logs

    def viterbi(self, obs: Sequence[float]) -> tuple[list[int], float]:
        """Возвращает путь argmaxₛ P(s,obs) и его log-правдоподобие.

        Dₜ(j) = log f(xₜ|j) + maxᵢ(Dₜ₋₁(i) + log Aᵢⱼ).
        Логарифмы A и pi защищены np.log(np.maximum(x, 1e-300)).
        Время O(T*|S|²), память O(T*|S| + |S|²). Пустой вход → ([], 0.0).
        """
        emissions = self._emission_logs(obs)
        if not len(emissions):
            return [], 0.0
        log_a = np.log(np.maximum(self.A, 1e-300))
        scores = np.log(np.maximum(self.pi, 1e-300)) + emissions[0]
        back = np.zeros(emissions.shape, dtype=np.int64)
        for t in range(1, len(emissions)):
            candidates = scores[:, None] + log_a
            back[t] = np.argmax(candidates, axis=0)
            scores = candidates[back[t], np.arange(self.n_states)] + emissions[t]
        state = int(np.argmax(scores))
        log_prob = float(scores[state])
        path = [state]
        for t in range(len(emissions) - 1, 0, -1):
            state = int(back[t, state])
            path.append(state)
        path.reverse()
        return path, log_prob

    def forward_loglik(self, obs: Sequence[float]) -> float:
        """Считает log Σₛ P(s,obs) за O(T*|S|²), память O(T*|S| + |S|²).

        Fₜ(j) = log f(xₜ|j) + LSEᵢ(Fₜ₋₁(i) + log Aᵢⱼ),
        LSE(z) = max(z) + log Σ exp(z−max(z)). Пустой вход → 0.0.
        """
        emissions = self._emission_logs(obs)
        if not len(emissions):
            return 0.0
        log_a = np.log(np.maximum(self.A, 1e-300))
        scores = np.log(np.maximum(self.pi, 1e-300)) + emissions[0]
        for row in emissions[1:]:
            candidates = scores[:, None] + log_a
            maxima = candidates.max(axis=0)
            scores = row + maxima + np.log(np.exp(candidates - maxima).sum(axis=0))
        maximum = float(scores.max())
        return maximum + math.log(float(np.exp(scores - maximum).sum()))


def hysteresis(probs: Sequence[float], th_on: float, th_off: float) -> list[int]:
    """Реле: sₜ=1 при pₜ≥th_on, sₜ=0 при pₜ<th_off, иначе sₜ=sₜ₋₁.

    Начальное состояние 0, 0 ≤ th_off ≤ th_on ≤ 1. При равных порогах
    sₜ = [pₜ ≥ th_on]. Время O(T), память O(T).
    """
    if not 0.0 <= th_off <= th_on <= 1.0:
        raise ValueError("Нужны пороги 0 <= th_off <= th_on <= 1")
    state = 0
    result = []
    for prob in probs:
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError("Вероятность должна принадлежать [0, 1]")
        if prob >= th_on:
            state = 1
        elif prob < th_off:
            state = 0
        result.append(state)
    return result


def loop_width_for(eps: float, frame_sec: float, p_flip: float) -> float:
    """Временная ширина петли: frame_sec * max(1, ceil(log eps / log p_flip)).

    При независимых ошибочных кадрах вероятность серии из r переключений
    равна p_flip**r ≤ eps. Это длительность подтверждения в секундах,
    а не разность вероятностных порогов: без модели распределения такую
    разность из этих аргументов определить нельзя. При p_flip=0 нужен один
    кадр. Область: 0<eps<1, frame_sec>0, 0≤p_flip<1. Время и память O(1).
    """
    if not 0.0 < eps < 1.0 or not 0.0 <= p_flip < 1.0:
        raise ValueError("Нужны 0 < eps < 1 и 0 <= p_flip < 1")
    if not math.isfinite(frame_sec) or frame_sec <= 0:
        raise ValueError("Длительность кадра должна быть конечной и положительной")
    frames = 1 if p_flip == 0.0 else max(1, math.ceil(math.log(eps) / math.log(p_flip)))
    return frames * frame_sec
