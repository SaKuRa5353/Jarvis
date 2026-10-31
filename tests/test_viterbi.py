"""Проверки HMM на синтетике без моделей, микрофона и сети."""

from __future__ import annotations

import builtins
import itertools
import math
from unittest.mock import Mock

import numpy as np
import pytest

import config
from src.mathcore.viterbi import HMM, hysteresis, loop_width_for


@pytest.fixture
def model() -> HMM:
    """P(сохранения состояния)=0.98; средние эмиссий равны 0.1 и 0.9."""
    return HMM(
        2, np.array([[0.98, 0.02], [0.02, 0.98]]), np.array([0.5, 0.5]), [(2.0, 18.0), (18.0, 2.0)]
    )


def test_synthetic_recovery(model: HMM) -> None:
    rng = np.random.default_rng(42)
    labels = np.repeat([0, 1, 0, 1, 0], 200)
    obs = [float(rng.beta(*model.emissions[state])) for state in labels]
    path, log_prob = model.viterbi(obs)
    assert np.mean(np.asarray(path) == labels) >= 0.95
    assert model.forward_loglik(obs) >= log_prob


def test_fit_transitions(model: HMM) -> None:
    model.fit_transitions([0, 0, 1, 1, 1, 0])
    for row in model.A:
        assert math.isclose(float(row.sum()), 1.0, rel_tol=1e-12)
    assert np.all(model.A >= 0)
    assert np.all(model.pi >= 0)
    assert math.isclose(float(model.pi.sum()), 1.0, rel_tol=1e-12)
    # N00=1, N01=1: сглаженная строка равна (2/4, 2/4).
    assert math.isclose(float(model.A[0, 0]), 0.5, rel_tol=1e-9)
    model.fit_transitions([])
    assert np.allclose(model.A, 0.5, rtol=1e-9, atol=0.0)


def test_empty_and_single(model: HMM) -> None:
    assert model.viterbi([]) == ([], 0.0)
    assert math.isclose(model.forward_loglik([]), 0.0, rel_tol=1e-9)
    assert model.viterbi([0.9])[0] == [1]


def test_forward_matches_enumeration(model: HMM) -> None:
    obs = [0.1, 0.7, 0.9]
    scores = []
    # Независимый расчёт плотностей и суммы всех 2**3 путей.
    for path in itertools.product(range(2), repeat=len(obs)):
        score = float(model.pi[path[0]])
        for t, (value, state) in enumerate(zip(obs, path, strict=True)):
            alpha, beta = model.emissions[state]
            normalizer = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
            score *= normalizer * value ** (alpha - 1) * (1 - value) ** (beta - 1)
            if t:
                score *= model.A[path[t - 1], state]
        scores.append(score)
    assert math.isclose(model.forward_loglik(obs), math.log(sum(scores)), rel_tol=1e-9)
    assert math.isclose(model.viterbi(obs)[1], math.log(max(scores)), rel_tol=1e-9)


def test_endpoints_and_zero_transitions() -> None:
    model = HMM(2, np.eye(2), np.array([1.0, 0.0]), [(0.5, 0.5), (2.0, 2.0)])
    assert math.isfinite(model.viterbi([0.0, 1.0, 0.0])[1])
    assert math.isfinite(model.forward_loglik([0.0, 1.0, 0.0]))


def test_fit_emissions_mle(model: HMM) -> None:
    rng = np.random.default_rng(42)
    obs = rng.beta(3.0, 7.0, 3000).tolist()
    model.fit_emissions(obs, [0] * len(obs))
    alpha, beta = model.emissions[0]
    assert abs(alpha - 3.0) < 0.4
    assert abs(beta - 7.0) < 0.8
    assert model.emissions[1] == (18.0, 2.0)


def test_fit_emissions_without_scipy(model: HMM, monkeypatch) -> None:
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        """Имитирует отсутствие SciPy для проверки метода моментов."""
        if name.startswith("scipy"):
            raise ImportError("Проверка без SciPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    model.fit_emissions([0.2, 0.4], [0, 0])
    # mean=0.3, var=0.01, k=20 → alpha=6, beta=14.
    assert math.isclose(model.emissions[0][0], 6.0, rel_tol=1e-9)
    assert math.isclose(model.emissions[0][1], 14.0, rel_tol=1e-9)


def test_constant_emissions(model: HMM) -> None:
    model.fit_emissions([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
    assert np.all(np.isfinite(model.emissions))
    assert np.all(np.asarray(model.emissions) > 0)


def test_hysteresis() -> None:
    assert hysteresis([0.1, 0.5, 0.9, 0.2], 0.5, 0.5) == [0, 1, 1, 0]
    assert hysteresis([0.8, 0.6, 0.5, 0.4], 0.7, 0.3) == [1, 1, 1, 1]
    assert hysteresis([0.4, 0.5, 0.6], 0.7, 0.3) == [0, 0, 0]
    assert hysteresis([], 0.7, 0.3) == []


def test_loop_width() -> None:
    # 0.1**3=0.001: три кадра по 32 мс дают 0.096 секунды.
    assert math.isclose(loop_width_for(0.001, 0.032, 0.1), 0.096, rel_tol=1e-9)
    assert math.isclose(loop_width_for(0.001, 0.032, 0.0), 0.032, rel_tol=1e-9)


@pytest.mark.parametrize("obs", [[-0.1], [1.1], [math.nan], [math.inf], [[0.5]]])
def test_invalid_observations(model: HMM, obs) -> None:
    with pytest.raises(ValueError):
        model.viterbi(obs)


def test_invalid_training(model: HMM) -> None:
    for labels in ([2], [-1], [0.5], [math.nan], [[0]]):
        with pytest.raises(ValueError):
            model.fit_transitions(labels)
    with pytest.raises(ValueError):
        model.fit_emissions([0.2], [])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_states": 0},
        {"A": np.ones((2, 3))},
        {"pi": np.array([1.0])},
        {"A": np.array([[-1.0, 2.0], [0.5, 0.5]])},
        {"A": np.ones((2, 2))},
        {"pi": np.array([0.2, 0.2])},
        {"emissions": [(0.0, 1.0), (1.0, 1.0)]},
        {"emissions": [(1.0, 1.0)]},
    ],
)
def test_invalid_model(kwargs) -> None:
    params = dict(n_states=2, A=np.eye(2), pi=np.array([0.5, 0.5]), emissions=[(1.0, 1.0)] * 2)
    params.update(kwargs)
    with pytest.raises(ValueError):
        HMM(**params)


def test_invalid_hysteresis_and_loop_width() -> None:
    with pytest.raises(ValueError):
        hysteresis([0.5], 0.3, 0.7)
    with pytest.raises(ValueError):
        hysteresis([math.nan], 0.7, 0.3)
    for args in ((0.0, 0.032, 0.1), (0.1, 0.032, 1.0), (0.1, 0.0, 0.1)):
        with pytest.raises(ValueError):
            loop_width_for(*args)


@pytest.mark.hw
@pytest.mark.parametrize("decoder", ["threshold", "viterbi"])
def test_recorder_without_hardware(decoder: str, model: HMM, monkeypatch, tmp_path) -> None:
    """Оба декодера записывают одинаковые сильную речь и тишину без устройств."""
    from src import vad

    path = tmp_path / "vad_hmm.npz"
    np.savez(path, A=model.A, pi=model.pi, emissions=model.emissions)
    monkeypatch.setattr(config, "VAD_DECODER", decoder)
    monkeypatch.setattr(config, "VAD_HMM_MODEL", str(path))
    monkeypatch.setattr(
        vad, "SileroVAD", Mock(return_value=Mock(prob=Mock(side_effect=[0.99] * 3 + [0.01] * 3)))
    )
    monkeypatch.setattr(vad, "get_vosk_model", Mock(return_value=None))
    monkeypatch.setattr(
        vad,
        "KaldiRecognizer",
        Mock(return_value=Mock(FinalResult=Mock(return_value='{"text":"проверка"}'))),
    )
    monkeypatch.setattr(vad, "VAD_MIN_SPEECH_SEC", 0.3)
    monkeypatch.setattr(vad, "VAD_MIN_SILENCE_SEC", 0.2)
    sub = Mock(read=Mock(return_value=np.ones(10, dtype=np.int16)))
    recorder = vad.SpeechRecorder(sub, sample_rate=100, frame=10)
    audio, text = recorder.record()
    assert text == "проверка"
    assert len(audio) == 50
    assert sub.read.call_count == 5
    assert (recorder.hmm is None) == (decoder == "threshold")


@pytest.mark.hw
def test_threshold_does_not_load_hmm(monkeypatch) -> None:
    from src import vad

    monkeypatch.setattr(config, "VAD_DECODER", "threshold")
    monkeypatch.setattr(config, "VAD_HMM_MODEL", "несуществующий-файл.npz")
    monkeypatch.setattr(vad, "SileroVAD", Mock())
    recorder = vad.SpeechRecorder(Mock())
    assert recorder.hmm is None


@pytest.mark.hw
@pytest.mark.parametrize(
    "probs",
    [
        [0.5] * 10,
        [0.1, 0.5, 0.51, 0.9, 0.7, 0.5, 0.4],
        [0.9, 0.1, 0.9, 0.9, 0.1, 0.1],
        [0.9] * 10,
    ],
)
def test_threshold_matches_original(probs, monkeypatch) -> None:
    """Сверяет прежние >0.5, минимумы речи/тишины и предел записи по кадрам."""
    from src import vad

    monkeypatch.setattr(config, "VAD_DECODER", "threshold")
    monkeypatch.setattr(vad, "VAD_SPEECH_THRESHOLD", 0.5)
    monkeypatch.setattr(vad, "VAD_MIN_SPEECH_SEC", 0.3)
    monkeypatch.setattr(vad, "VAD_MIN_SILENCE_SEC", 0.2)
    monkeypatch.setattr(vad, "VAD_MAX_RECORD_SEC", len(probs) / 10)
    monkeypatch.setattr(vad, "SileroVAD", Mock(return_value=Mock(prob=Mock(side_effect=probs))))
    monkeypatch.setattr(vad, "get_vosk_model", Mock(return_value=None))
    monkeypatch.setattr(
        vad,
        "KaldiRecognizer",
        Mock(return_value=Mock(FinalResult=Mock(return_value='{"text":"проверка"}'))),
    )
    sub = Mock(read=Mock(return_value=np.ones(10, dtype=np.int16)))
    audio, text = vad.SpeechRecorder(sub, sample_rate=100, frame=10).record()
    started, speech, silence, buffered, consumed = False, 0, 0, 0, 0
    for prob in probs:
        consumed += 1
        if prob > 0.5:
            started = True
            silence = 0
            speech += 1
            buffered += 1
        elif started:
            silence += 1
            buffered += 1
            if silence >= 2:
                break
    assert sub.read.call_count == consumed
    assert len(audio) == (buffered * 10 if speech >= 3 else 0)
    assert text == ("проверка" if speech >= 3 else "")


@pytest.mark.hw
def test_viterbi_missing_model_and_invalid_decoder(monkeypatch) -> None:
    from src import vad

    monkeypatch.setattr(config, "VAD_DECODER", "неизвестный")
    with pytest.raises(ValueError):
        vad.SpeechRecorder(Mock())
    monkeypatch.setattr(config, "VAD_DECODER", "viterbi")
    monkeypatch.setattr(config, "VAD_HMM_MODEL", "несуществующий-файл.npz")
    with pytest.raises(FileNotFoundError):
        vad.SpeechRecorder(Mock())
