"""Проверки DTW-детектора обращения на синтетике, без микрофона."""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.mathcore.features import mfcc_sequence
from src.wakeword_dtw import WakewordDTW


@pytest.fixture
def refs(tmp_path):
    """Три эталона из синтетического «тембра»: тон + шумовая форма."""
    rng = np.random.default_rng(42)
    t = np.arange(5120) / 16000
    audio = (0.4 * np.sin(2 * np.pi * 160 * t) + 0.2 * np.sin(2 * np.pi * 2400 * t)
             + 0.1 * rng.uniform(-1, 1, len(t))) * 32767
    directory = tmp_path / "refs"
    directory.mkdir()
    for index in range(3):
        np.save(directory / f"ref_{index:03d}.npy", mfcc_sequence(audio))
    return directory, audio


def test_score_and_match(refs) -> None:
    directory, audio = refs
    detector = WakewordDTW(refs_dir=directory)
    assert detector.score(audio) == pytest.approx(0.0, abs=1e-9)
    assert detector.is_match(audio)
    assert detector.is_match(audio, threshold=1e-9)


def test_score_grows_with_distortion(refs) -> None:
    directory, audio = refs
    detector = WakewordDTW(refs_dir=directory)
    noise = np.random.default_rng(7).uniform(-32767, 32767, len(audio))
    assert detector.score(noise) > detector.score(audio)


def test_add_reference_persists(tmp_path, refs) -> None:
    directory, audio = refs
    monkey_dir = tmp_path / "own"
    import src.wakeword_dtw as module

    original = module.config.DATA_WAKEWORD_DIR
    module.config.DATA_WAKEWORD_DIR = monkey_dir
    try:
        detector = WakewordDTW(refs_dir=directory)
        detector.add_reference(audio)
        assert len(list(monkey_dir.glob("ref_*.npy"))) == 1
        assert len(detector._refs) == 4
    finally:
        module.config.DATA_WAKEWORD_DIR = original


def test_empty_audio_and_missing_refs(tmp_path, refs) -> None:
    directory, _ = refs
    detector = WakewordDTW(refs_dir=directory)
    assert detector.score(np.zeros(10, dtype=np.int16)) == float("inf")
    with pytest.raises(FileNotFoundError):
        WakewordDTW(refs_dir=tmp_path / "пусто")
    with pytest.raises(ValueError):
        WakewordDTW(refs_dir=directory, threshold=0.0)


def test_default_config_is_text() -> None:
    """Режим DTW не включён по умолчанию: прежний текстовый путь активен."""
    assert config.WAKEWORD_MODE == "text"