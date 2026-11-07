"""Wakeword-детектор по акустическим эталонам (MFCC + DTW).

Хранит эталонные MFCC-последовательности произнесений «Джарвис» и оценивает
входящий аудио фрагмент минимумом DTW-расстояния до эталонов. Файлы эталонов
лежат в data/wakeword/*.npy; этот модуль их читает, но не пишет — запись
делает scripts/record_wakeword_refs.py (требует микрофон, маркер hw).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from src.logger import get_logger
from src.mathcore.features import dtw, mfcc_sequence

logger = get_logger("wakeword_dtw")

REFS_DIR = config.DATA_WAKEWORD_DIR
MATH_INF = float("inf")


class WakewordDTW:
    """Классификатор «обращение/не обращение» по акустике.

    Параметры признаков фиксированы (16 кГц, кадр 512, сдвиг 160, 13
    коэффициентов), чтобы эталоны и оценки были сравнимы.
    """

    def __init__(
        self,
        refs_dir: Path = REFS_DIR,
        sample_rate: int = config.SAMPLE_RATE,
        threshold: float = config.WAKEWORD_DTW_THRESHOLD,
        window: int = config.WAKEWORD_DTW_WINDOW,
        allow_empty: bool = False,
    ) -> None:
        if threshold <= 0:
            raise ValueError("Порог должен быть положительным")
        if window < 0:
            raise ValueError("Окно DTW должно быть неотрицательным")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window = window
        self._refs_dir = refs_dir
        self._refs = self._load_refs(refs_dir)
        if not self._refs and not allow_empty:
            raise FileNotFoundError(
                f"В {refs_dir} нет эталонов (*.npy). Записать: "
                "python scripts/record_wakeword_refs.py"
            )

    def _load_refs(self, refs_dir: Path) -> list[np.ndarray]:
        """Читает эталоны MFCC (T, 13); битые файлы пропускает с журналом."""
        refs = []
        for path in sorted(refs_dir.glob("*.npy")):
            try:
                matrix = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                logger.warning("Пропускаю битый эталон %s: %s", path, exc)
                continue
            if matrix.ndim == 2 and matrix.shape[1] == 13:
                converted = matrix.astype(np.float64)
                if len(converted) and np.all(np.isfinite(converted)):
                    refs.append(converted)
                else:
                    logger.warning("Пропускаю пустой или нечисловой эталон %s", path)
            else:
                logger.warning("Пропускаю эталон %s: нужна матрица (T, 13)", path)
        return refs

    def add_reference(self, audio_int16: np.ndarray) -> None:
        """Добавляет эталон из аудио int16 и сохраняет его в refs_dir."""
        sequence = mfcc_sequence(audio_int16, self.sample_rate)
        if len(sequence) == 0:
            raise ValueError("Слишком короткий фрагмент для эталона")
        self._refs.append(sequence)
        refs_dir = self._refs_dir
        refs_dir.mkdir(parents=True, exist_ok=True)
        used = {
            int(path.stem.removeprefix("ref_"))
            for path in refs_dir.glob("ref_*.npy")
            if path.stem.removeprefix("ref_").isdigit()
        }
        index = 0
        while index in used:
            index += 1
        np.save(refs_dir / f"ref_{index:03d}.npy", sequence)

    def score(self, audio_int16: np.ndarray) -> float:
        """Минимальное нормированное DTW-расстояние до эталонов.

        Меньше — ближе к «Джарвису». Для пустого входа — inf.
        """
        sequence = mfcc_sequence(audio_int16, self.sample_rate)
        if len(sequence) == 0:
            return MATH_INF
        return min(dtw(sequence, ref, window=self.window)[0] for ref in self._refs)

    def is_match(self, audio_int16: np.ndarray, threshold: float | None = None) -> bool:
        """True, если score <= порога (по умолчанию self.threshold)."""
        return self.score(audio_int16) <= (self.threshold if threshold is None else threshold)