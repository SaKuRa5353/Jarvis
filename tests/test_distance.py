"""Unit-тесты src/mathcore/distance.py: взвешенное DL, метрика, сходство."""

from __future__ import annotations

import difflib
import math
import random
import time

import pytest

from src.mathcore.distance import (
    damerau_levenshtein,
    directed_distance,
    is_metric,
    similarity,
)
from src.mathcore.weights import (
    DEFAULT_DEL,
    DEFAULT_INS,
    DEFAULT_OTHER_SUB,
    DEFAULT_SUB,
    DEFAULT_TR,
)

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def _random_pair(rng: random.Random, max_len: int = 6) -> tuple[str, str]:
    return (
        "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_len))),
        "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_len))),
    )


# ───────────────────────── базовые значения ─────────────────────────


def test_zero_distance_for_identical_strings() -> None:
    assert damerau_levenshtein("кот", "кот") == 0.0


def test_single_insertion() -> None:
    assert damerau_levenshtein("кот", "скот") == DEFAULT_INS


def test_empty_strings() -> None:
    assert damerau_levenshtein("", "") == 0.0
    # «скот» — 4 символа, значит 4 вставки
    assert math.isclose(
        damerau_levenshtein("", "скот"), 4 * DEFAULT_INS, rel_tol=1e-9
    )


def test_transposition_is_cheaper_than_two_substitutions() -> None:
    assert damerau_levenshtein("аб", "ба") == DEFAULT_TR
    assert damerau_levenshtein("аб", "ба") < 2 * DEFAULT_OTHER_SUB


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("вк", "фк", 0.2),      # оглушение
        ("дом", "том", 0.2),    # д-т
        ("кот", "код", 0.2),    # т-д
        ("он", "ан", 0.25),     # редукция о-а
        ("ил", "ыл", 0.3),      # редукция и-ы
    ],
)
def test_phonetic_weights_applied(a: str, b: str, expected: float) -> None:
    got = damerau_levenshtein(a, b)
    assert math.isclose(got, expected, rel_tol=1e-9), f"d({a!r},{b!r}) = {got}"


def test_every_default_sub_pair_has_its_weight() -> None:
    """Регрессия на «таблица импортирована, но не подключена».

    Проверяет все 12 пар DEFAULT_SUB individually: без подключения таблицы
    каждая замена стоила бы DEFAULT_OTHER_SUB = 1.0.
    """
    assert len(DEFAULT_SUB) == 12
    for pair, weight in DEFAULT_SUB.items():
        x, y = sorted(pair)
        got = damerau_levenshtein(x, y)
        expected = min(weight, 2 * DEFAULT_INS)  # две вставки могут быть дешевле
        assert got <= expected + 1e-9, f"пара {pair}: {got} > {expected}"
        assert got < DEFAULT_OTHER_SUB, f"пара {pair} не применилась: {got}"


def test_sub_table_can_be_disabled() -> None:
    """Пустой словарь отключает фонетические веса."""
    assert damerau_levenshtein("вк", "фк", sub_table={}) == DEFAULT_OTHER_SUB


def test_sub_table_none_means_default() -> None:
    assert damerau_levenshtein("вк", "фк", sub_table=None) == 0.2


def test_custom_sub_table_overrides_default() -> None:
    custom = {frozenset(("в", "ф")): 0.05}
    assert math.isclose(
        damerau_levenshtein("вк", "фк", sub_table=custom), 0.05, rel_tol=1e-9
    )


# ───────────────────────── аксиомы метрики ─────────────────────────


def test_default_weights_are_symmetric() -> None:
    """w_ins == w_del — необходимое условие симметрии (аксиома метрики)."""
    assert DEFAULT_INS == DEFAULT_DEL


def test_asymmetric_weights_break_symmetry() -> None:
    """Документирует, почему w_ins обязан равняться w_del.

    При 0.8 / 0.9 направленное расстояние асимметрично, поэтому публичный API
    и использует равные веса, а не min() из двух ориентаций.
    """
    fwd = directed_distance("кот", "скот", w_ins=0.8, w_del=0.9)
    bwd = directed_distance("скот", "кот", w_ins=0.8, w_del=0.9)
    assert fwd == 0.8
    assert bwd == 0.9
    assert fwd != bwd


def test_non_negative_on_random_pairs() -> None:
    rng = random.Random(42)
    for _ in range(200):
        a, b = _random_pair(rng)
        assert damerau_levenshtein(a, b) >= 0.0


def test_symmetry_on_random_pairs() -> None:
    rng = random.Random(43)
    for _ in range(200):
        a, b = _random_pair(rng)
        assert math.isclose(
            damerau_levenshtein(a, b),
            damerau_levenshtein(b, a),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


def test_triangle_on_random_triples() -> None:
    rng = random.Random(44)
    for _ in range(500):
        a, _ = _random_pair(rng)
        b, c = _random_pair(rng)
        d_ac = damerau_levenshtein(a, c)
        d_ab = damerau_levenshtein(a, b)
        d_bc = damerau_levenshtein(b, c)
        assert d_ac <= d_ab + d_bc + 1e-9


def test_zero_iff_equal() -> None:
    rng = random.Random(45)
    for _ in range(200):
        a, b = _random_pair(rng, max_len=5)
        assert (a == b) == (damerau_levenshtein(a, b) == 0.0)


@pytest.mark.parametrize(
    "a,b,c",
    [
        ("кот", "скот", "тигр"),
        ("кот", "скот", "кот"),
        ("джарвис", "джервис", "гараж"),
        ("вк", "фк", "вконтакте"),
        ("", "а", "аб"),
    ],
)
def test_is_metric_on_named_triples(a: str, b: str, c: str) -> None:
    assert is_metric(a, b, c) is True


def test_is_metric_detects_asymmetry() -> None:
    """is_metric обязан вернуть False при асимметричных весах."""
    assert is_metric("кот", "скот", "тигр", w_ins=0.8, w_del=0.9) is False


# ───────────────────────── полоса Укконена ─────────────────────────


@pytest.mark.parametrize(
    "a,b",
    [("кот", "кот"), ("кот", "скот"), ("джарвис", "джервис"), ("", "абв")],
)
def test_wide_band_equals_full_matrix(a: str, b: str) -> None:
    full = damerau_levenshtein(a, b)
    wide = damerau_levenshtein(a, b, band=max(len(a), len(b)) + 1)
    assert wide == full


def test_band_returns_exact_value_when_distance_is_small() -> None:
    # истинное расстояние 0.85 <= band=1 -> возвращается точное значение
    assert math.isclose(
        damerau_levenshtein("кот", "скот", band=1), DEFAULT_INS, rel_tol=1e-9
    )


def test_band_signals_overflow() -> None:
    # истинное расстояние больше полосы -> band + 1
    got = damerau_levenshtein("abcdef", "zyxwvu", band=1)
    assert got == 2.0


# ───────────────────────── производительность ─────────────────────────


def test_performance_500_pairs_of_length_30() -> None:
    rng = random.Random(46)
    words = ["".join(rng.choice(ALPHABET) for _ in range(30)) for _ in range(1000)]
    pairs = [(words[i], words[i + 1]) for i in range(0, 1000, 2)]
    start = time.monotonic()
    for a, b in pairs:
        damerau_levenshtein(a, b)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5, f"500 пар длиной 30 за {elapsed:.2f} с — дольше 1.5 с"


def test_band_is_faster_than_full_matrix() -> None:
    rng = random.Random(47)
    words = ["".join(rng.choice(ALPHABET) for _ in range(60)) for _ in range(60)]
    pairs = [(words[i], words[i + 1]) for i in range(0, 60, 2)]

    t0 = time.monotonic()
    for a, b in pairs:
        damerau_levenshtein(a, b)
    full = time.monotonic() - t0

    t0 = time.monotonic()
    for a, b in pairs:
        damerau_levenshtein(a, b, band=3)
    banded = time.monotonic() - t0

    assert banded < full, f"полоса {banded:.3f} с не быстрее полного {full:.3f} с"


# ───────────────────────── difflib не метрика ─────────────────────────


def test_difflib_ratio_is_not_a_metric() -> None:
    """difflib.SequenceMatcher.ratio() не порождает метрику.

    Контрпример 1 (транзитивность сходства): r = 2M/(|x|+|y|)
        a='рчннп', b='ешп', c='мч'
        r(a,b)=0.25, r(b,c)=0.0, r(a,c)=0.2857
        a похож на c СИЛЬНЕЕ, чем на b, хотя b и c не похожи вовсе.

    Контрпример 2 (неравенство треугольника для d = 1 - r):
        a='т', b='тч', c='нмч'
        d(a,b)=1/3, d(b,c)=0.6, d(a,c)=1.0
        1.0 > 1/3 + 0.6 = 0.9333

    Это обоснование замены difflib на взвешенный Дамерау-Левенштейн.
    """
    def ratio(x: str, y: str) -> float:
        return difflib.SequenceMatcher(None, x, y).ratio()

    def dist(x: str, y: str) -> float:
        return 1.0 - ratio(x, y)

    a, b, c = "рчннп", "ешп", "мч"
    assert ratio(a, c) > ratio(a, b) + ratio(b, c)

    p, q, s = "т", "тч", "нмч"
    assert dist(p, s) > dist(p, q) + dist(q, s) + 1e-12

    # наша реализация на тех же тройках аксиомы не нарушает
    assert is_metric(a, b, c) is True
    assert is_metric(p, q, s) is True


# ───────────────────────── similarity ─────────────────────────


def test_similarity_properties() -> None:
    assert similarity("", "") == 1.0
    assert similarity("кот", "кот") == 1.0
    for a, b in [("кот", "скот"), ("кот", "тигр"), ("", "скот"), ("а", "б")]:
        value = similarity(a, b)
        assert value <= 1.0 + 1e-12


def test_similarity_known_value() -> None:
    # d('кот','скот') = 0.85, max(len) = 4 -> 1 - 0.2125 = 0.7875
    assert math.isclose(similarity("кот", "скот"), 0.7875, rel_tol=1e-9)


def test_similarity_is_symmetric() -> None:
    rng = random.Random(48)
    for _ in range(100):
        a, b = _random_pair(rng)
        assert math.isclose(
            similarity(a, b), similarity(b, a), rel_tol=1e-9, abs_tol=1e-12
        )
def test_is_metric_rejects_negative_weights() -> None:
    """Аксиома неотрицательности: отрицательный вес вставки -> не метрика."""
    # d("", "а") = -1.0 < 0, при этом d("а", "") = +0.85 -> нарушены обе аксиомы
    assert is_metric("", "а", "аб", w_ins=-1.0) is False


def test_is_metric_rejects_violated_identity() -> None:
    """Аксиома тождества неразличимых: d(a, b) = 0 при a != b -> не метрика.

    Нулевой вес ЗАМЕНЫ (а не вставки) обнуляет расстояние между строками равной
    длины, но сохраняет симметрию — поэтому проверка доходит до аксиомы
    тождества, а не останавливается на аксиоме симметрии.
    """
    assert damerau_levenshtein("а", "б", w_sub=0.0) == 0.0
    assert damerau_levenshtein("б", "а", w_sub=0.0) == 0.0  # симметрия сохранена
    assert is_metric("а", "б", "в", w_sub=0.0) is False
