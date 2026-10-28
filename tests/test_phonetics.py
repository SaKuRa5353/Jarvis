"""Unit-тесты src/mathcore/phonetics.py: фонетический код и комбинированная метрика."""

from __future__ import annotations

import math
import random

import pytest

from src.mathcore.distance import damerau_levenshtein
from src.mathcore.phonetics import (
    ASSIMILATE,
    REDUCED,
    SIBILANTS,
    SOFT_SIGNS,
    VOICED,
    _assimilate,
    _collapse_doubles,
    _devoice,
    _drop_soft,
    _reduce,
    combined_distance,
    phonetic_code,
    phonetic_distance,
)

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


# ---------------------- таблицы правил ----------------------


def test_voiced_map_contents() -> None:
    assert VOICED == {"б": "п", "в": "ф", "г": "к", "д": "т", "ж": "ш", "з": "с"}


def test_reduced_map_contents() -> None:
    assert REDUCED == {"о": "а", "е": "и", "я": "и"}


def test_soft_and_sibilant_sets() -> None:
    assert SOFT_SIGNS == frozenset(("ь", "ъ"))
    assert SIBILANTS == frozenset(("ш", "ж", "ч", "щ"))
    assert ASSIMILATE == {"с": "ш", "з": "ж"}


# ---------------------- правила по отдельности ----------------------


def test_rule_3_devoicing_at_word_end() -> None:
    assert _devoice("вк") == "фк"
    # сонорные (м, н, р, л) не глушат предшествующий согласный
    assert _devoice("дом") == "дом"
    assert _devoice("дуб") == "дуп"          # абсолютный конец слова
    assert _devoice("лодка") == "лотка"      # перед глухим к


def test_rule_3_devoicing_cascade_reaches_fixpoint() -> None:
    # ж -> ш (таблица VOICED), затем з -> с перед ш; фиксированная точка за 2 прохода
    assert _devoice("зж") == "сш"
    assert _devoice(_devoice("зж")) == "сш"


def test_rule_4_reduction_skips_first_syllable() -> None:
    assert _reduce("вода") == "вада"
    # слово начинается с гласной: первый гласный не редуцируется
    assert _reduce("озеро") == "озира"


def test_rule_5_soft_signs_removed() -> None:
    assert _drop_soft("громкость") == "громкост"
    assert _drop_soft("объект") == "обект"


def test_rule_6_assimilation_before_sibilants() -> None:
    assert _assimilate("сшить") == "шшить"
    assert _assimilate("зжать") == "жжать"
    assert _assimilate("сок") == "сок"


def test_rule_7_collapse_doubles() -> None:
    # правило 7 схлопывает только удвоенные; ь убирает правило 5 (_drop_soft)
    assert _collapse_doubles("шшить") == "шить"
    assert _collapse_doubles("программма") == "програма"
    assert _collapse_doubles("абв") == "абв"
    # а полный конвейер даёт «шит»: сшить -> шшить -> шить -> шит
    assert phonetic_code("сшить") == "шит"


# ---------------------- конкретные переходы ----------------------


@pytest.mark.parametrize(
    "src,expected",
    [
        ("вк", "фк"),
        ("вода", "вада"),
        ("джарвис", "джарвис"),
        ("джарвись", "джарвис"),
        ("громкость", "грамкаст"),
        ("процессора", "працисара"),
        ("сшить", "шит"),
        ("открой века", "открай вика"),
    ],
)
def test_concrete_transitions(src: str, expected: str) -> None:
    """Эталонные значения вычислены независимой реализацией правил 1-7."""
    assert phonetic_code(src) == expected


def test_soft_sign_makes_jarvis_variants_equal() -> None:
    """Главный мотивирующий пример: варианты обращения дают один код.

    Без правила 5 (удаление ь) коды различаются и фонетический слой
    не решает задачу, ради которой введён.
    """
    assert phonetic_code("джарвис") == phonetic_code("джарвись")


def test_case_and_punctuation_normalized() -> None:
    assert phonetic_code("Джарвис!") == phonetic_code("джарвис")
    assert phonetic_code("  ДЖАРВИС  ") == phonetic_code("джарвис")


def test_yo_replaced_by_ye() -> None:
    assert phonetic_code("ёж") == phonetic_code("еж")


def test_empty_string() -> None:
    assert phonetic_code("") == ""
    assert phonetic_code("   ") == ""
    assert phonetic_code("!!!") == ""


# ---------------------- свойства ----------------------


def test_idempotence_on_random_strings() -> None:
    """phi(phi(x)) == phi(x) на 3000 случайных строках.

    Однократный проход правил это свойство НАРУШАЕТ (60 случаев из 3000):
        'даъжъ' -> 'даж' -> 'даш'
        'зъв'   -> 'зф'  -> 'сф'
        'ефёбъ' -> 'ефиб' -> 'ефип'
    Поэтому _code_word итерирует правила 3-7 до фиксированной точки.
    """
    rng = random.Random(99)
    violations = []
    for _ in range(3000):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 9)))
        once = phonetic_code(text)
        if phonetic_code(once) != once:
            violations.append((text, once, phonetic_code(once)))
    assert not violations, f"нарушений {len(violations)}, первые: {violations[:3]}"


def test_idempotence_on_named_examples() -> None:
    for text in ("джарвис", "джарвись", "громкость", "процессора",
                 "сшить", "открой века", "объект", "даъжъ", "зъв", "ефёбъ"):
        once = phonetic_code(text)
        assert phonetic_code(once) == once, text


def test_length_never_grows_for_single_words() -> None:
    """Для однословных входов длина не растёт.

    Для многословных свойство не гарантировано: слова соединяются одиночным
    пробелом, которого могло не быть в исходной строке («а  б» -> «а б»).
    """
    for text in ("вк", "громкость", "процессора", "вода", "джарвис", "джарвись"):
        assert len(phonetic_code(text)) <= len(text), text


def test_multiword_joins_with_single_space() -> None:
    assert phonetic_code("открой  века") == "открай вика"
    assert phonetic_code("открой века").count("  ") == 0


def test_output_alphabet_is_restricted() -> None:
    """В коде не остаётся ь/ъ, заглавных букв и знаков препинания."""
    rng = random.Random(7)
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzабувгдежзийклмнопрстфхцчшщыэюя ")
    for _ in range(300):
        text = "".join(rng.choice(ALPHABET + "Ё!?,.-") for _ in range(rng.randint(0, 12)))
        code = phonetic_code(text)
        extra = set(code) - allowed - set("ьъ")
        assert not extra, f"{text!r} -> {code!r}, лишние символы {extra}"
        assert "ь" not in code and "ъ" not in code, f"{text!r} -> {code!r}"


# ---------------------- комбинированная метрика ----------------------


def test_combined_distance_lam_boundaries() -> None:
    assert math.isclose(
        combined_distance("вк", "фк", lam=1.0),
        damerau_levenshtein("вк", "фк"),
        rel_tol=1e-9,
    )
    assert math.isclose(
        combined_distance("вк", "фк", lam=0.0),
        phonetic_distance("вк", "фк"),
        rel_tol=1e-9,
    )


def test_combined_distance_rejects_lam_out_of_range() -> None:
    with pytest.raises(ValueError):
        combined_distance("а", "б", lam=1.5)
    with pytest.raises(ValueError):
        combined_distance("а", "б", lam=-0.1)


def test_phonetic_layer_makes_jarvis_variants_closer() -> None:
    """Фонетический слой уменьшает расстояние между вариантами обращения."""
    a, b = "джарвис", "джарвись"
    sym = damerau_levenshtein(a, b)
    comb = combined_distance(a, b, lam=0.5)
    assert comb < sym, f"комбинированная {comb} не меньше символьной {sym}"
    assert math.isclose(comb, 0.425, rel_tol=1e-9)


def test_phonetic_distance_zero_for_same_code() -> None:
    assert phonetic_distance("джарвис", "джарвись") == 0.0
    assert phonetic_distance("громкость", "грамкаст") == 0.0


def test_combined_distance_is_metric_for_valid_lam() -> None:
    """Выпуклая комбинация двух метрик с неотрицательными весами есть метрика."""
    rng = random.Random(5)
    for _ in range(300):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 6)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 6)))
        c = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 6)))
        for lam in (0.0, 0.5, 1.0):
            d_ab = combined_distance(a, b, lam)
            d_bc = combined_distance(b, c, lam)
            d_ac = combined_distance(a, c, lam)
            assert d_ab >= 0.0
            assert math.isclose(d_ab, combined_distance(b, a, lam), rel_tol=1e-9, abs_tol=1e-12)
            assert d_ac <= d_ab + d_bc + 1e-9, f"ламбда {lam}: {a!r} {b!r} {c!r}"


def test_combined_distance_zero_iff_same_code_at_lam_zero() -> None:
    """При lam=0 расстояние 0 означает одинаковый фонетический код, не одинаковую строку."""
    assert combined_distance("джарвис", "джарвись", lam=0.0) == 0.0
    # строки при этом различаются: ноль означает совпадение фонетического кода
    assert phonetic_code("джарвис") == phonetic_code("джарвись")


def test_phonetic_layer_does_not_merge_different_words() -> None:
    """«века» и «вк» остаются разными словами: их различает словарь, не phi."""
    assert phonetic_code("века") != phonetic_code("вк")
    assert phonetic_code("открой века") == "открай вика"
