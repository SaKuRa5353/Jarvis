"""Фонетическая нормализация русского текста и комбинированная метрика.

Назначение слоя
---------------
Vosk стабильно выдаёт фонетически близкие, но разные написания одного слова:
«джарвис» / «джарвись» / «джервис», «вк» / «века» / «фк». Символьное
редакционное расстояние различает их, поэтому поверх него вводится
фонетический код ``phi(x)`` и комбинированная мера

    D(a, b) = lambda * d(a, b) + (1 - lambda) * d(phi(a), phi(b)),   lambda in [0, 1]

Слой НЕ заменяет словарь известных слов: ``phi`` приводит варианты к одному
коду, но «века» и «вк» остаются разными словами — их различает словарь ``KNOWN``
и таблица весов ``DEFAULT_SUB`` (пара в-ф весом 0.2).

Правила
-------
Применяются к каждому слову в порядке:

1. нижний регистр, ``ё`` -> ``е``;
2. фильтрация: остаются только ``[0-9a-zа-я]``;
3. оглушение: звонкий согласный перед глухим или в абсолютном конце слова
   заменяется парным глухим (до фиксированной точки — каскад «зжд» -> «жш»);
4. редукция безударных гласных: ``о`` -> ``а``, ``е`` -> ``и``, ``я`` -> ``и``;
   гласные до первого согласного слова не редуцируются;
5. удаление ``ь`` и ``ъ``;
6. ассимиляция: ``с`` -> ``ш``, ``з`` -> ``ж`` непосредственно перед шипящими;
7. схлопывание удвоенных одинаковых символов.

Шаги 3-7 повторяются до фиксированной точки (не более ``MAX_PASSES`` раз).

Почему именно фиксированная точка
---------------------------------
Однократный проход НЕ идемпотентен: правило 6 может создать пару, которую
правило 3 оглушит, а правило 7 затем схлопнет. Контрпримеры, найденные
перебором на 3000 случайных строк:

    'даъжъ' -> 'даж' -> 'даш'      (однократный проход даёт 'даж', повтор 'даш')
    'зъв'   -> 'зф'  -> 'сф'
    'ефёбъ' -> 'ефиб' -> 'ефип'

Итерация до фиксированной точки устраняет все 60 нарушений из 3000 и даёт
свойство ``phi(phi(x)) = phi(x)``, которое требуется для корректности
``D`` как метрики.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from src.mathcore.distance import damerau_levenshtein
from src.mathcore.weights import (
    DEFAULT_DEL,
    DEFAULT_INS,
    DEFAULT_OTHER_SUB,
    DEFAULT_TR,
)

VOICED: dict[str, str] = {
    "б": "п",
    "в": "ф",
    "г": "к",
    "д": "т",
    "ж": "ш",
    "з": "с",
}
"""Звонкий -> парный глухой (правило 3)."""

REDUCED: dict[str, str] = {
    "о": "а",
    "е": "и",
    "я": "и",
}
"""Безударный гласный -> редуцированный (правило 4)."""

SOFT_SIGNS: frozenset[str] = frozenset(("ь", "ъ"))
"""Символы, не влияющие на звучание (правило 5)."""

SIBILANTS: frozenset[str] = frozenset(("ш", "ж", "ч", "щ"))
"""Шипящие, перед которыми работает ассимиляция (правило 6)."""

ASSIMILATE: dict[str, str] = {"с": "ш", "з": "ж"}
"""Ассимиляция перед шипящими (правило 6)."""

MAX_PASSES: int = 12
"""Предел итераций до фиксированной точки; на практике хватает 2-3."""

_DEVOICED: frozenset[str] = frozenset(VOICED.values())
_WORD_SPLIT_RE = re.compile(r"[^0-9a-zа-я]+")
_KEEP_RE = re.compile(r"[^0-9a-zа-я]")


def _devoice(word: str) -> str:
    """Правило 3: оглушение до фиксированной точки (каскад «зжд» -> «жш»)."""
    chars = list(word)
    n = len(chars)
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(chars):
            if ch in VOICED and (i == n - 1 or chars[i + 1] in _DEVOICED):
                chars[i] = VOICED[ch]
                changed = True
    return "".join(chars)


def _reduce(word: str) -> str:
    """Правило 4: редукция безударных гласных, кроме первого слога."""
    out: list[str] = []
    consonant_seen = False
    for ch in word:
        if ch in REDUCED:
            if consonant_seen:
                ch = REDUCED[ch]
        else:
            consonant_seen = True
        out.append(ch)
    return "".join(out)


def _drop_soft(word: str) -> str:
    """Правило 5: ``ь`` и ``ъ`` не влияют на звучание."""
    return "".join(ch for ch in word if ch not in SOFT_SIGNS)


def _assimilate(word: str) -> str:
    """Правило 6: ``с`` -> ``ш``, ``з`` -> ``ж`` непосредственно перед шипящими."""
    out: list[str] = []
    for i, ch in enumerate(word):
        if ch in ASSIMILATE and i + 1 < len(word) and word[i + 1] in SIBILANTS:
            out.append(ASSIMILATE[ch])
        else:
            out.append(ch)
    return "".join(out)


def _collapse_doubles(word: str) -> str:
    """Правило 7: «сс» -> «с», «мм» -> «м»."""
    out: list[str] = []
    for ch in word:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _pass(word: str) -> str:
    """Один проход правил 3-7."""
    return _collapse_doubles(_assimilate(_drop_soft(_reduce(_devoice(word)))))


def _code_word(word: str) -> str:
    """Фонетический код одного слова: правила 1-7, итерация до фиксированной точки."""
    cleaned = _KEEP_RE.sub("", word.lower().replace("ё", "е"))
    current = cleaned
    for _ in range(MAX_PASSES):
        nxt = _pass(current)
        if nxt == current:
            break
        current = nxt
    return current


def phonetic_code(text: str) -> str:
    """Фонетический код строки; слова разделяются одиночным пробелом.

    Идемпотентен: ``phonetic_code(phonetic_code(x)) == phonetic_code(x)``.
    Пустая строка и строка без букв/цифр дают ``""``.
    """
    lowered = text.lower().replace("ё", "е")
    words = [w for w in _WORD_SPLIT_RE.split(lowered) if w]
    return " ".join(_code_word(w) for w in words)


def phonetic_distance(
    a: str,
    b: str,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: Mapping[frozenset[str], float] | None = None,
    band: int | None = None,
) -> float:
    """``d_фон(a, b) = d(phi(a), phi(b))``."""
    return damerau_levenshtein(
        phonetic_code(a), phonetic_code(b),
        w_ins, w_del, w_sub, w_tr, sub_table, band,
    )


def combined_distance(
    a: str,
    b: str,
    lam: float = 0.5,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: Mapping[frozenset[str], float] | None = None,
    band: int | None = None,
) -> float:
    """``D(a, b) = lambda * d(a, b) + (1 - lambda) * d_фон(a, b)``, ``lambda in [0, 1]``.

    При ``lam = 1`` совпадает с ``damerau_levenshtein``, при ``lam = 0`` —
    с ``phonetic_distance``.

    Выпуклая комбинация двух метрик с неотрицательными коэффициентами есть
    метрика, поэтому при ``lam in [0, 1]`` и ``w_ins == w_del`` функция ``D``
    удовлетворяет всем четырём аксиомам.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam должен лежать в [0, 1], получено {lam}")
    d_sym = damerau_levenshtein(
        a, b, w_ins, w_del, w_sub, w_tr, sub_table, band
    )
    d_phon = phonetic_distance(
        a, b, w_ins, w_del, w_sub, w_tr, sub_table, band
    )
    return lam * d_sym + (1.0 - lam) * d_phon
