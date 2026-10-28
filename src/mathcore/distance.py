"""Взвешенное редакционное расстояние Дамерау-Левенштейна.

Метрика
-------
Расстояние ``d`` является метрикой при условиях

    w_ins == w_del >= 0,   w_tr >= 0,   w_sub(x, y) >= 0,   w_sub(x, x) = 0

*Симметрия* следует из равенства весов вставки и удаления: каждой вставке в
редакционном скрипте ``a -> b`` соответствует удаление в скрипте ``b -> a`` той
же стоимости. *Неравенство треугольника* следует из конкатенации скриптов:
скрипт ``a -> b`` плюс скрипт ``b -> c`` есть корректный (не обязательно
оптимальный) скрипт ``a -> c``. Транспозиция симметрична (обратная
транспозиция есть та же транспозиция), поэтому ограничений на ``w_tr``
не требуется.

Сложность
---------
Без полосы: ``O(n * m)`` по времени, ``O(n * m)`` по памяти.
С полосой Укконена ``band = k``: ``O(n * k)`` по времени.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from src.mathcore.weights import (
    DEFAULT_DEL,
    DEFAULT_INS,
    DEFAULT_OTHER_SUB,
    DEFAULT_SUB,
    DEFAULT_TR,
)

SubTable = Mapping[frozenset[str], float]


def _substitution_cost(x: str, y: str, sub_table: SubTable, default: float) -> float:
    """Вес замены символа ``x`` на ``y``.

    Одинаковые символы стоят 0 (аксиома ``d(a, a) = 0``); без этого условия
    расстояние перестаёт быть метрикой даже при симметричных весах.
    """
    if x == y:
        return 0.0
    cost: float | None = sub_table.get(frozenset((x, y)))
    return default if cost is None else cost


def directed_distance(
    a: str,
    b: str,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: SubTable = DEFAULT_SUB,
    band: int | None = None,
) -> float:
    """Направленное расстояние: стоимость скрипта преобразования ``a`` в ``b``.

    Рекуррент (динамическое программирование, матрица ``(n+1) x (m+1)``):

        d(i, j) = min(
            d(i-1, j)   + w_del,               # удаление a_i
            d(i,   j-1) + w_ins,               # вставка b_j
            d(i-1, j-1) + sub(a_i, b_j),       # замена
            d(i-2, j-2) + w_tr,                # транспозиция
        )

    Транспозиция применяется только при ``a[i-1] == b[j-2] and a[i-2] == b[j-1]``.

    Если ``band`` задан, заполняются только ячейки с ``|i - j| <= band``;
    остальные считаются бесконечными. Результат — верхняя оценка истинного
    расстояния; если она превышает ``band``, возвращается ``band + 1.0``
    (сигнал «расстояние больше полосы»).

    При ``w_ins != w_del`` функция НЕсимметрична — это полезная диагностика,
    но для публичного API используйте ``damerau_levenshtein``.
    """
    n, m = len(a), len(b)
    inf = math.inf
    d: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i * w_del if band is None or i <= band else inf
    for j in range(m + 1):
        d[0][j] = j * w_ins if band is None or j <= band else inf
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i - j) > band:
                d[i][j] = inf
                continue
            cost = _substitution_cost(a[i - 1], b[j - 1], sub_table, w_sub)
            best = min(
                d[i - 1][j] + w_del,
                d[i][j - 1] + w_ins,
                d[i - 1][j - 1] + cost,
            )
            if i >= 2 and j >= 2 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, d[i - 2][j - 2] + w_tr)
            d[i][j] = best
    result = d[n][m]
    if band is not None and result > band:
        return float(band + 1)
    return result


def damerau_levenshtein(
    a: str,
    b: str,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: SubTable | None = None,
    band: int | None = None,
) -> float:
    """Симметричное взвешенное расстояние Дамерау-Левенштейна.

    ``sub_table=None`` означает таблицу по умолчанию ``DEFAULT_SUB``;
    пустой словарь ``{}`` отключает фонетические веса и оставляет только
    ``w_sub`` для всех замен.

    Симметрия обеспечивается равенством ``DEFAULT_INS == DEFAULT_DEL``,
    а не взятием минимума из двух ориентаций: последнее удваивает стоимость
    и не гарантирует неравенство треугольника.
    """
    table: SubTable = DEFAULT_SUB if sub_table is None else sub_table
    return directed_distance(a, b, w_ins, w_del, w_sub, w_tr, table, band)


def similarity(
    a: str,
    b: str,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: SubTable | None = None,
    band: int | None = None,
) -> float:
    """Нормированное сходство: ``sim(a, b) = 1 - d(a, b) / max(|a|, |b|)``.

    Для двух пустых строк возвращает 1.0. Может быть отрицательным, если
    расстояние превышает ``max(|a|, |b|)`` (возможно при ``w_sub > 1``).
    """
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    dist = damerau_levenshtein(a, b, w_ins, w_del, w_sub, w_tr, sub_table, band)
    return 1.0 - dist / denom


def is_metric(
    a: str,
    b: str,
    c: str,
    w_ins: float = DEFAULT_INS,
    w_del: float = DEFAULT_DEL,
    w_sub: float = DEFAULT_OTHER_SUB,
    w_tr: float = DEFAULT_TR,
    sub_table: SubTable | None = None,
    band: int | None = None,
) -> bool:
    """Проверяет четыре аксиомы метрики на тройке ``(a, b, c)``.

    1. неотрицательность: ``d(x, y) >= 0``;
    2. тождество неразличимых: ``d(a, b) == 0 <=> a == b``;
    3. симметрия: ``d(a, b) == d(b, a)``;
    4. неравенство треугольника: ``d(a, c) <= d(a, b) + d(b, c)``.

    Достаточно трёх вызовов ``d``: симметрия при ``w_ins == w_del`` следует
    из равенства весов, но проверяется явно на паре ``(a, b)``.
    """
    # Явные именованные аргументы вместо **kw: словарь значений имеет тип
    # dict[str, float | Mapping[...] | int | None], и mypy не может вывести
    # тип отдельного значения при распаковке (18 ошибок arg-type).
    d_ab = damerau_levenshtein(
        a, b, w_ins, w_del, w_sub, w_tr, sub_table, band
    )
    d_ba = damerau_levenshtein(
        b, a, w_ins, w_del, w_sub, w_tr, sub_table, band
    )
    d_bc = damerau_levenshtein(
        b, c, w_ins, w_del, w_sub, w_tr, sub_table, band
    )
    d_ac = damerau_levenshtein(
        a, c, w_ins, w_del, w_sub, w_tr, sub_table, band
    )

    if min(d_ab, d_ba, d_bc, d_ac) < 0.0:
        return False
    if not math.isclose(d_ab, d_ba, rel_tol=1e-9, abs_tol=1e-12):
        return False
    if (a == b) != (d_ab == 0.0):
        return False
    return d_ac <= d_ab + d_bc + 1e-9
