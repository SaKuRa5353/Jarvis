"""Общие хелперы для команд."""

from src.actions import open_url, run  # noqa: F401  (реэкспорт для совместимости)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Русская плюрализация: 1 час, 2 часа, 5 часов."""
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 19:
        return many
    n_abs %= 10
    if n_abs == 1:
        return one
    if 2 <= n_abs <= 4:
        return few
    return many
