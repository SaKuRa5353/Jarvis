"""Общие хелперы для команд."""

import subprocess

from src.logger import get_logger

logger = get_logger("commands.utils")


def open_url(url: str):
    """Открывает URL в brave."""
    subprocess.Popen(
        ["brave", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run(cmd: list[str]):
    """Запускает команду в фоне, не дожидаясь завершения."""
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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
