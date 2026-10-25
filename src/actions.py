"""Низкоуровневые действия: запуск процессов и открытие ссылок.

Отдельный модуль вне пакета commands, чтобы его мог импортировать
src.llm_tools без цикла commands/__init__ -> llm -> llm_tools.
"""

import subprocess


def open_url(url: str):
    """Открывает URL в браузере по умолчанию."""
    subprocess.Popen(
        ["xdg-open", url],
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
