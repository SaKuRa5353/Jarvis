"""
Настройка логирования для Джарвиса.

- Пишет в консоль (с цветом) и в файл logs/jarvis.log
- Ротация: 5 файлов по 1 MB
- Формат: время | уровень | модуль | сообщение
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "jarvis.log"

# Цвета для консоли (ANSI)
COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[41m",  # red background
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветом для консоли."""

    def format(self, record):
        color = COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record)


def setup_logging(level: int = logging.INFO):
    """Настраивает root logger. Вызывать один раз при старте."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # пропускаем всё, фильтруем в хендлерах

    # Чистим старые хендлеры (если setup вызывается повторно)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ── Консоль ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # ── Файл (без цвета) ──
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_000_000,     # 1 MB
        backupCount=5,          # 5 файлов истории
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)    # INFO вместо DEBUG
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

        # ── Приглушаем шумные библиотеки ──
    for noisy in (
        "httpx2", "httpx",
        "httpcore2", "httpcore",
        "openai", "openai._base_client",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Удобная обёртка для получения логгера по имени модуля."""
    return logging.getLogger(name)
