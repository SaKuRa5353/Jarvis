"""
Настройка логирования для Джарвиса.

- Пишет в консоль (с цветом) и в файл logs/jarvis.log
- Ротация: каждый день в полночь, архивы вида jarvis.log.2026-10-20,
  старше 7 дней удаляются автоматически (TimedRotatingFileHandler).
- Формат: время | уровень | модуль | сообщение
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
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

    # ── Файл (ротация по времени: ежедневно, архив старше 7 дней) ──
    # when="midnight" даёт суффикс %Y-%m-%d: jarvis.log.2026-10-20
    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",        # каждый день в полночь
        backupCount=7,          # хранить 7 архивов
        encoding="utf-8",
        utc=True,
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
