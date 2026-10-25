"""
Настройка логирования для Джарвиса.

- Пишет в консоль (с цветом) и в файл logs/jarvis.log
- Ротация: каждый день в полночь, архивы вида jarvis.log.2026-10-20,
  старше 7 дней удаляются автоматически (TimedRotatingFileHandler).
- Формат: время | уровень | модуль | сообщение
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


from config import (
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
)

LOG_FILE = LOG_DIR / LOG_FILE_NAME

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


def setup_logging(level: int | str | None = None):
    """Настраивает root logger. Вызывать один раз при старте.

    level=None → JARVIS_LOG_LEVEL из окружения (по умолчанию INFO).
    Путь к файлу лога печатается в консоль, чтобы его не искать.
    """
    if level is None:
        level = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

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
    # Ротация по размеру: jarvis.log.1 ... jarvis.log.N
    # (Раньше было TimedRotatingFileHandler(when="midnight", utc=True) —
    #  из-за utc=True граница суток съезжала на часовой пояс, а backupCount=2
    #  не соответствовал docstring про «7 дней».)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(os.getenv("JARVIS_FILE_LOG_LEVEL", "INFO").upper())
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)
    print(f"[logs] {LOG_FILE.resolve()}  "
          f"(уровень файла {logging.getLevelName(file_handler.level)}, "
          f"{LOG_MAX_BYTES // (1024 * 1024)} МБ x {LOG_BACKUP_COUNT})")

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
