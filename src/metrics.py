"""Базовый мониторинг и метрики для Джарвиса.

Предоставляет:
- timing декоратор для измерения времени выполнения функций
- счётчик вызовов функций
- реестр метрик для сбора и отчёта
- логирование метрик в структурированном виде

Пример использования:

    from src.metrics import timed, count_calls, metrics

    @timed()
    def my_function():
        ...

    # В конце работы или периодически:
    metrics.log_report()
"""

import functools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger("metrics")


@dataclass
class MetricEntry:
    """Одна метрика: время выполнения и количество вызовов."""

    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0


class MetricsRegistry:
    """Реестр всех метрик приложения."""

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricEntry] = defaultdict(MetricEntry)

    def record(self, name: str, duration: float) -> None:
        """Записывает время выполнения для метрики."""
        entry = self._metrics[name]
        entry.call_count += 1
        entry.total_time += duration
        entry.min_time = min(entry.min_time, duration)
        entry.max_time = max(entry.max_time, duration)

    def get_report(self) -> str:
        """Возвращает отчёт по всем метрикам в виде строки."""
        if not self._metrics:
            return "Нет собранных метрик"

        lines = ["\n📊 ОТЧЁТ МЕТРИК:", "=" * 60]
        lines.append(
            f"  {'метрика':44} | вызовы | ср.время | мин | макс | всего"
        )
        lines.append("  " + "-" * 60)
        for name, entry in sorted(self._metrics.items()):
            lines.append(
                f"  {name:44} | {entry.call_count:6d} | "
                f"{entry.avg_time*1000:8.1f}мс | "
                f"{entry.min_time*1000:6.1f}мс | "
                f"{entry.max_time*1000:6.1f}мс | "
                f"{entry.total_time*1000:8.1f}мс"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def log_report(self, level: int = logging.INFO, reset: bool = True) -> None:
        """Логгирует отчёт по метрикам.

        reset=True (по умолчанию) очищает реестр после печати, чтобы отчёт
        не дублировался при втором вызове (например, atexit после Ctrl-C).
        """
        if not reset and not self._metrics:
            return  # отчёт уже напечатан и реестр очишен — не дублируем
        logger.log(level, self.get_report())
        if reset:
            self.reset()

    def reset(self) -> None:
        """Сбрасывает все накопленные метрики."""
        self._metrics.clear()


# Глобальный реестр метрик
metrics = MetricsRegistry()


def timed(name: Optional[str] = None):
    """Декоратор для измерения времени выполнения функции.

    Записывает метрику в глобальный реестр `metrics`.
    Время измеряется через time.perf_counter() для высокой точности.

    Пример:

        @timed()
        def my_function():
            ...

        @timed("custom_metric_name")
        def another_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                metrics.record(metric_name, duration)

        return wrapper

    return decorator


def count_calls(name: Optional[str] = None):
    """Декоратор для подсчёта вызовов функции (без измерения времени).

    Полезен для подсчёта количества вызовов LLM, TTS, API и т.п.
    """

    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics.record(metric_name, 0.0)
            return func(*args, **kwargs)

        return wrapper

    return decorator
