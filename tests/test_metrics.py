"""Unit-тесты реестра метрик."""

from src.metrics import MetricsRegistry, count_calls, metrics, timed


def test_timed_runs_and_reports():
    @timed("tests.sample")
    def sample():
        return 42

    assert sample() == 42
    assert "tests.sample" in metrics.get_report()


def test_count_calls_increments():
    @count_calls("tests.calls")
    def counted():
        return 1

    counted()
    counted()
    assert "tests.calls" in metrics.get_report()


def test_global_metrics_reset():
    metrics.reset()
    assert isinstance(metrics.get_report(), str)
    metrics.reset()
