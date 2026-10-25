"""Точка входа Джарвиса.

    python main.py              запуск ассистента
    python main.py --setup      мастер настройки микрофона (пишет .env)
    python main.py --devices    просто список устройств ввода

JARVIS_LOG_LEVEL=DEBUG python main.py — подробный лог в консоль.
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

from src.logger import setup_logging

setup_logging()


def _main() -> int:
    # Тяжёлые импорты (torch, silero, vosk) — ТОЛЬКО после разбора аргументов,
    # иначе мастер настройки открывается несколько секунд.
    if "--setup" in sys.argv or "--devices" in sys.argv:
        from src.setup import run_setup
        return run_setup(list_only="--devices" in sys.argv)

    import atexit

    from src.assistant import run
    from src.metrics import metrics

    def _final_report():
        try:
            metrics.log_report(reset=False)
        except Exception:
            pass

    # Отчёт метрик при любом выходе, включая SIGTERM от systemd.
    atexit.register(_final_report)

    try:
        run()
    except KeyboardInterrupt:
        # Раньше Ctrl-C ловился только внутри главного цикла: прерывание на
        # старте (нет Ollama, занят DEVICE) падало с необработанным traceback.
        metrics.log_report()
        print("\n👋 Выход")
    return 0


if __name__ == "__main__":
    sys.exit(_main())