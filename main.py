"""Точка входа Джарвиса."""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

from src.logger import setup_logging

setup_logging()

from src.assistant import run


if __name__ == "__main__":
    run()