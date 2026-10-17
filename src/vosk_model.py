"""Единственный экземпляр Vosk-модели на весь процесс.

Wakeword и STT работают с одной и той же моделью: Model() грузит
i-vector экстрактор и граф ~1 секунду и держит ~50 MB, загружать её
дважды незачем.
"""

from vosk import Model, SetLogLevel

from config import VOSK_MODEL

from src.logger import get_logger

logger = get_logger("vosk")

# Vosk пишет LOG(...) напрямую в stderr, минуя logging.
SetLogLevel(-1)

_model: Model | None = None


def get_vosk_model() -> Model:
    """Лениво загружает и отдаёт общую Vosk-модель."""
    global _model
    if _model is None:
        logger.info("Загружаю Vosk модель...")
        _model = Model(VOSK_MODEL)
        logger.info("Vosk модель загружена")
    return _model
