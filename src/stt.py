"""Google Speech-to-Text."""
from src.logger import get_logger

logger = get_logger("stt")
import numpy as np
import speech_recognition as sr

from config import SAMPLE_RATE


_recognizer = sr.Recognizer()


def recognize(audio_int16: np.ndarray) -> str | None:
    if len(audio_int16) == 0:
        return None
    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
    logger.info("🧠 Распознаю...")
    try:
        text = _recognizer.recognize_google(audio_data, language="ru-RU")
        logger.info("📝 Ты: %s", text)
        return text
    except sr.UnknownValueError:
        logger.warning("❌ Не разобрал")
        return None
    except sr.RequestError as e:
        logger.error("❌ Ошибка STT: %s", e)
        return None
