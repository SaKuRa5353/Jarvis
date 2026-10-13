"""Google Speech-to-Text."""

import numpy as np
import speech_recognition as sr

from config import SAMPLE_RATE


_recognizer = sr.Recognizer()


def recognize(audio_int16: np.ndarray) -> str | None:
    if len(audio_int16) == 0:
        return None
    audio_data = sr.AudioData(audio_int16.tobytes(), SAMPLE_RATE, 2)
    print("🧠 Распознаю...")
    try:
        text = _recognizer.recognize_google(audio_data, language="ru-RU")
        print(f"📝 Ты: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Не разобрал")
        return None
    except sr.RequestError as e:
        print(f"❌ Ошибка STT: {e}")
        return None
