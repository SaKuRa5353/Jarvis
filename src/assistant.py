"""Оркестратор — главный цикл ассистента."""

import time

from src.llm import JarvisLLM
from src.router import plan_command
from src.stt import recognize
from src.tts import speak
from src.vad import record_until_silence
from src.wakeword import wait_for_wakeword


def _is_false_positive(text: str) -> bool:
    """Отсекает ложные срабатывания: распознали только обращение."""
    normalized = text.lower().strip().replace(",", "").replace(".", "")
    return len(normalized.split()) <= 1 and "джарвис" in normalized


def run():
    llm = JarvisLLM()

    print("=" * 50)
    print("         ДЖАРВИС ЗАПУЩЕН")
    print("=" * 50)

    while True:
        try:
            wait_for_wakeword()
            audio = record_until_silence()
            text = recognize(audio)

            if text is None:
                continue

            if _is_false_positive(text):
                print("[ложное срабатывание — игнор]")
                continue

            response, action = plan_command(text, llm)

            if response is None:
                continue

            speak(response)
            if action is not None:
                try:
                    action()
                except Exception as e:
                    print(f"[action error] {e}")

        except KeyboardInterrupt:
            print("\n👋 Выход")
            break
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            time.sleep(1)
