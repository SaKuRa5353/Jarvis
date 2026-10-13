"""Оркестратор — главный цикл ассистента."""

import time

from src.llm import JarvisLLM
from src.logger import get_logger
from src.router import plan_command
from src.stt import recognize
from src.tts import speak
from src.vad import record_until_silence
from src.wakeword import wait_for_wakeword


logger = get_logger("assistant")


def _is_false_positive(text: str) -> bool:
    """Отсекает ложные срабатывания: распознали только обращение."""
    normalized = text.lower().strip().replace(",", "").replace(".", "")
    return len(normalized.split()) <= 1 and "джарвис" in normalized


def run():
    llm = JarvisLLM()

    logger.info("=" * 50)
    logger.info("         ДЖАРВИС ЗАПУЩЕН")
    logger.info("=" * 50)

    while True:
        try:
            wait_for_wakeword()
            audio = record_until_silence()
            text = recognize(audio)

            if text is None:
                continue

            if _is_false_positive(text):
                logger.warning("[ложное срабатывание — игнор]")
                continue

            response, action = plan_command(text, llm)

            # Если и ответа, и действия нет — пропускаем
            if response is None and action is None:
                continue

            # Сначала говорим (если есть что)
            if response is not None:
                speak(response)

            # Потом делаем (если есть что)
            if action is not None:
                try:
                    action()
                except Exception as e:
                    logger.error("[action error] %s", e)

        except KeyboardInterrupt:
            logger.info("\n👋 Выход")
            break
        except Exception as e:
            logger.error("⚠️ Ошибка: %s", e)
            time.sleep(1)