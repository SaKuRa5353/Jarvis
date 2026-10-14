"""Оркестратор — главный цикл ассистента с прерыванием."""

import threading
import time

from src.commands import LLM_MARKER, plan_command
from src.llm import JarvisLLM
from src.logger import get_logger
from src.stt import recognize
from src.tts import _get_voice, speak, stop_speaking
from src.vad import record_until_silence
from src.wakeword import check_wakeword_or_done, wait_for_wakeword


logger = get_logger("assistant")


def _is_false_positive(text: str) -> bool:
    """Отсекает ложные срабатывания: распознали только обращение."""
    normalized = text.lower().strip().replace(",", "").replace(".", "")
    return len(normalized.split()) <= 1 and "джарвис" in normalized


def _ask_with_interrupt(llm: JarvisLLM, text: str) -> str | None:
    """Отправляет запрос в LLM, но прерывает по wakeword."""
    result = {"answer": None, "done": False, "error": None}

    def _do_ask():
        try:
            result["answer"] = llm.ask(text)
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    thread = threading.Thread(target=_do_ask, daemon=True)
    thread.start()

    interrupted = check_wakeword_or_done(lambda: result["done"], timeout=30.0)

    if interrupted:
        logger.info("🛑 LLM прервана — начинаем заново")
        return None

    thread.join(timeout=1)
    if result["error"]:
        logger.error("[LLM error] %s", result["error"])
        return None
    return result["answer"]


def _speak_with_interrupt(text: str):
    """Говорит текст, но если услышит 'Джарвис' — прерывает."""
    tts_thread = threading.Thread(target=speak, args=(text,), daemon=True)
    tts_thread.start()

    interrupted = check_wakeword_or_done(
        lambda: not tts_thread.is_alive(),
        timeout=30.0,
    )

    if interrupted:
        stop_speaking()
        logger.info("🛑 TTS прерван — начинаем заново")

    tts_thread.join(timeout=1)


def run():
    llm = JarvisLLM()

    # Прогрев TTS — грузим Silero заранее
    logger.info("Прогреваю TTS...")
    _get_voice()
    logger.info("TTS готов")

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

            # Обработка маркера LLM
            if response == LLM_MARKER:
                logger.info("🤔 Думаю (LLM)...")
                answer = _ask_with_interrupt(llm, text)
                if answer:
                    _speak_with_interrupt(answer)
                continue

            # Обычный поток (правила)
            if response is None and action is None:
                continue

            if response is not None:
                _speak_with_interrupt(response)
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