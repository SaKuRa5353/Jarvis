"""Оркестратор — главный цикл ассистента с прерыванием.

Режим диалога: после первого обращения (wake word + команда) ассистент
остается «на связи» и слушает следующие 15 секунд без повторного «Джарвис».
Если за это время пользователь скажет «Джарвис» — диалог прерывается.
"""

import threading
import time
import traceback

from config import CONVERSATION_TIMEOUT_SEC

from src.audio import Microphone, MicrophoneError
from src.commands import LLM_MARKER, plan_command
from src.fuzzy import correct
from src.llm import JarvisLLM, junk_phrase
from src.logger import get_logger
from src.tts import _get_voice, speak, speak_phrase, stop_speaking
from src.vad import SpeechRecorder
from src.wakeword import WakewordDetector
from src.window_context import get_window_context

logger = get_logger("assistant")


def _is_false_positive(text: str) -> bool:
    """Отсекает ложные срабатывания: распознали только обращение."""
    normalized = text.lower().strip().replace(",", "").replace(".", "")
    return len(normalized.split()) <= 1 and "джарвис" in normalized


def _process_command(
    text: str,
    llm: JarvisLLM,
    detector: WakewordDetector,
    plan_command_func,
    _ask_fn,
    _speak_fn,
    _get_window_ctx,
):
    """Обрабатывает одну команду: ищет правило → LLM → ответ."""
    # Фильтруем ложные срабатывания
    if _is_false_positive(text):
        logger.warning("[ложное срабатывание — игнор]")
        return

    # Нечёткое исправление ошибок Vosk
    corrected = correct(text)
    if corrected != text:
        logger.info("🔧 ИИ исправил: %r → %r", text, corrected)
        text = corrected

    # Ищем команду или отправляем в LLM
    response, action, phrase_key = plan_command_func(text, llm)

    if response == LLM_MARKER:
        speak_phrase("llm_thinking")
        logger.info("🤔 Думаю (LLM)...")
        context = _get_window_ctx()
        if context:
            logger.info("📎 %s", context)
        answer = _ask_fn(llm, detector, text, context)
        if answer:
            phrase_key = junk_phrase(answer)
            if phrase_key:
                logger.info("🔹 Мусорный ответ — играю фразу: %s", phrase_key)
            _speak_fn(detector, answer, phrase_key)
        return

    if response is None and action is None:
        return

    if response is not None:
        _speak_fn(detector, response, phrase_key)
    if action is not None:
        try:
            action()
        except Exception as e:
            logger.error("[action error] %s", e)


def _run_conversation(
    llm,
    detector,
    recorder,
    plan_command_func,
    _ask_fn,
    _speak_fn,
    _get_window_ctx,
):
    """Режим диалога: 15 сек без «Джарвис».

    Слушает фразы через wakeword-детектор. Если видит wakeword —
    прерывает диалог. Если видит команду — обрабатывает.
    """
    logger.info("💬 Режим диалога: %d сек", CONVERSATION_TIMEOUT_SEC)
    deadline = time.monotonic() + CONVERSATION_TIMEOUT_SEC

    while time.monotonic() < deadline:
        # Ждём следующую фразу через _utterances() — он использует VAD-гейт
        interrupted = False
        for text in detector._utterances(deadline=deadline):
            # Если это wakeword — прерываем диалог
            if detector._match(text):
                logger.info("🎯 Wakeword в диалоге — выход")
                interrupted = True
                break

            # Обработка команды
            _process_command(
                text, llm, detector, plan_command_func,
                _ask_fn, _speak_fn, _get_window_ctx,
            )

        if interrupted:
            return True  # wakeword → начинаем новый цикл с «К вашим услугам»

    logger.info("⏰ Режим диалога завершён (таймаут)")
    return False  # таймаут → начинаем новый цикл с «К вашим услугам»


def _ask_with_interrupt(
    llm: JarvisLLM,
    detector: WakewordDetector,
    text: str,
    context: str | None = None,
) -> str | None:
    """Отправляет запрос в LLM, но прерывает по wakeword."""
    result = {"answer": None, "done": False, "error": None}

    def _do_ask():
        try:
            result["answer"] = llm.ask(text, context=context)
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    thread = threading.Thread(target=_do_ask, daemon=True)
    thread.start()

    interrupted = detector.wait_or_done(lambda: result["done"], timeout=30.0)

    if interrupted:
        logger.info("🛑 LLM прервана — начинаем заново")
        return None

    thread.join(timeout=1)
    if result["error"]:
        logger.error("[LLM error] %s", result["error"])
        return None
    return result["answer"]


def _speak_with_interrupt(detector: WakewordDetector, text: str, phrase_key: str | None = None):
    """Говорит текст, но если услышит 'Джарвис' — прерывает."""
    tts_thread = threading.Thread(target=speak, args=(text, phrase_key), daemon=True)
    tts_thread.start()

    interrupted = detector.wait_or_done(
        lambda: not tts_thread.is_alive(),
        timeout=30.0,
    )

    if interrupted:
        stop_speaking()
        logger.info("🛑 TTS прерван — начинаем заново")

    tts_thread.join(timeout=1)


def _warmup_llm(llm: JarvisLLM):
    """Прогревает модель без записи в историю."""
    logger.info("Прогреваю LLM...")
    try:
        history = llm.history
        llm.history = []
        llm.ask("привет")
        llm.history = history
        logger.info("LLM готова")
    except Exception as e:
        logger.warning("[LLM] прогрев не удался: %s", e)


def run():
    # ── Ядро ──
    llm = JarvisLLM()
    mic = Microphone()

    # Подписки: wakeword и VAD слушают ОДИН поток одновременно.
    ww_sub = mic.subscribe()
    vad_sub = mic.subscribe()

    detector = WakewordDetector(ww_sub)
    recorder = SpeechRecorder(vad_sub)

    # ── Прогрев ──
    logger.info("Прогреваю TTS...")
    _get_voice()
    logger.info("TTS готов")

    _warmup_llm(llm)

    logger.info("=" * 50)
    logger.info("         ДЖАРВИС ЗАПУЩЕН")
    logger.info("=" * 50)

    # ── Главный цикл ──
    with mic:
        while True:
            try:
                # 1. Ждём обращения
                wake_text = detector.wait()
                speak_phrase("welcome")

                # 2. Если пользователь сказал «Джарвис, открой код» —
                #    хвост фразы обрабатываем сразу, не тратя время на VAD.
                command_text = detector.strip_wakeword(wake_text).strip()

                if command_text:
                    text = command_text
                    logger.info("🗣️ Команда из хвоста wakeword: %s", text)
                else:
                    # 3. Записываем + распознаём параллельно (VAD + STT)
                    audio, text = recorder.record()
                    if not text:
                        continue

                # 4. Обрабатываем команду
                _process_command(
                    text, llm, detector, plan_command,
                    _ask_with_interrupt, _speak_with_interrupt,
                    get_window_context,
                )

                # 5. Режим диалога — слушаем следующие 15 сек без «Джарвис»
                speak_phrase("done")
                _run_conversation(
                    llm, detector, recorder, plan_command,
                    _ask_with_interrupt, _speak_with_interrupt,
                    get_window_context,
                )

            except MicrophoneError as e:
                logger.error("⚠️ Микрофон: %s — перезапуск через 3 сек", e)
                time.sleep(3)
            except KeyboardInterrupt:
                logger.info("\n👋 Выход")
                break
            except Exception:
                logger.error("⚠️ Ошибка цикла: %s", traceback.format_exc())
                time.sleep(1)