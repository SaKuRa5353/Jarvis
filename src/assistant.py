"""Оркестратор — главный цикл ассистента с прерыванием.

Режим диалога: после первого обращения (wake word + команда) ассистент
остаётся «на связи» и слушает следующие 15 секунд без повторного «Джарвис».
Если за это время пользователь скажет «Джарвис» — диалог прерывается.

Фразы-филлеры не тормозят обработку: LLM-запрос стартует немедленно,
а «Секунду» / «К вашим услугам» играют параллельно размышлению.
Длинные ответы озвучиваются по предложениям во время генерации LLM.
"""

import atexit
import threading
import time
import traceback

from config import CONVERSATION_TIMEOUT_SEC

from src.audio import Microphone, MicrophoneError
from src.commands import LLM_MARKER, plan_command
from src.fuzzy import correct
from src.llm import JarvisLLM
from src.logger import get_logger
from src.metrics import metrics
from src.tts import (
    SentenceSpeaker,
    _get_voice,
    speak,
    speak_phrase,
    stop_speaking,
)
from src.vad import SpeechRecorder
from src.wakeword import WakewordDetector
from src.window_context import get_window_context

logger = get_logger("assistant")


def _is_false_positive(text: str) -> bool:
    """Отсекает ложные срабатывания: распознали только обращение."""
    normalized = text.lower().strip().replace(",", "").replace(".", "")
    return len(normalized.split()) <= 1 and "джарвис" in normalized


# ──────────────────── РАЗМЫШЛЕНИЕ И ВЫДАЧА ────────────────────


def _plan_reply(text: str, llm: JarvisLLM, plan_command_func):
    """Размышление над репликой: фильтр, fuzzy-correct, поиск по правилам.

    Возвращает (corrected, response, action, phrase_key) или None,
    если реплика отброшена как ложное срабатывание.
    """
    if _is_false_positive(text):
        logger.warning("[ложное срабатывание — игнор]")
        return None

    corrected = correct(text)
    if corrected != text:
        logger.info("🔧 ИИ исправил: %r → %r", text, corrected)

    response, action, phrase_key = plan_command_func(corrected, llm)
    return corrected, response, action, phrase_key


def _run_plan(response, action, phrase_key, detector, _speak_fn):
    """Исполняет готовый план: реплика + действие."""
    if response is None and action is None:
        return

    if response is not None:
        _speak_fn(detector, response, phrase_key)
    if action is not None:
        try:
            action()
        except Exception as e:
            logger.error("[action error] %s", e)


def _speak_llm_answer(ask: "_AskHandle", detector, _speak_fn):
    """Озвучивает результат LLM-запроса (handle завершён или брошен).

    Длинные ответы уже озвучены стримингом — остаётся хвост (finish).
    Короткие и мусорные идут обычным путём: готовая фраза или speak().
    """
    if ask.abandoned:
        return
    result = ask.result()
    if not result:
        ask.speaker.stop()
        return
    answer, phrase_key = result

    already = ask.speaker.started()
    ask.speaker.finish()  # договаривает хвост, если озвучка началась
    if already:
        return

    if phrase_key == "not_listened":
        logger.info("🔹 Мусорный ответ — играю фразу: %s", phrase_key)
    _speak_fn(detector, answer, phrase_key)


def _process_command(
    text: str,
    llm: JarvisLLM,
    detector: WakewordDetector,
    plan_command_func,
    _ask_fn,
    _speak_fn,
    _get_window_ctx,
):
    """Обрабатывает одну команду: правила → LLM → ответ."""
    plan = _plan_reply(text, llm, plan_command_func)
    if plan is None:
        return
    corrected, response, action, phrase_key = plan

    if response == LLM_MARKER:
        logger.info("🤔 Думаю (LLM)...")
        context = _get_window_ctx()
        if context:
            logger.info("📎 %s", context)
        # Запрос стартует сразу; фраза «Секунду» играет параллельно ему.
        ask = _ask_fn(llm, detector, corrected, context, thinking_phrase="llm_thinking")
        _speak_llm_answer(ask, detector, _speak_fn)
        return

    _run_plan(response, action, phrase_key, detector, _speak_fn)


def _process_tail_command(
    text: str,
    llm: JarvisLLM,
    detector: WakewordDetector,
    plan_command_func,
    _get_window_ctx,
):
    """Команда, сказанная вместе с обращением («Джарвис, открой код»).

    Правила и LLM-запрос стартуют немедленно и идут параллельно
    приветствию — ответ готов почти сразу после «К вашим услугам».
    """
    box: dict = {}

    def _think():
        try:
            plan = _plan_reply(text, llm, plan_command_func)
            if plan is None:
                return
            corrected, response, *_ = plan
            if response == LLM_MARKER:
                context = _get_window_ctx()
                if context:
                    logger.info("📎 %s", context)
                logger.info("🤔 Думаю (LLM)...")
                box["ask"] = _AskHandle(llm, corrected, context)
            box["plan"] = plan
        except Exception:
            logger.error("⚠️ Ошибка размышления: %s", traceback.format_exc())

    thread = threading.Thread(target=_think, daemon=True)
    thread.start()

    # Приветствие играет, пока в фоне идут правила и LLM.
    speak_phrase("welcome")
    thread.join()

    plan = box.get("plan")
    if plan is None:
        return
    corrected, response, action, phrase_key = plan

    if response == LLM_MARKER:
        ask: _AskHandle | None = box.get("ask")
        if ask is None:
            return
        # LLM ещё думает после приветствия — заполняем паузу фразой.
        if not ask.done:
            speak_phrase("llm_thinking")
        if detector.wait_or_done(lambda: ask.done, timeout=30.0):
            logger.info("🛑 LLM прервана — начинаем заново")
            ask.abandoned = True
            ask.stop_audio()
            return
        _speak_llm_answer(ask, detector, _speak_with_interrupt)
        return

    _run_plan(response, action, phrase_key, detector, _speak_with_interrupt)


# ──────────────────── LLM-ЗАПРОСЫ ────────────────────


class _AskHandle:
    """Запущенный LLM-запрос: стартует немедленно, результат забирают позже.

    Длинный ответ озвучивается по предложениям прямо во время генерации
    (SentenceSpeaker). Пока модель думает, можно играть фразу-филлер
    или приветствие — воспроизведение выстроено в общую очередь.
    """

    def __init__(self, llm: JarvisLLM, text: str, context: str | None = None):
        self._answer: tuple[str, str | None] | None = None
        self._error: Exception | None = None
        self.done = False
        self.abandoned = False  # прерван по wakeword — озвучку погасить
        self.speaker = SentenceSpeaker()

        def _do_ask():
            try:
                self._answer = llm.ask(text, context=context, on_delta=self.speaker.feed)
            except Exception as e:
                self._error = e
            finally:
                self.done = True

        self.thread = threading.Thread(target=_do_ask, daemon=True)
        self.thread.start()

    def result(self) -> tuple[str, str | None] | None:
        """Ответ LLM; None при ошибке (ошибка логируется здесь же)."""
        if self._error is not None:
            logger.error("[LLM error] %s", self._error)
            return None
        return self._answer

    def stop_audio(self):
        """Гасит стриминговую озвучку (прерывание по wakeword)."""
        self.speaker.stop()


def _ask_with_interrupt(
    llm: JarvisLLM,
    detector: WakewordDetector,
    text: str,
    context: str | None = None,
    thinking_phrase: str | None = None,
) -> "_AskHandle":
    """LLM-запрос с прерыванием по wakeword.

    Запрос стартует немедленно; фраза-филлер играет параллельно ожиданию.
    Возвращает handle — завершённый или брошенный (abandoned, озвучку
    которого уже погасили).
    """
    ask = _AskHandle(llm, text, context)
    if thinking_phrase:
        speak_phrase(thinking_phrase)

    if detector.wait_or_done(lambda: ask.done, timeout=30.0):
        logger.info("🛑 LLM прервана — начинаем заново")
        ask.stop_audio()
        ask.abandoned = True
    return ask


def _ask_no_interrupt(
    llm: JarvisLLM,
    _detector: WakewordDetector,
    text: str,
    context: str | None = None,
    thinking_phrase: str | None = None,
) -> "_AskHandle":
    """LLM-запрос без прерывания (для режима диалога). Фраза — параллельно."""
    ask = _AskHandle(llm, text, context)
    if thinking_phrase:
        speak_phrase(thinking_phrase)
    ask.thread.join()
    return ask


# ──────────────────── ОЗВУЧКА ────────────────────


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


def _speak_no_interrupt(_detector: WakewordDetector, text: str, phrase_key: str | None = None):
    """Говорит текст без прерывания (для режима диалога, где один читатель ww_sub)."""
    speak(text, phrase_key)


# ──────────────────── РЕЖИМ ДИАЛОГА ────────────────────


def _run_conversation(
    llm,
    detector,
    recorder,
    plan_command_func,
):
    """Режим диалога: 15 сек без «Джарвис».

    Слушает фразы через wakeword-детектор. Если видит wakeword —
    прерывает диалог. Если видит команду — обрабатывает.

    ВАЖНО: используем версии без прерывания (_speak_no_interrupt /
    _ask_no_interrupt), чтобы не создавать второй читатель на ww_sub —
    внешний цикл и так ловит «Джарвис» и выходит из диалога.
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

            # Обработка команды (без прерывания по wakeword)
            _process_command(
                text, llm, detector, plan_command_func,
                _ask_no_interrupt, _speak_no_interrupt,
                get_window_context,
            )

        if interrupted:
            return True  # wakeword → начинаем новый цикл с «К вашим услугам»

    logger.info("⏰ Режим диалога завершён (таймаут)")
    return False  # таймаут → начинаем новый цикл с «К вашим услугам»


# ──────────────────── ГЛАВНЫЙ ЦИКЛ ────────────────────


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

    # Отчёт метрик при любом выходе из процесса (в т.ч. SIGTERM, SystemExit).
    # reset=False: KeyboardInterrupt в main.py уже мог напечатать отчёт,
    # а повторная печать очищенного реестра ничего не даёт.
    atexit.register(lambda: metrics.log_report(reset=False))

    # ── Главный цикл ──
    with mic:
        while True:
            try:
                # 1. Ждём обращения
                wake_text = detector.wait()

                # 2. Если пользователь сказал «Джарвис, открой код» —
                #    команда уже здесь: размышление и LLM-запрос стартуют
                #    сразу, параллельно приветствию (VAD не нужен).
                command_text = detector.strip_wakeword(wake_text).strip()

                if command_text:
                    logger.info("🗣️ Команда из хвоста wakeword: %s", command_text)
                    _process_tail_command(
                        command_text, llm, detector, plan_command,
                        get_window_context,
                    )
                else:
                    # 3. Приветствие, затем запись + распознавание (VAD + STT)
                    speak_phrase("welcome")
                    audio, text = recorder.record()
                    if not text:
                        continue

                    # 4. Обрабатываем команду
                    _process_command(
                        text, llm, detector, plan_command,
                        _ask_with_interrupt, _speak_with_interrupt,
                        get_window_context,
                    )

                # 5. Режим диалога — слушаем следующие 15 сек без «Джарвис».
                #    Фраза-подтверждение не играется: команду уже озвучил
                #    _process_command, а после LLM-ответа реплика и так есть.
                _run_conversation(
                    llm, detector, recorder, plan_command,
                )

            except MicrophoneError as e:
                logger.error("⚠️ Микрофон: %s — перезапуск через 3 сек", e)
                time.sleep(3)
            except KeyboardInterrupt:
                metrics.log_report()
                logger.info("\n👋 Выход")
                break
            except Exception:
                logger.error("⚠️ Ошибка цикла: %s", traceback.format_exc())
                time.sleep(1)
