"""Silero TTS (локальный) + mpv с возможностью прерывания.

Синтез кэшируется на диск: hash(текст) → WAV в data/tts_cache, поэтому
повторные ответы играются мгновенно. Длинные ответы озвучиваются
по предложениям ещё во время генерации LLM (SentenceSpeaker).
"""

import hashlib
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from silero_tts.silero_tts import SileroTTS

from config import (
    CONFIRM_SOUND,
    EARCON_TAIL_SEC,
    SILERO_MODEL,
    SILERO_SAMPLE_RATE,
    SILERO_SPEAKER,
    TTS_CACHE_DIR,
    TTS_CACHE_MAX_FILES,
    TTS_STREAM_MIN_CHARS,
)

from src.audio import pause_input, resume_input
from src.logger import get_logger
from src.metrics import timed
from src.phrases import PHRASES, get_phrase_path, resolve_phrase_key

logger = get_logger("tts")


# ──────────────────────── СОСТОЯНИЕ ────────────────────────

_tts = None
_current_mpv: subprocess.Popen | None = None
_mpv_lock = threading.Lock()
_play_lock = threading.Lock()  # воспроизведения строго по очереди


def _echo_tail(path: str | Path) -> float:
    """Пауза после воспроизведения, чтобы эхо не попало в микрофон.

    Для коротких earcon (~17 КБ) хвост не нужен — экономия ~0.2 с на каждом
    подтверждении; для секундных речевых фраз оставляем прежние 200 мс.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return 0.2
    return EARCON_TAIL_SEC if size < 30_000 else 0.2


# ──────────────────────── ВОСПРОИЗВЕДЕНИЕ ────────────────────────


@timed("tts._play_now")
def _play_now(path: str | Path):
    """Проигрывает WAV через mpv с эхо-паузой микрофона.

    Пока играет — вход заглушен, чтобы Джарвис не распознавал
    собственный голос как команды. resume строго в finally.
    """
    global _current_mpv
    pause_input()
    try:
        with _mpv_lock:
            _current_mpv = subprocess.Popen(
                ["mpv", "--no-terminal", "--really-quiet", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mpv = _current_mpv
        mpv.wait()
        with _mpv_lock:
            if _current_mpv is mpv:
                _current_mpv = None
        time.sleep(_echo_tail(path))  # «хвост» эха в комнате/звуковом буфере
    finally:
        resume_input()


def _play_file(path: str | Path):
    """Ставит воспроизведение в общую очередь.

    Фраза-филлер, предложения стриминга и финальная реплика не
    накладываются друг на друга: _play_lock выстраивает их по порядку.
    """
    with _play_lock:
        _play_now(path)


@timed("tts._get_voice")
def _get_voice() -> SileroTTS:
    global _tts
    if _tts is None:
        logger.info("[TTS] Загружаю Silero (%s)...", SILERO_SPEAKER)
        _tts = SileroTTS(
            model_id=SILERO_MODEL,
            language="ru",
            speaker=SILERO_SPEAKER,
            sample_rate=SILERO_SAMPLE_RATE,
            device="cpu",
            put_accent=True,
            put_yo=True,
        )
        logger.info("[TTS] Silero загружен")
    return _tts


def stop_speaking():
    """Прерывает текущее воспроизведение mpv, если оно идёт."""
    global _current_mpv
    with _mpv_lock:
        if _current_mpv is not None and _current_mpv.poll() is None:
            logger.info("[TTS] Прерываю воспроизведение")
            _current_mpv.terminate()
            try:
                _current_mpv.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _current_mpv.kill()
        _current_mpv = None


# ──────────────────────── КЭШ СИНТЕЗА ────────────────────────


def _cache_path(text: str) -> Path:
    """Путь WAV в кэше для текста (подразумевается уже после _fix_text)."""
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return TTS_CACHE_DIR / f"{key}.wav"


def _trim_cache():
    """Держит кэш в пределах TTS_CACHE_MAX_FILES (удаляет старые по mtime)."""
    try:
        files = list(TTS_CACHE_DIR.glob("*.wav"))
        if len(files) <= TTS_CACHE_MAX_FILES:
            return
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[TTS_CACHE_MAX_FILES:]:
            old.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[TTS] чистка кэша не удалась: %s", e)


@timed("tts._synthesize")
def _synthesize(text: str, path: Path):
    """Синтезирует текст в WAV-файл кэша."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tts = _get_voice()
    tts.tts(text, str(path))
    _trim_cache()


# ──────────────────────── НОРМАЛИЗАЦИЯ ЧИСЕЛ ────────────────────────

_ONES = ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
)
_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
    "семьсот", "восемьсот", "девятьсот",
)
_OPER_WORDS = {"+": "плюс", "×": "умножить", "÷": "делить"}
_DIGIT_RE = re.compile(r"\d+")
_OPER_RE = re.compile(r"[+×÷]")


def _int_to_words(n: int) -> str:
    """Целое → русские слова (0–9999)."""
    if n == 0:
        return "ноль"
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _HUNDREDS[hundreds] + (f" {_int_to_words(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = _int_to_words(thousands) + " тысяч" if thousands > 1 else ("тысяча" if thousands == 1 else "")
    if not rest:
        return head
    return f"{head} {_int_to_words(rest)}".strip()


def _normalize_numbers(text: str) -> str:
    """Нормализация чисел: цифры → слова («23» → «двадцать три», «2+2» → «два плюс два»)."""
    text = _OPER_RE.sub(lambda m: f" {_OPER_WORDS[m.group(0)]} ", text)
    return _DIGIT_RE.sub(lambda m: _int_to_words(int(m.group(0))), text)


# ──────────────────────── ОЧИСТКА ТЕКСТА ────────────────────────


def _fix_text(text: str) -> str:
    """Исправляет ударения и чистит символы для Silero."""
    # Ударения (Silero понимает + перед ударной гласной)
    stress_fixes = {
        "Джарвис": "Дж+арвис",
        "джарвис": "Дж+арвис",
    }
    for wrong, right in stress_fixes.items():
        text = text.replace(wrong, right)

    # Чистка символов, которые Silero читает плохо
    symbol_fixes = {
        "/": " ",           # косая черта → пробел
        "_": " ",           # подчёркивание → пробел
        "~": " ",           # тильда → пробел
        "%": " процентов ",
        "&": " и ",
        "@": " собака ",
        "#": " номер ",
        "№": " номер ",
        "°C": " градусов Цельсия",
        "°": " градусов ",
    }
    for wrong, right in symbol_fixes.items():
        text = text.replace(wrong, right)

    # Нормализация чисел: цифры → слова («23» → «двадцать три», «2+2» → «два плюс два»).
    text = _normalize_numbers(text)

    # Убираем двойные пробелы после замен
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────── ФРАЗЫ (0 сек синтеза) ────────────────────────


def speak_phrase(key: str) -> bool:
    """Проигрывает готовую фразу из src/phrases/. True если файл найден.

    При CONFIRM_SOUND="earcon" подтверждающие фразы (welcome, llm_thinking,
    done) заменяются коротким звуком; при "off" не звучат вовсе.
    Остальные ключи (not_listened, error, volume_*) играют как раньше.
    """
    key = resolve_phrase_key(key)
    if key is None:
        return False
    path = get_phrase_path(key)
    if path is None or not path.exists():
        return False

    text = PHRASES.get(key, (None, "?"))[1]
    logger.info("🔊 Джарвис (фраза): %s", text)

    _play_file(path)
    return True


# ──────────────────────── СИНТЕЗ + ВОСПРОИЗВЕДЕНИЕ ────────────────────────


@timed("tts.speak")
def speak(text: str, phrase_key: str | None = None):
    if not text or not text.strip():
        logger.warning("[TTS] пустой текст — не озвучиваю")
        return
    """Блокирующий синтез + воспроизведение. Прерывается через stop_speaking().

    Если phrase_key задан и WAV-файл существует — играет mpv за 0 сек.
    Иначе — Silero TTS с дисковым кэшем: тот же текст второй раз
    играет мгновенно, без синтеза.
    """
    if phrase_key and speak_phrase(phrase_key):
        return

    text = _fix_text(text)
    logger.info("🔊 Джарвис: %s", text)
    try:
        wav = _cache_path(text)
        if not wav.exists():
            _synthesize(text, wav)
        _play_file(wav)
    except Exception as e:
        logger.error("[TTS error] %s", e)


# ──────────────────────── СТРИМИНГ ОТВЕТА ────────────────────────

# Граница предложения: . ! ? … с закрывающими кавычками/скобками.
# Точка внутри числа («2.5») границей не считается.
_SENT_END = re.compile(r"[.!?…]+[\»)\"'\]]*")


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    """Достаёт из буфера завершённые предложения. Возвращает (их, остаток)."""
    sentences: list[str] = []
    pos = 0
    while True:
        m = _SENT_END.search(buf, pos)
        if m is None:
            break
        end = m.end()
        if end < len(buf) and buf[end].isdigit():
            pos = end  # «2.5» — не граница, ищем дальше
            continue
        sent = buf[:end].strip()
        buf = buf[end:].lstrip()
        pos = 0
        if sent:
            sentences.append(sent)
    return sentences, buf


class SentenceSpeaker:
    """Озвучивает ответ LLM по предложениям, не дожидаясь его конца.

    feed() получает куски текста из стриминга; когда в буфере набралось
    не меньше TTS_STREAM_MIN_CHARS, готовые предложения синтезируются и
    уходят в очередь на воспроизведение (синтез — в кэш, как у speak()).

    Короткие ответы (меньше порога) вообще не озвучиваются здесь:
    они остаются обычному пути speak() — так мусорные ответы
    («Не расслышал») заменяются готовой фразой, а короткая реплика
    не рвётся на куски.
    """

    def __init__(self, min_start_chars: int = TTS_STREAM_MIN_CHARS):
        self._min_start_chars = min_start_chars
        self._buf = ""
        self._spoken = False
        self._stopped = threading.Event()
        self._q: "queue.Queue[Path | None]" = queue.Queue()
        self._player = threading.Thread(target=self._player_loop, daemon=True)
        self._player.start()

    # ── публичное ──

    def feed(self, delta: str):
        """Принимает очередной кусок ответа (вызывается из потока LLM)."""
        if self._stopped.is_set() or not delta:
            return
        self._buf += delta
        if not self._spoken and len(self._buf) < self._min_start_chars:
            return
        sentences, self._buf = _pop_sentences(self._buf)
        for sent in sentences:
            self._speak_sentence(sent)

    def started(self) -> bool:
        """Хотя бы одно предложение уже ушло в озвучку."""
        return self._spoken

    def finish(self):
        """Договаривает хвост, если озвучка уже началась."""
        if not self._spoken or self._stopped.is_set():
            self._close_player()
            return
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._speak_sentence(rest)
        self._close_player()

    def stop(self):
        """Прерывает озвучку: сбрасывает очередь и текущее воспроизведение."""
        self._stopped.set()
        self._drain_queue()
        stop_speaking()
        self._q.put(None)
        self._player.join(timeout=2)

    # ── внутреннее ──

    def _speak_sentence(self, sent: str):
        if self._stopped.is_set():
            return
        self._spoken = True
        try:
            fixed = _fix_text(sent)
            wav = _cache_path(fixed)
            if not wav.exists():
                _synthesize(fixed, wav)
            self._q.put(wav)
        except Exception as e:
            logger.error("[TTS stream error] %s", e)

    def _player_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            with _play_lock:
                if self._stopped.is_set():
                    return
                _play_now(item)

    def _drain_queue(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _close_player(self):
        self._q.put(None)
        self._player.join()
