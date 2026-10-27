"""Единый ввод с микрофона.

Один открытый поток на всё время работы ассистента + BroadcastBuffer,
который копирует каждый блок аудио во все очереди подписчиков.

Зачем: раньше wakeword и VAD по очереди открывали свой InputStream.
Между закрытием одного и открытием другого речь терялась.
Теперь оба потребителя слушают один поток одновременно, что позволяет
запускать STT параллельно с VAD и экономить ~1 секунду на ответ.
"""

import collections
import queue
import threading

import numpy as np
import sounddevice as sd

from config import (
    AUDIO_BLOCK_SIZE,
    AUDIO_BUFFER_SEC,
    DEVICE,
    SAMPLE_RATE,
)

from src.logger import get_logger

logger = get_logger("audio")


# ──────────────────── ПАУЗА ВХОДА (эхо-защита) ────────────────────
# Пока Джарвис говорит, его голос из колонки попадает в микрофон и
# распознаётся как команда («Делаю громче» → снова «громче» → петля).
# На время воспроизведения фан-аут перестает раздавать блоки.

_input_paused = threading.Event()


def pause_input():
    """Перестать раздавать аудио подписчикам (идёт озвучка)."""
    _input_paused.set()


def resume_input():
    """Снова раздавать аудио (озвучка закончилась)."""
    _input_paused.clear()


def input_paused() -> bool:
    return _input_paused.is_set()


class BroadcastBuffer:
    """Холдит скользящее окно блоков аудио, раздавая копии подписчикам."""

    def __init__(self, maxlen: int = 1000, mic: "Microphone | None" = None):
        self._buffer: collections.deque[bytes] = collections.deque(maxlen=maxlen)
        self._subscribers: list["Subscriber"] = []
        self._lock = threading.Lock()
        self._mic = mic

    def check(self):
        if self._mic is not None:
            self._mic.check()

    def push(self, block: bytes):
        """Добавляет блок и разсылает подписчикам.

        Во время паузы (говорит TTS) блок не попадает ни в историю,
        ни в очереди подписчиков — эхо собственного голоса игнорируется.
        """
        if _input_paused.is_set():
            return
        self._buffer.append(block)
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            sub._offer(block)

    def subscribe(self, maxsize: int = 100) -> "Subscriber":
        """Возвращает подписчика, который можно читать как старый mic.read()."""
        sub = Subscriber(self, maxsize)
        with self._lock:
            self._subscribers.append(sub)
        # Заполняем историей (чтобы не пропустить то, что уже записано)
        for block in self._buffer:
            sub._offer(block)
        return sub

    def unsubscribe(self, sub: "Subscriber"):
        """Отписывает подписчика от буфера."""
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)


class Subscriber:
    """Потребитель одного канала BroadcastBuffer.

    Держит свою очередь и leftover-буфер, чтобы читать ровно n_samples
    так же, как раньше работал mic.read().
    """

    def __init__(self, broadcast: BroadcastBuffer, maxsize: int = 100):
        self._broadcast = broadcast
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)
        self._leftover = np.zeros(0, dtype=np.int16)

    # ── внутреннее для BroadcastBuffer ──

    def _offer(self, block: bytes):
        if self._q.full():
            try:
                self._q.get_nowait()  # отстал — выбрасываем старый блок
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(block)
        except queue.Full:
            pass

    # ── публичное ──

    def read(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """Возвращает ровно n_samples сэмплов int16.

        timeout — сколько секунд ждать один блок; None = ждать вечно.
        При таймауте уже прочитанное возвращается в буфер.
        """
        parts: list[np.ndarray] = []
        got = 0

        if len(self._leftover):
            parts.append(self._leftover)
            got = len(self._leftover)
            self._leftover = np.zeros(0, dtype=np.int16)

        try:
            while got < n_samples:
                raw = self._q.get(timeout=timeout if timeout else 0.5)
                chunk = np.frombuffer(raw, dtype=np.int16)
                if len(chunk) > n_samples - got:
                    self._leftover = chunk[n_samples - got:]
                    chunk = chunk[: n_samples - got]
                parts.append(chunk)
                got += len(chunk)
        except queue.Empty:
            # Возвращаем недочитанное обратно, данные не теряются.
            if parts:
                rest = parts[-1]
                self._leftover = (
                    rest if not len(self._leftover)
                    else np.concatenate([rest, self._leftover])
                )
            raise TimeoutError from None

        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def read_bytes(self, n_samples: int, timeout: float | None = None) -> bytes:
        """То же, что read(), но сразу в bytes для Vosk."""
        return self.read(n_samples, timeout).tobytes()

    def drain(self):
        """Выбрасывает накопленное аудио."""
        self._leftover = np.zeros(0, dtype=np.int16)
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def check(self):
        """Бросает MicrophoneError, если поток читателя умер."""
        self._broadcast.check()

    def close(self):
        """Отписывает от буфера."""
        self._broadcast.unsubscribe(self)


class MicrophoneError(RuntimeError):
    """Микрофон умер (устройство пропало, поток упал)."""


class Microphone:
    """Непрерывный поток с микрофона, раздающий блоки через BroadcastBuffer."""

    def __init__(
        self,
        device: int = DEVICE,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = AUDIO_BLOCK_SIZE,
        buffer_sec: float = AUDIO_BUFFER_SEC,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size

        # 1000 блоков * 1280 сэмплов = ~64 сек истории
        self._broadcast = BroadcastBuffer(maxlen=1000, mic=self)

        self._stream: sd.RawInputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    # ──────────────────── ЖИЗНЕННЫЙ ЦИКЛ ────────────────────

    def start(self) -> "Microphone":
        """Открывает поток и запускает фоновый читатель."""
        if self._stream is not None:
            return self

        self._stop.clear()
        self._error = None

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            dtype="int16",
            channels=1,
        )
        self._stream.start()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="mic-reader",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "🎙️ Микрофон #%s открыт (%d Hz, блок %d)",
            self.device,
            self.sample_rate,
            self.block_size,
        )
        return self

    def stop(self):
        """Останавливает читателя и закрывает поток."""
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("[mic] ошибка при закрытии потока: %s", e)
            self._stream = None

        self.drain()
        logger.info("🎙️ Микрофон закрыт")

    def __enter__(self) -> "Microphone":
        return self.start()

    def __exit__(self, *_):
        self.stop()

    # ──────────────────── ПОДПИСКА ────────────────────

    def subscribe(self, maxsize: int = 100) -> Subscriber:
        """Возвращает подписчика, который можно читать как старый mic.read()."""
        return self._broadcast.subscribe(maxsize)

    def unsubscribe(self, sub: Subscriber):
        """Отписывает подписчика от буфера."""
        self._broadcast.unsubscribe(sub)

    def drain(self):
        """Выбрасывает накопленное аудио у всех подписчиков и из истории.
        Вызывать при смене фазы цикла, чтобы VAD не начинал с речи,
        сказанной во время wakeword."""
        self._broadcast._buffer.clear()
        with self._broadcast._lock:
            subs = list(self._broadcast._subscribers)
        for sub in subs:
            sub.drain()

    def check(self):
        """Бросает MicrophoneError, если поток читателя умер."""
        if self._error is not None:
            raise MicrophoneError(self._error)

    # ──────────────────── ВНУТРЕННЕЕ ────────────────────

    def _reader_loop(self):
        """Фоновый читатель: микрофон -> BroadcastBuffer."""
        stream = self._stream
        while not self._stop.is_set():
            try:
                data, overflowed = stream.read(self.block_size)
            except Exception as e:
                if self._stop.is_set():
                    # штатная остановка — не считаем это падением потока
                    break
                self._error = f"поток микрофона упал: {e}"
                logger.error("[mic] %s", self._error)
                return

            if overflowed:
                logger.debug("[mic] overflow — часть аудио пропущена")

            raw = bytes(data)
            self._broadcast.push(raw)
