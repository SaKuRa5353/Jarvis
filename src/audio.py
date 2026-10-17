"""Единый ввод с микрофона.

Один открытый поток на всё время работы ассистента + фоновый читатель,
складывающий блоки в буфер. Потребители (wakeword, VAD) забирают блоки
из буфера и не открывают микрофон сами.

Зачем: раньше wakeword и VAD по очереди открывали свой InputStream.
Между закрытием одного и открытием другого речь терялась, а открытие
потока занимает десятки миллисекунд.
"""

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


class MicrophoneError(RuntimeError):
    """Микрофон умер (устройство пропало, поток упал)."""


class Microphone:
    """Непрерывный поток с микрофона, отдающий блоки сэмплов int16."""

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

        max_blocks = max(4, int(buffer_sec * sample_rate / block_size))
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_blocks)

        self._stream: sd.RawInputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

        self._leftover = np.zeros(0, dtype=np.int16)
        self._read_lock = threading.Lock()

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

    # ──────────────────── ЧТЕНИЕ ────────────────────

    def read(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """
        Возвращает ровно n_samples сэмплов int16.

        Блокируется, пока данные не наберутся. timeout — сколько секунд
        ждать один блок; None = ждать вечно. При таймауте уже прочитанное
        возвращается в буфер, чтобы не терять аудио.
        """
        parts: list[np.ndarray] = []
        got = 0

        with self._read_lock:
            if len(self._leftover):
                parts.append(self._leftover)
                got = len(self._leftover)
                self._leftover = np.zeros(0, dtype=np.int16)

            try:
                while got < n_samples:
                    chunk = self._get_block(timeout)
                    if len(chunk) > n_samples - got:
                        self._leftover = chunk[n_samples - got :]
                        chunk = chunk[: n_samples - got]
                    parts.append(chunk)
                    got += len(chunk)
            except TimeoutError:
                # Возвращаем недочитанное обратно, данные не теряются.
                if parts:
                    rest = parts[-1]
                    self._leftover = (
                        rest if not len(self._leftover)
                        else np.concatenate([rest, self._leftover])
                    )
                raise

        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def drain(self):
        """Выбрасывает накопленное аудио. Вызывать при смене фазы цикла,
        чтобы VAD не начал с речи, сказанной во время wakeword."""
        with self._read_lock:
            self._leftover = np.zeros(0, dtype=np.int16)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    # ──────────────────── ВНУТРЕННЕЕ ────────────────────

    def _get_block(self, timeout: float | None) -> np.ndarray:
        while True:
            if self._error is not None:
                raise MicrophoneError(self._error)
            try:
                raw = self._queue.get(timeout=timeout if timeout else 0.5)
            except queue.Empty:
                if timeout is not None:
                    raise TimeoutError
                continue
            return np.frombuffer(raw, dtype=np.int16)

    def _reader_loop(self):
        """Фоновый читатель: микрофон -> буфер."""
        stream = self._stream
        while not self._stop.is_set():
            try:
                data, overflowed = stream.read(self.block_size)
            except Exception as e:
                self._error = f"поток микрофона упал: {e}"
                logger.error("[mic] %s", self._error)
                return

            if overflowed:
                logger.debug("[mic] overflow — часть аудио пропущена")

            raw = bytes(data)

            if self._queue.full():
                # Буфер переполнен —consumer отстал. Выбрасываем самый
                # старый блок, чтобы задержка оставалась минимальной.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(raw)
            except queue.Full:
                pass
