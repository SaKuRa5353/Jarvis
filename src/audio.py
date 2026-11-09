"""Единый ввод с микрофона.

Один открытый поток на всё время работы ассистента + BroadcastBuffer,
который копирует каждый блок аудио во все очереди подписчиков.

Зачем: раньше wakeword и VAD по очереди открывали свой InputStream.
Между закрытием одного и открытием другого речь терялась.
Теперь оба потребителя слушают один поток одновременно, что позволяет
запускать STT параллельно с VAD и экономить ~1 секунду на ответ.
"""

from __future__ import annotations

import collections
import contextlib
import queue
import re
import threading
from typing import Any

import numpy as np

from config import (
    AUDIO_BLOCK_SIZE,
    AUDIO_BUFFER_SEC,
    DEVICE,
    DEVICE_NAME,
    HOSTAPI,
    SAMPLE_RATE,
)
from src.logger import get_logger

logger = get_logger("audio")


# ──────────────────── ПАУЗА ВХОДА (эхо-защита) ────────────────────
# Пока Джарвис говорит, его голос из колонки попадает в микрофон и
# распознаётся как команда («Делаю громче» → снова «громче» → петля).
# На время воспроизведения фан-аут перестает раздавать блоки.


class StreamResampler:
    """Потоковый ресемплер с фазово выровненной фильтровой историей."""

    def __init__(self, orig_sr: int, target_sr: int, quality_window_len: int = 64) -> None:
        import math

        from scipy.signal import resample_poly

        self.orig_sr = orig_sr
        self.target_sr = target_sr
        g = math.gcd(target_sr, orig_sr)
        self.up = target_sr // g
        self.down = orig_sr // g
        if self.up == self.down == 1:
            self._flushed = False
            return

        half_len = 10 * max(self.up, self.down)
        pre_pad = self.down - half_len % self.down
        filter_len = 2 * half_len + 1 + pre_pad
        # Один радиус нужен для будущего выхода, а запас в четыре длины
        # фильтра скрывает искусственную границу внутреннего resample_poly.
        required_context = int(np.ceil(2 * filter_len / self.up))
        context_len = int(np.ceil(4 * filter_len / self.up))
        self.overlap_len = max(quality_window_len, context_len)
        self.overlap_len = (
            (self.overlap_len + self.down - 1) // self.down
        ) * self.down
        self._future_len = required_context
        self._history = np.zeros(0, dtype=np.int16)
        self._history_start = 0
        self._input_total = 0
        self._output_total = 0
        self._flushed = False
        self._resample_poly = resample_poly

    def resample(self, block_int16: np.ndarray) -> np.ndarray:
        """Ресемплирует блок; хвост фильтра выдаётся методом ``flush``."""
        if self.orig_sr == self.target_sr:
            return block_int16.copy()
        if self._flushed:
            raise RuntimeError("Нельзя добавлять блок после flush()")
        if block_int16.ndim != 1 or block_int16.dtype != np.int16:
            raise ValueError("Ожидается одномерный массив dtype=int16")

        previous_total = self._input_total
        combined = np.concatenate([self._history, block_int16])
        segment_start = self._history_start
        self._input_total += len(block_int16)
        return self._process_segment(
            combined,
            segment_start=segment_start,
            previous_total=previous_total,
            stable_end=max(
                self._output_total,
                (self._input_total - self._future_len) * self.up // self.down,
            ),
        )

    def flush(self) -> np.ndarray:
        """Выдаёт задержанный хвост, дополняя вход нулями."""
        if self.orig_sr == self.target_sr or self._flushed:
            return np.zeros(0, dtype=np.int16)
        self._flushed = True
        expected_end = (self._input_total * self.up + self.down - 1) // self.down
        combined = np.concatenate(
            [self._history, np.zeros(self._future_len, dtype=np.int16)]
        )
        return self._process_segment(
            combined,
            segment_start=self._history_start,
            previous_total=self._input_total,
            stable_end=expected_end,
        )

    def _process_segment(
        self,
        combined: np.ndarray,
        *,
        segment_start: int,
        previous_total: int,
        stable_end: int,
    ) -> np.ndarray:
        """Обрабатывает сегмент и сохраняет правый фильтровый контекст."""
        segment_output = self._resample_poly(
            combined.astype(np.float32), self.up, self.down
        )
        segment_offset = segment_start * self.up // self.down
        start = max(self._output_total, segment_offset)
        end = min(stable_end, segment_offset + len(segment_output))
        if end <= start:
            result = np.zeros(0, dtype=np.int16)
        else:
            result = segment_output[start - segment_offset : end - segment_offset]
            result = np.clip(result, -32768, 32767).astype(np.int16)
            self._output_total = end

        if not self._flushed:
            end_input = previous_total + len(combined) - len(self._history)
            self._save_history(combined, segment_start, end_input)
        return result

    def _save_history(self, combined: np.ndarray, segment_start: int, end: int) -> None:
        """Сохраняет правый контекст с началом на границе down."""
        new_start = max(0, ((end - self.overlap_len) // self.down) * self.down)
        offset = new_start - segment_start
        self._history = combined[offset:]
        self._history_start = new_start


_input_paused = threading.Event()


def _normalized_device_name(name: str) -> str:
    """Убирает меняющуюся ALSA-пару ``(hw:card,device)`` из имени."""
    normalized = re.sub(r"\s*\(hw:\d+,\d+\)\s*$", "", name.strip(), flags=re.IGNORECASE)
    return " ".join(normalized.split()).casefold()


def _hostapi_name(sd: Any, device: dict[str, Any]) -> str:
    """Возвращает имя host API без падения на неполной fake sounddevice."""
    hostapi = device.get("hostapi")
    try:
        hostapis = sd.query_hostapis()
        if isinstance(hostapi, int) and 0 <= hostapi < len(hostapis):
            return str(hostapis[hostapi].get("name", hostapi))
    except Exception:
        pass
    return str(hostapi if hostapi is not None else "неизвестно")


def _hostapi_matches(
    sd: Any, device: dict[str, Any], requested_hostapi: str | int
) -> bool:
    requested = str(requested_hostapi).strip().casefold()
    actual = str(device.get("hostapi", "")).casefold()
    return requested == actual or requested == _hostapi_name(sd, device).casefold()


def select_input_device(
    sd: Any,
    requested_device: int | None = None,
    requested_name: str | None = None,
    requested_hostapi: str | int | None = None,
) -> int:
    """Выбирает индекс входного устройства на текущем запуске.

    Приоритет: устойчивое имя (+ host API), legacy-индекс, системный input.
    Имя сравнивается без ALSA-суффикса ``(hw:N,M)``. Неоднозначный результат
    никогда не разрешается угадыванием.
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise MicrophoneError(f"Не удалось получить список аудиоустройств: {exc}") from exc
    if not devices:
        raise MicrophoneError("В системе не найдено ни одного аудиоустройства.")

    if requested_name:
        inputs = [
            (index, info)
            for index, info in enumerate(devices)
            if int(info.get("max_input_channels", 0)) > 0
        ]
        if not inputs:
            raise MicrophoneError("В списке нет устройств с входными каналами (микрофонов).")
        normalized = _normalized_device_name(requested_name)
        matches = [
            (index, info)
            for index, info in inputs
            if _normalized_device_name(str(info.get("name", ""))) == normalized
        ]
        if requested_hostapi is not None:
            matches = [
                (index, info)
                for index, info in matches
                if _hostapi_matches(sd, info, requested_hostapi)
            ]
        if len(matches) == 1:
            return matches[0][0]
        if not matches:
            suffix = f" и host API {requested_hostapi!r}" if requested_hostapi else ""
            raise MicrophoneError(f"Устройство с именем {requested_name!r}{suffix} не найдено.")
        details = ", ".join(f"#{index} {info.get('name', '')}" for index, info in matches)
        raise MicrophoneError(
            f"Имя устройства {requested_name!r} неоднозначно: {details}. "
            "Укажите JARVIS_HOSTAPI."
        )

    if requested_device is not None and requested_device >= 0:
        if requested_device >= len(devices):
            raise MicrophoneError(
                f"Устройство #{requested_device} не существует (всего устройств: {len(devices)})."
            )
        if int(devices[requested_device].get("max_input_channels", 0)) <= 0:
            raise MicrophoneError(
                f"Устройство #{requested_device} ({devices[requested_device].get('name', 'неизвестно')}) "
                "не имеет входных каналов (микрофона)."
            )
        return requested_device

    inputs = [
        (index, info)
        for index, info in enumerate(devices)
        if int(info.get("max_input_channels", 0)) > 0
    ]
    if not inputs:
        raise MicrophoneError("В списке нет устройств с входными каналами (микрофонов).")

    try:
        default_device = sd.default.device[0]
    except (AttributeError, IndexError, TypeError):
        default_device = None
    if not isinstance(default_device, int) or not 0 <= default_device < len(devices):
        raise MicrophoneError(
            "Системное устройство ввода не определено. "
            "Укажите JARVIS_DEVICE_NAME или JARVIS_DEVICE."
        )
    if int(devices[default_device].get("max_input_channels", 0)) <= 0:
        raise MicrophoneError(
            f"Системное устройство ввода #{default_device} не имеет входных каналов. "
            "Укажите JARVIS_DEVICE_NAME или JARVIS_DEVICE."
        )
    return default_device


def validate_input_device(
    sd: Any,
    requested_device: int | None = None,
    target_sr: int = 16000,
    requested_name: str | None = None,
    requested_hostapi: str | int | None = None,
) -> tuple[int, int]:
    """Проверяет пригодность аудиоустройства для записи и подбирает частоту.

    Функция действует как валидатор: если указано устройство и оно не подходит,
    выдаётся понятная ошибка. Если устройство поддерживает target_sr, возвращает её.
    Иначе возвращает дефолтную частоту устройства для последующего ресемплинга.

    Returns:
        (device_index, actual_samplerate)
    """
    device_index = select_input_device(
        sd, requested_device, requested_name, requested_hostapi
    )
    info = sd.query_devices()[device_index]

    # Проверяем поддержку целевой частоты
    try:
        sd.check_input_settings(
            device=device_index, channels=1, samplerate=target_sr, dtype="int16"
        )
        return device_index, target_sr
    except Exception:
        # Целевая частота не поддерживается напрямую.
        # Используем дефолтную частоту устройства.
        native_sr = int(info.get("default_samplerate", 44100))
        logger.info(
            "Устройство #%d не поддерживает %d Hz, будет использован ресемплинг из %d Hz",
            device_index,
            target_sr,
            native_sr,
        )
        return device_index, native_sr


def pause_input() -> None:
    """Перестать раздавать аудио подписчикам (идёт озвучка)."""
    _input_paused.set()


def resume_input() -> None:
    """Снова раздавать аудио (озвучка закончилась)."""
    _input_paused.clear()


def input_paused() -> bool:
    return _input_paused.is_set()


class BroadcastBuffer:
    """Холдит скользящее окно блоков аудио, раздавая копии подписчикам."""

    def __init__(self, maxlen: int = 1000, mic: Microphone | None = None):
        self._buffer: collections.deque[bytes] = collections.deque(maxlen=maxlen)
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()
        self._mic = mic

    def check(self) -> None:
        if self._mic is not None:
            self._mic.check()

    def push(self, block: bytes) -> None:
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

    def subscribe(self, maxsize: int = 100) -> Subscriber:
        """Возвращает подписчика, который можно читать как старый mic.read()."""
        sub = Subscriber(self, maxsize)
        with self._lock:
            self._subscribers.append(sub)
        # Заполняем историей (чтобы не пропустить то, что уже записано)
        for block in self._buffer:
            sub._offer(block)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
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

    def _offer(self, block: bytes) -> None:
        if self._q.full():
            with contextlib.suppress(queue.Empty):
                self._q.get_nowait()  # отстал — выбрасываем старый блок
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(block)

    # ── публичное ──

    def read(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """Возвращает ровно n_samples сэмплов int16.

        timeout — сколько секунд ждать один блок; None = ждать вечно.
        При таймауте уже прочитанное возвращается в буфер.
        """
        parts: list[np.ndarray] = []
        got = 0

        if len(self._leftover):
            take = min(len(self._leftover), n_samples)
            parts.append(self._leftover[:take])
            got = take
            self._leftover = self._leftover[take:]

        try:
            while got < n_samples:
                raw = self._q.get(timeout=timeout)
                chunk = np.frombuffer(raw, dtype=np.int16)
                if len(chunk) > n_samples - got:
                    self._leftover = chunk[n_samples - got :]
                    chunk = chunk[: n_samples - got]
                parts.append(chunk)
                got += len(chunk)
        except queue.Empty:
            # Возвращаем недочитанное обратно, данные не теряются.
            if parts:
                rest = parts[-1]
                self._leftover = (
                    rest if not len(self._leftover) else np.concatenate([rest, self._leftover])
                )
            raise TimeoutError from None

        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def read_bytes(self, n_samples: int, timeout: float | None = None) -> bytes:
        """То же, что read(), но сразу в bytes для Vosk."""
        return self.read(n_samples, timeout).tobytes()

    def drain(self) -> None:
        """Выбрасывает накопленное аудио."""
        self._leftover = np.zeros(0, dtype=np.int16)
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def check(self) -> None:
        """Бросает MicrophoneError, если поток читателя умер."""
        self._broadcast.check()

    def close(self) -> None:
        """Отписывает от буфера."""
        self._broadcast.unsubscribe(self)


class MicrophoneError(RuntimeError):
    """Микрофон умер (устройство пропало, поток упал)."""


class Microphone:
    """Непрерывный поток с микрофона, раздающий блоки через BroadcastBuffer."""

    def __init__(
        self,
        device: int | None = DEVICE,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = AUDIO_BLOCK_SIZE,
        buffer_sec: float = AUDIO_BUFFER_SEC,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size

        # 1000 блоков * 1280 сэмплов = ~64 сек истории
        self._broadcast = BroadcastBuffer(maxlen=1000, mic=self)

        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    # ──────────────────── ЖИЗНЕННЫЙ ЦИКЛ ────────────────────

    def start(self) -> Microphone:
        """Открывает поток и запускает фоновый читатель."""
        if self._stream is not None:
            return self

        self._stop.clear()
        self._error = None

        # sounddevice загружается только при реальном открытии микрофона.
        # Это позволяет тестировать буфер и маршрутизацию в headless CI.
        import sounddevice as sd  # type: ignore[import-untyped]

        valid_device, actual_sr = validate_input_device(
            sd,
            self.device,
            self.sample_rate,
            requested_name=DEVICE_NAME,
            requested_hostapi=HOSTAPI,
        )
        self.device = valid_device
        self._actual_samplerate = actual_sr

        # Если частота отличается, создаем потоковый ресемплер и корректируем размер блока
        if actual_sr != self.sample_rate:
            self._resampler = StreamResampler(actual_sr, self.sample_rate)
            self._input_block_size = int(actual_sr * (self.block_size / self.sample_rate))
        else:
            self._resampler = None
            self._input_block_size = self.block_size

        device_info = sd.query_devices()[self.device]
        self._stream = sd.RawInputStream(
            samplerate=self._actual_samplerate,
            blocksize=self._input_block_size,
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
            "🎙️ Микрофон #%s открыт: %s, host API=%s, входов=%d, %d Hz -> %d Hz, "
            "ресемплинг=%s, блок %d",
            self.device,
            device_info.get("name", "неизвестно"),
            _hostapi_name(sd, device_info),
            device_info.get("max_input_channels", 0),
            self._actual_samplerate,
            self.sample_rate,
            self._actual_samplerate != self.sample_rate,
            self.block_size,
        )
        return self

    def stop(self) -> None:
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

    def __enter__(self) -> Microphone:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ──────────────────── ПОДПИСКА ────────────────────

    def subscribe(self, maxsize: int = 100) -> Subscriber:
        """Возвращает подписчика, который можно читать как старый mic.read()."""
        return self._broadcast.subscribe(maxsize)

    def unsubscribe(self, sub: Subscriber) -> None:
        """Отписывает подписчика от буфера."""
        self._broadcast.unsubscribe(sub)

    def drain(self) -> None:
        """Выбрасывает накопленное аудио у всех подписчиков и из истории.
        Вызывать при смене фазы цикла, чтобы VAD не начинал с речи,
        сказанной во время wakeword."""
        self._broadcast._buffer.clear()
        with self._broadcast._lock:
            subs = list(self._broadcast._subscribers)
        for sub in subs:
            sub.drain()

    def check(self) -> None:
        """Бросает MicrophoneError, если поток читателя умер."""
        if self._error is not None:
            raise MicrophoneError(self._error)

    # ──────────────────── ВНУТРЕННЕЕ ────────────────────

    def _reader_loop(self) -> None:
        """Фоновый читатель: микрофон -> BroadcastBuffer."""
        stream = self._stream
        input_block_size = self._input_block_size

        while not self._stop.is_set():
            try:
                data, overflowed = stream.read(input_block_size)
            except Exception as e:
                if self._stop.is_set():
                    # штатная остановка — не считаем это падением потока
                    break
                self._error = (
                    "аудиоустройство отключено или поток микрофона недоступен: "
                    f"{e}. Подключите устройство и перезапустите Jarvis."
                )
                logger.error("[mic] %s", self._error)
                return

            if overflowed:
                logger.debug("[mic] overflow — часть аудио пропущена")

            if self._resampler is not None:
                # Потоковый ресемплинг с антиалиасингом и предысторией
                audio_data = np.frombuffer(data, dtype=np.int16)
                resampled = self._resampler.resample(audio_data)
                raw = resampled.tobytes()
            else:
                raw = bytes(data)

            self._broadcast.push(raw)
