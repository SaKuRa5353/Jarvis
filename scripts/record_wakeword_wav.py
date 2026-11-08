#!/usr/bin/env python3
"""Локальная запись raw WAV-примеров «Джарвис» (требует микрофон, marker hw)."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402


def _next_path(directory: Path, prefix: str) -> Path:
    index = 1
    while True:
        candidate = directory / f"{prefix}_{index:03d}.wav"
        if not candidate.exists():
            return candidate
        index += 1


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    with path.open("xb") as raw, wave.open(raw, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(audio.astype(np.int16, copy=False).tobytes())


def _record(sd: Any, seconds: float, device: int) -> tuple[np.ndarray, bool]:
    """Записывает аудио и возвращает (аудио, был_прерван)."""
    sample_rate = config.SAMPLE_RATE
    total_frames = int(seconds * sample_rate)
    audio = np.zeros((total_frames, 1), dtype="int16")
    recorded_frames = 0
    chunk_frames = int(sample_rate * 0.1)
    last_print_time = -1.0
    interrupted = False

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            device=device,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames,
        ) as stream:
            while recorded_frames < total_frames:
                remaining = total_frames - recorded_frames
                to_read = min(chunk_frames, remaining)
                data, _ = stream.read(to_read)
                audio[recorded_frames : recorded_frames + len(data)] = data
                recorded_frames += len(data)
                now = recorded_frames / sample_rate
                if now - last_print_time >= 5.0 or recorded_frames >= total_frames:
                    percent = 100 * recorded_frames / total_frames
                    print(f"\rЗапись: {now:.1f}/{seconds:.1f} сек ({percent:.1f}%)", end="", flush=True)
                    last_print_time = now
        print()
    except (KeyboardInterrupt, RuntimeError):
        interrupted = True
        print(f"\nЗапись прервана. Получено {recorded_frames / sample_rate:.1f} сек.")

    return audio[:recorded_frames, 0], interrupted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("wakeword", "noise"),
        default="wakeword",
        help="wakeword: отдельные произнесения; noise: один файл фонового шума",
    )
    parser.add_argument("--out", type=Path, required=True, help="каталог для WAV")
    parser.add_argument("--count", type=int, default=None, help="число файлов в режиме wakeword, по умолчанию 5")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--prefix", help="префикс имени файла")
    parser.add_argument("--device", type=int, default=config.DEVICE)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds должен быть > 0")
    if args.mode == "noise" and args.count is not None:
        parser.error("--count несовместим с режимом noise: он записывает ровно один файл")
    count = 5 if args.count is None else args.count
    if count < 1:
        parser.error("--count должен быть >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    import sounddevice as sd  # type: ignore[import-untyped]

    prefix = args.prefix or ("noise" if args.mode == "noise" else "wakeword")
    expected_frames = int(args.seconds * config.SAMPLE_RATE)

    if args.mode == "noise":
        input("Нажми Enter, чтобы начать запись обычного фонового шума. Не произноси wakeword...")
        audio, interrupted = _record(sd, args.seconds, args.device)
        if len(audio) == 0:
            print("Запись отменена, данных нет.")
            return 0

        is_partial = interrupted or len(audio) < expected_frames
        stem = f"{prefix}{'_partial' if is_partial else ''}"
        path = _next_path(args.out, stem)
        # Для частичной записи меняем расширение на .partial, чтобы *.wav её не подобрал
        out_path = path.with_name(path.stem + ".partial") if is_partial else path
        _write_wav(out_path, audio, config.SAMPLE_RATE)
        print(f"{'Частичная запись сохранена' if is_partial else 'Сохранено'}: {out_path}")
        return 0

    for number in range(1, count + 1):
        input(f"Запись {number}/{count}: Enter, затем скажи «Джарвис»...")
        audio, interrupted = _record(sd, args.seconds, args.device)
        if len(audio) == 0:
            print("Запись отменена, данных нет.")
            break

        is_partial = interrupted or len(audio) < expected_frames
        stem = f"{prefix}{'_partial' if is_partial else ''}"
        path = _next_path(args.out, stem)
        out_path = path.with_name(path.stem + ".partial") if is_partial else path
        _write_wav(out_path, audio, config.SAMPLE_RATE)
        print(f"{'Частичная запись сохранена' if is_partial else 'Сохранено'}: {out_path}")
        if is_partial:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
