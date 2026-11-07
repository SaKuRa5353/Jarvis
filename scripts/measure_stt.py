#!/usr/bin/env python3
"""Измерение качества и задержки STT по WAV и эталонному JSON."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"{path}: нужен моно WAV PCM int16")
        rate = stream.getframerate()
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).copy()
    return audio, rate


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            substitution = table[i - 1][j - 1]
            deletion = table[i - 1][j]
            insertion = table[i][j - 1]
            candidates = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            table[i][j] = min(candidates, key=lambda value: value[0])
    distance, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions, len(reference)


def _cer(reference: str, hypothesis: str) -> float:
    substitutions, deletions, insertions, total = _edit_counts(list(reference), list(hypothesis))
    return (substitutions + deletions + insertions) / total if total else float(hypothesis != reference)


def _recognize_vosk(audio: np.ndarray, sample_rate: int) -> str:
    try:
        from vosk import KaldiRecognizer

        from src.vosk_model import get_vosk_model, text_of
    except ModuleNotFoundError as exc:
        raise RuntimeError("backend vosk недоступен") from exc
    recognizer = KaldiRecognizer(get_vosk_model(), sample_rate)
    for start in range(0, len(audio), 4000):
        recognizer.AcceptWaveform(audio[start : start + 2000].tobytes())
    return text_of(recognizer.FinalResult())


def _recognize(backend: str, audio: np.ndarray, sample_rate: int, expected: str) -> str:
    if backend == "synthetic":
        return expected
    if backend == "vosk":
        return _recognize_vosk(audio, sample_rate)
    if backend == "sherpa":
        raise RuntimeError("backend sherpa требует отдельной конфигурации модели")
    if backend == "faster":
        raise RuntimeError("backend faster требует отдельной конфигурации модели")
    raise ValueError(f"неизвестный backend: {backend}")


def _write_wav(path: Path, audio: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(audio.astype(np.int16).tobytes())


def measure(audio_dir: Path, golden: dict[str, str], backends: list[str], verbose: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for backend in backends:
        for name, reference_text in golden.items():
            path = audio_dir / name
            try:
                audio, sample_rate = _read_wav(path)
                started = time.perf_counter()
                hypothesis = _recognize(backend, audio, sample_rate, reference_text)
                latency_ms = (time.perf_counter() - started) * 1000.0
                ref_words, hyp_words = reference_text.split(), hypothesis.split()
                substitutions, deletions, insertions, total = _edit_counts(ref_words, hyp_words)
                results.append({
                    "backend": backend,
                    "file": name,
                    "status": "ok",
                    "wer": (substitutions + deletions + insertions) / total if total else 0.0,
                    "cer": _cer(reference_text, hypothesis),
                    "S": substitutions,
                    "D": deletions,
                    "I": insertions,
                    "N": total,
                    "latency_ms": latency_ms,
                })
            except (OSError, RuntimeError, ValueError) as exc:
                results.append({"backend": backend, "file": name, "status": "unavailable", "error": str(exc)})
                if verbose:
                    print(f"[{backend}] {name}: {exc}")
    summary: list[dict[str, Any]] = []
    for backend in backends:
        rows = [row for row in results if row["backend"] == backend and row["status"] == "ok"]
        if not rows:
            summary.append({"backend": backend, "status": "unavailable"})
            continue
        total_words = sum(row["N"] for row in rows)
        summary.append({
            "backend": backend,
            "status": "ok",
            "WER": sum(row["S"] + row["D"] + row["I"] for row in rows) / total_words,
            "CER": float(np.mean([row["cer"] for row in rows])),
            "S": sum(row["S"] for row in rows),
            "D": sum(row["D"] for row in rows),
            "I": sum(row["I"] for row in rows),
            "N": total_words,
            "Q50_latency_ms": float(np.percentile([row["latency_ms"] for row in rows], 50)),
            "Q95_latency_ms": float(np.percentile([row["latency_ms"] for row in rows], 95)),
        })
    return {"kind": "stt", "summary": summary, "samples": results}


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        audio = root / "sample.wav"
        _write_wav(audio, np.zeros(1600, dtype=np.int16))
        return measure(root, {audio.name: "тестовая команда"}, ["synthetic"], False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--backends", default="vosk", help="vosk,sherpa,faster")
    parser.add_argument("--json", type=Path, default=Path("stt_results.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test()
    else:
        if args.audio is None or args.golden is None:
            parser.error("нужны --audio и --golden, либо --self-test")
        result = measure(args.audio, json.loads(args.golden.read_text(encoding="utf-8")),
                         [item.strip() for item in args.backends.split(",") if item.strip()], args.verbose)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
