#!/usr/bin/env python3
"""Сборка markdown-отчёта из JSON-результатов измерений."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mathcore.thresholds import roc_curve, zipf_hit_rate  # noqa: E402


def _load(results_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            documents.append({"file": path.name, **value})
    return documents


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output) + "\n"


def _save_plot(path: Path, x: list[float], y: list[float], xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except ModuleNotFoundError:
        path.with_suffix(".csv").write_text(
            f"{xlabel},{ylabel}\n" + "\n".join(f"{a},{b}" for a, b in zip(x, y, strict=False)) + "\n",
            encoding="utf-8",
        )
        return
    figure, axis = plt.subplots()
    axis.plot(x, y)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=path.stem)
    axis.grid(True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def build(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    documents = _load(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stt_rows = []
    threshold_rows = []
    latency_rows = []
    metrics_rows = []
    for document in documents:
        kind = document.get("kind")
        if kind == "stt":
            for row in document.get("summary", []):
                stt_rows.append([row.get("backend"), row.get("WER", "n/a"), row.get("CER", "n/a"), row.get("N", "n/a")])
                latency_rows.append([row.get("backend"), row.get("Q50_latency_ms", "n/a"), row.get("Q95_latency_ms", "n/a")])
        elif kind == "thresholds":
            threshold_rows.append([document.get("FUZZY_THRESHOLD"), document.get("INTENT_TAU"), document.get("INTENT_DELTA")])
        elif kind == "wakeword":
            metrics_rows.append(["wakeword ROC-AUC", document.get("ROC_AUC", "n/a")])
            metrics_rows.append(["wakeword FAH", document.get("FAH", "n/a")])
    (output_dir / "wer.md").write_text("# WER/CER\n\n" + _table(["Backend", "WER", "CER", "N"], stt_rows), encoding="utf-8")
    (output_dir / "latency.md").write_text("# Latency\n\n" + _table(["Backend", "Q50 ms", "Q95 ms"], latency_rows), encoding="utf-8")
    (output_dir / "thresholds.md").write_text("# Thresholds\n\n" + _table(["FUZZY", "INTENT_TAU", "INTENT_DELTA"], threshold_rows), encoding="utf-8")
    (output_dir / "metrics_summary.md").write_text("# Metrics summary\n\n" + _table(["Metric", "Value"], metrics_rows), encoding="utf-8")
    cache_x = [float(index) for index in range(1, 101)]
    cache_y = [zipf_hit_rate(index, 100) for index in range(1, 101)]
    _save_plot(output_dir / "cache_zipf.png", cache_x, cache_y, "n_cache", "hit_rate")
    for document in documents:
        if document.get("kind") != "wakeword":
            continue
        positive = document.get("positive_scores", [])
        negative = document.get("negative_scores", [])
        if positive and negative:
            fpr, tpr, _ = roc_curve([1] * len(positive) + [0] * len(negative), positive + negative)
            _save_plot(output_dir / "roc_wakeword.png", list(fpr), list(tpr), "FPR", "TPR")
    return {"kind": "report", "files": sorted(path.name for path in output_dir.iterdir())}


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        results = root / "results"
        results.mkdir()
        (results / "stt.json").write_text(json.dumps({"kind": "stt", "summary": [{"backend": "synthetic", "WER": 0.0, "CER": 0.0, "N": 2, "Q50_latency_ms": 1.0, "Q95_latency_ms": 2.0}]}), encoding="utf-8")
        return build(results, root / "out")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--json", type=Path, default=Path("report_results.json"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test()
    else:
        if args.results is None:
            parser.error("нужен --results, либо --self-test")
        result = build(args.results, args.out)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
