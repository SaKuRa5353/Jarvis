#!/usr/bin/env python3
"""Подбор порогов fuzzy и intent по размеченным JSON/JSONL данным."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mathcore.thresholds import cost_threshold, roc_curve, youden_threshold  # noqa: E402


def _records_from(path: Path, source: str, limit: int | None) -> list[dict[str, Any]]:
    if source == "history":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else list(raw.values())
    return [record for record in records if isinstance(record, dict)][:limit]


def _series(records: list[dict[str, Any]], score_key: str, label_key: str) -> tuple[list[int], list[float]]:
    pairs = [(int(record[label_key]), float(record[score_key])) for record in records
             if score_key in record and label_key in record]
    if not pairs or {label for label, _ in pairs} != {0, 1}:
        raise ValueError(f"нужны поля {score_key!r}/{label_key!r} и оба класса")
    return [label for label, _ in pairs], [score for _, score in pairs]


def _fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    fuzzy_labels, fuzzy_scores = _series(records, "fuzzy_score", "fuzzy_label")
    intent_labels, intent_scores = _series(records, "intent_prob", "intent_label")
    _, margins = _series(records, "intent_margin", "intent_label")
    fuzzy_threshold = youden_threshold(fuzzy_labels, fuzzy_scores)
    intent_tau = cost_threshold(intent_labels, intent_scores, 10.0, 1.0)
    intent_delta = cost_threshold(intent_labels, margins, 10.0, 1.0)
    return {
        "kind": "thresholds",
        "FUZZY_THRESHOLD": fuzzy_threshold,
        "INTENT_TAU": intent_tau,
        "INTENT_DELTA": intent_delta,
        "costs": {"C_FP": 10.0, "C_FN": 1.0},
        "config_block": (f"FUZZY_THRESHOLD = {fuzzy_threshold:.6g}\n"
                          f"INTENT_TAU = {intent_tau:.6g}\n"
                          f"INTENT_DELTA = {intent_delta:.6g}"),
    }


def _save_roc(labels: list[int], scores: list[float], path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        path.with_suffix(".csv").write_text(
            "fpr,tpr\n" + "\n".join(f"{x},{y}" for x, y in zip(fpr, tpr, strict=False)) + "\n",
            encoding="utf-8",
        )
        return
    figure, axis = plt.subplots()
    axis.plot(fpr, tpr)
    axis.set(xlabel="FPR", ylabel="TPR", title=path.stem)
    axis.grid(True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _synthetic_records() -> list[dict[str, Any]]:
    records = []
    for index in range(40):
        positive = index % 2 == 0
        records.append({
            "fuzzy_score": 0.85 if positive else 0.25,
            "fuzzy_label": int(positive),
            "intent_prob": 0.9 if positive else 0.2,
            "intent_margin": 0.7 if positive else 0.05,
            "intent_label": int(positive),
        })
    return records


def _self_test() -> dict[str, Any]:
    return _fit(_synthetic_records())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("history", "golden"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path, default=Path("thresholds_results.json"))
    parser.add_argument("--plots", type=Path, default=Path("plots"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        records = _synthetic_records()
        result = _fit(records)
    else:
        if args.source is None:
            parser.error("нужен --source history|golden, либо --self-test")
        path = Path("data/history") if args.source == "history" else Path("tests/golden.json")
        if args.source == "history":
            files = sorted(path.glob("*.jsonl"))
            if not files:
                raise SystemExit("в data/history нет JSONL-файлов")
            records = _records_from(files[0], args.source, args.limit)
        else:
            records = _records_from(path, args.source, args.limit)
        result = _fit(records)
    _save_roc(*_series(records, "fuzzy_score", "fuzzy_label")[0:2], args.plots / "roc_fuzzy.png")
    _save_roc(*_series(records, "intent_prob", "intent_label")[0:2], args.plots / "roc_intent.png")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["config_block"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
