#!/usr/bin/env python3
"""Evaluate OpenJev choice probabilities as an SLM-offload classifier."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _validate_inputs(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[list[int], list[float]]:
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    if any(label not in (0, 1) or isinstance(label, bool) for label in labels):
        raise ValueError("labels must contain only integer 0/1 values (SLM is positive)")
    if any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return labels, [float(p) for p in probabilities]


def roc_auc(labels: Iterable[int], probabilities: Iterable[float]) -> float | None:
    """ROC-AUC with average ranks for tied scores; return None if only one class exists."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    order = sorted(range(len(labels)), key=probabilities.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        rank_sum += sum(labels[order[i]] for i in range(start, end)) * average_rank
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: Iterable[int], probabilities: Iterable[float]) -> float | None:
    """Non-interpolated average precision, grouping tied probability thresholds."""
    labels, probabilities = _validate_inputs(labels, probabilities)
    positives = sum(labels)
    if not positives:
        return None

    order = sorted(range(len(labels)), key=probabilities.__getitem__, reverse=True)
    true_positive = false_positive = 0
    previous_recall = 0.0
    score = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        group = [labels[order[i]] for i in range(start, end)]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        score += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return score


def threshold_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict:
    labels, probabilities = _validate_inputs(labels, probabilities)
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value in [0, 1]")
    predictions = [int(p >= threshold) for p in probabilities]
    tp = sum(y == 1 and pred == 1 for y, pred in zip(labels, predictions))
    fp = sum(y == 0 and pred == 1 for y, pred in zip(labels, predictions))
    fn = sum(y == 1 and pred == 0 for y, pred in zip(labels, predictions))
    tn = sum(y == 0 and pred == 0 for y, pred in zip(labels, predictions))
    predicted_slm = sum(predictions)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "predicted_slm": predicted_slm,
        "predicted_llm": len(labels) - predicted_slm,
        "gold_slm": sum(labels),
        "gold_llm": len(labels) - sum(labels),
        "sample_count": len(labels),
    }


def _parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds:
        raise argparse.ArgumentTypeError("provide at least one threshold")
    try:
        for threshold in thresholds:
            threshold_metrics([0, 1], [0.25, 0.75], threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("thresholds must not contain duplicates")
    return thresholds


def evaluate_split(*, run: str, data_dir: Path, out_dir: Path, thresholds: list[float],
                   split: str = "dev", dtype: str = "bf16", device: str | None = None,
                   limit: int = 0, max_tokens: int = 6144) -> dict:
    """Score a labelled dev/test split; save probabilities and gold routes without training."""
    if split not in {"dev", "test"}:
        raise ValueError("evaluation split must be dev or test")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {out_dir}")

    import torch
    from wev.data import load_items, split_file
    from wev.inference import load

    model = load(run, device=device, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
    source = split_file(data_dir, split)
    items, dropped = load_items(model.tokenizer, source, model.max_state, model.max_branch,
                                max_tokens=max_tokens, limit=limit or None)
    if not items:
        raise ValueError(f"OpenJev loader accepted no rows from {source}")

    predictions = []
    labels = []
    probabilities = []
    for item in items:
        meta = item["meta"]
        gold = item["enc"]["labels"]
        if len(meta) != 1 or len(gold) != 1 or meta[0].get("type") != "choice":
            raise ValueError("expected exactly one labelled choice question per input row")
        keys = meta[0].get("keys") or []
        if keys.count("SLM") != 1 or keys.count("LLM") != 1:
            raise ValueError("route choice options must contain exactly LLM and SLM")
        slm_index = keys.index("SLM")
        gold_index = gold[0]
        if not isinstance(gold_index, int) or not 0 <= gold_index < len(keys):
            raise ValueError("loader returned an invalid choice label index")
        route_probabilities = model.model.probs([item["enc"]])[0][0]
        if len(route_probabilities) != len(keys):
            raise ValueError("OpenJev probability count does not match route options")
        p_slm = float(route_probabilities[slm_index])
        y_slm = int(gold_index == slm_index)
        labels.append(y_slm)
        probabilities.append(p_slm)
        predictions.append({
            **item.get("_meta", {}),
            "gold_route": keys[gold_index],
            "p_slm": p_slm,
            "p_llm": float(route_probabilities[keys.index("LLM")]),
        })

    threshold_rows = [threshold_metrics(labels, probabilities, value) for value in thresholds]
    summary = {
        "run": run,
        "split": split,
        "data_file": str(source),
        "positive_class": "SLM",
        "sample_count": len(labels),
        "dropped_by_openjev_loader": dropped,
        "max_tokens": max_tokens,
        "label_distribution": {"LLM": labels.count(0), "SLM": labels.count(1)},
        "roc_auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "threshold_rule": "predict SLM (offload) when p_slm >= threshold; otherwise LLM",
        "threshold_metrics": threshold_rows,
    }

    out_dir.mkdir(parents=True)
    with (out_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    with (out_dir / "threshold_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    return summary


def evaluate_dev(**kwargs) -> dict:
    """Keep the existing Python entry point dev-only for historical callers."""
    return evaluate_split(**kwargs, split="dev")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="OpenJev training run or exported model directory")
    parser.add_argument("--data", type=Path, required=True, help="directory containing the selected split JSONL")
    parser.add_argument("--split", choices=("dev", "test"), default="dev",
                        help="labelled evaluation split; test never updates model weights")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; existing paths are refused")
    parser.add_argument("--thresholds", type=_parse_thresholds,
                        default=_parse_thresholds("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
                        help="comma-separated SLM probability cutoffs, e.g. 0.3,0.5,0.7")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None, help="defaults to OpenJev's device selection")
    parser.add_argument("--limit", type=int, default=0, help="optional subset for smoke evaluation")
    parser.add_argument("--max-tokens", type=int, default=6144,
                        help="match the training loader's whole-record token limit")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_tokens < 1:
        parser.error("--limit must be >= 0 and --max-tokens must be > 0")
    summary = evaluate_split(run=args.run, data_dir=args.data, out_dir=args.out,
                             thresholds=args.thresholds, split=args.split, dtype=args.dtype,
                             device=args.device, limit=args.limit, max_tokens=args.max_tokens)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
