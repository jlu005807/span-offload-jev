"""Paired native Jev inference on fixed prefixes; never execute Agent actions."""
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter

from traceoffload.m2.jev.types import JevUnavailable, load_jev_config, validate_response
from experiments.jev_fixed_span_data import MODES, check_acceptance, export_dataset, file_hash, fingerprint, load_dataset

THRESHOLDS = (.5, .55, .6, .65, .7, .75, .8)


def _stats(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    scores = [r["p_slm"] for r in ok]
    return {"samples": len(rows), "successes": len(ok),
        "errors": dict(Counter(r["error"] for r in rows if r["status"] != "ok")),
        "mean_slm_probability": mean(scores) if scores else None,
        "median_slm_probability": median(scores) if scores else None,
        "max_slm_probability": max(scores) if scores else None,
        "input_tokens_known": sum(r["input_tokens"] for r in ok if r.get("input_tokens") is not None),
        "input_tokens_known_responses": sum(r.get("input_tokens") is not None for r in ok),
        "client_total_ms": sum(r["client_elapsed_ms"] for r in rows),
        "thresholds": {str(t): {"candidates": sum(p >= t for p in scores),
            "all_sample_rate": sum(p >= t for p in scores)/len(rows) if rows else None,
            "successful_response_rate": sum(p >= t for p in scores)/len(ok) if ok else None}
            for t in THRESHOLDS}}


def summarize(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row["mode"] not in MODES or row["mode"] in grouped[row["sample_id"]]:
            raise ValueError("Duplicate or invalid mode")
        grouped[row["sample_id"]][row["mode"]] = row
    if not grouped or any(set(r) != set(MODES) for r in grouped.values()):
        raise ValueError("Missing mode results")
    paired = [r for r in grouped.values() if all(v["status"] == "ok" for v in r.values())]
    differences = {}
    for left, right in [("context", "four_features"), ("context", "combined"), ("combined", "four_features")]:
        values = [r[left]["p_slm"] - r[right]["p_slm"] for r in paired]
        differences[left + "_minus_" + right] = {"mean": mean(values) if values else None,
            "left_higher": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values)}
    return {"samples": len(grouped), "decisions": len(rows), "paired_samples": len(paired),
        "modes": {mode: _stats([r for r in rows if r["mode"] == mode]) for mode in MODES},
        "paired": {mode: _stats([r[mode] for r in paired]) for mode in MODES},
        "paired_differences": differences,
        "local_correctness": None, "net_saving": None,
        "limitation": "No SLM branch labels; probabilities and threshold candidates do not measure correctness or realised savings."}


def run_experiment(dataset, output, client, *, seed=42, timeout_ms=2000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    samples = load_dataset(dataset)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({"dataset": str(Path(dataset).resolve()),
        "dataset_manifest_sha256": file_hash(Path(dataset)/"manifest.json"),
        "samples_sha256": file_hash(Path(dataset)/"samples.jsonl"),
        "sample_count": len(samples), "seed": seed, "timeout_ms": timeout_ms}, indent=2), encoding="utf-8")
    rng = random.Random(seed)
    rows = []
    with (output / "decisions.jsonl").open("x", encoding="utf-8") as stream:
        for sample in samples:
            for mode in rng.sample(list(MODES), len(MODES)):
                prepared = sample["inputs"][mode]
                body = prepared["request"]
                row = {"sample_id": sample["sample_id"], "task_id": sample["task_id"],
                    "mode": mode, "status": "unavailable", "error": prepared["error"],
                    "request_sha256": fingerprint(body) if body is not None else None,
                    "http_attempted": False, "input_tokens": None, "client_elapsed_ms": 0.0}
                started = perf_counter()
                try:
                    if body is None:
                        raise JevUnavailable(prepared["error"] or "input_missing")
                    response = client.predict(body, timeout_ms=timeout_ms)
                    row["http_attempted"] = True
                    response = validate_response(response, body["model"])
                    row.update(status="ok", error=None, http_attempted=True,
                        p_slm=response["answers"]["route"]["probabilities"]["SLM"],
                        response=response, input_tokens=response.get("usage", {}).get("input_tokens"))
                except JevUnavailable as error:
                    row.update(error=error.reason, http_attempted=row["http_attempted"] or error.attempted)
                row["client_elapsed_ms"] = max(0, (perf_counter()-started)*1000)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    report = summarize(rows)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    """Environment-only entry point, matching the existing deployment scripts."""
    action = os.environ["JEV_FIXED_ACTION"]
    dataset = Path(os.environ["JEV_FIXED_DATASET"])
    if action == "extract":
        manifest = export_dataset(os.environ["JEV_FIXED_SOURCE"], dataset,
            count=int(os.environ.get("JEV_FIXED_COUNT", "100")),
            task_count=int(os.environ.get("JEV_FIXED_TASKS", "20")), seed=int(os.environ.get("JEV_FIXED_SEED", "42")))
        print(json.dumps({k: manifest[k] for k in ["sample_count", "tasks", "eligible_spans", "excluded"]}))
    elif action == "run":
        check_acceptance(dataset, os.environ["JEV_FIXED_ACCEPTANCE"])
        from deploy.jev.openjev_preflight import OpenJevPreflight
        from traceoffload.m2.jev.client import HttpJevClient
        import yaml
        config_bytes = Path(os.environ["JEV_RUN_CONFIG"]).read_bytes()
        config = load_jev_config(yaml.safe_load(config_bytes), mode="classifier_route")
        if config is None:
            raise ValueError("Jev must be enabled")
        samples = load_dataset(dataset)
        if any(s["inputs"]["combined"]["request"]["model"] != config.model_id for s in samples):
            raise ValueError("Dataset model differs from service configuration")
        preflight = OpenJevPreflight.from_directory(os.environ["JEV_MODEL_DIR"])
        client = HttpJevClient(config, input_preflight=preflight)
        output = Path(os.environ["JEV_RUN_OUTPUT"])
        report = run_experiment(dataset, output, client,
            seed=int(os.environ.get("JEV_FIXED_SEED", "42")), timeout_ms=config.request_timeout_ms)
        (output / "service-config.json").write_text(json.dumps({**asdict(config),
            "config_sha256": sha256(config_bytes).hexdigest(),
            "revision_verified": False}, indent=2), encoding="utf-8")
        print(json.dumps({"samples": report["samples"], "paired_samples": report["paired_samples"]}))
    else:
        raise ValueError("JEV_FIXED_ACTION must be extract or run")


if __name__ == "__main__":
    main()
