"""Read-only extraction of reproducible prefixes from a completed combined run."""
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace as NS

import yaml

from traceoffload.contracts import ModelRequest
from traceoffload.m2.jev.input_builder import JevInputBuilder, _serialize
from traceoffload.m2.jev.types import load_jev_config

MODES = ("context", "four_features", "combined")
IDENTITY = ("workflow_run_id", "episode_id", "logical_invocation_id")


def fingerprint(value):
    return sha256(_serialize(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row):
    key = tuple(row.get(k) for k in IDENTITY)
    if any(not isinstance(v, str) or not v for v in key):
        raise ValueError("Missing Span identity")
    return key


def select_prefixes(candidates, *, count, task_count, seed):
    """Balance tasks, then spread selected positions over each observed trajectory."""
    if type(count) is not int or type(task_count) is not int or count < 1 or task_count < 1:
        raise ValueError("Sample and task counts must be positive integers")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["task_id"]].append(row)
    tasks = sorted(grouped)
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:min(task_count, count)]
    allocation = Counter()
    while sum(allocation.values()) < count:
        changed = False
        for task in tasks:
            if allocation[task] < len(grouped[task]) and sum(allocation.values()) < count:
                allocation[task] += 1
                changed = True
        if not changed:
            raise ValueError("Not enough eligible Spans in the selected tasks")
    selected = []
    for task in tasks:
        rows = sorted(grouped[task], key=lambda r: r["sequence"])
        n = allocation[task]
        selected.extend(rows[min(len(rows)-1, int((i + .5) * len(rows) / n))] for i in range(n))
    return selected


def export_dataset(source, output, *, count=100, task_count=20, seed=42):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config_path = source / "config.yaml"
    raw_config = yaml.safe_load(config_path.read_bytes())
    config = load_jev_config(raw_config, mode="classifier_route")
    if config is None or config.input_mode != "combined":
        raise ValueError("A completed combined run is required")
    summary_path = source / "appworld-classifier_route-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = {r["workflow_run_id"]: r["instance_id"] for r in summary}
    if len(tasks) != len(summary):
        raise ValueError("Duplicate workflow in summary")
    audit_files = sorted(source.glob("routing-decisions-*.jsonl"))
    audits, excluded, seen = {}, Counter(), set()
    for path in audit_files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = identity(row)
                if key in seen:
                    raise ValueError("Duplicate audit Span identity")
                seen.add(key)
                body = row.get("request")
                if key[0] not in tasks:
                    excluded["unfinished_or_unknown_workflow"] += 1
                elif not body or not {"context", "features", "model_info"} <= set(body.get("state", {})):
                    excluded["no_complete_combined_input"] += 1
                else:
                    if fingerprint(body) != row.get("input_metadata", {}).get("input_sha256"):
                        raise ValueError("Original request hash mismatch")
                    audits[key] = row
    dbs = list(source.rglob("history.sqlite3"))
    if len(dbs) != 1 or not audit_files:
        raise ValueError("Exactly one history database and at least one audit file required")
    database = dbs[0]
    paths = [config_path, summary_path, database, *audit_files]
    hashes = {str(p.relative_to(source)): file_hash(p) for p in paths}
    anchors, candidates, seen_history = {}, [], set()
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        for sequence, text in con.execute("select sequence,payload_json from span_records order by sequence"):
            row = json.loads(text)
            profile = row.get("runtime_profile") or {}
            key = identity(profile)
            if key in seen_history:
                raise ValueError("Duplicate history Span identity")
            seen_history.add(key)
            if key not in audits:
                continue
            audit = audits[key]
            state, meta = audit["request"]["state"], audit["input_metadata"]
            if key[1:] == ("episode-1", "invocation-1"):
                v2 = state["features"].get("V2_goal_recent", "")
                if (state["context"].get("history") == [] and
                        meta.get("feature_v2_message_indices") == [] and
                        meta.get("feature_actions_message_indices") == [] and
                        v2.startswith("[GOAL] ") and v2.endswith("\n\n")):
                    anchors[key[0]] = (len(row["model_request"]["messages"]), v2[7:-2])
            candidates.append(dict(zip(IDENTITY, key), task_id=tasks[key[0]], sequence=sequence))
        excluded["audit_without_history"] = len(set(audits) - seen_history)
        excluded["no_verified_first_call_anchor"] = sum(r["workflow_run_id"] not in anchors for r in candidates)
        candidates = [r for r in candidates if r["workflow_run_id"] in anchors]
        selected = select_prefixes(candidates, count=count, task_count=task_count, seed=seed)
        samples = []
        for selected_row in selected:
            sequence = selected_row["sequence"]
            row = json.loads(con.execute("select payload_json from span_records where sequence=?", (sequence,)).fetchone()[0])
            key = identity(row["runtime_profile"])
            request = ModelRequest.model_validate_json(_serialize(row["model_request"]))
            skip, goal = anchors[key[0]]
            profile = NS(**row["runtime_profile"])
            context = NS(**dict(zip(IDENTITY, key)), prior_execution_facts=None,
                         workflow_context=NS(workflow_goal=NS(goal_text=goal)))
            original = audits[key]["request"]
            inputs = {}
            for mode in MODES:
                builder = JevInputBuilder(replace(config, input_mode=mode),
                    large_model=original["state"]["model_info"]["large_model"],
                    small_model=original["state"]["model_info"]["small_model"])
                builder.set_skip_leading_messages(skip)
                prepared = builder.build(context, request, profile)
                inputs[mode] = {"request": prepared.request, "error": prepared.error,
                                "metadata": prepared.metadata}
            if inputs["combined"]["request"] != original or fingerprint(inputs["combined"]["request"]) != fingerprint(original):
                raise ValueError(f"Combined reproduction failed for history sequence {sequence}")
            samples.append({**selected_row, "sample_id": fingerprint(list(key)),
                "goal_text": goal, "skip_leading_messages": skip,
                "model_request": row["model_request"],
                "online_features": profile.online_features,
                "source_request_sha256": fingerprint(original), "inputs": inputs})
    if any(file_hash(path) != hashes[str(path.relative_to(source))] for path in paths):
        raise ValueError("Source changed during extraction")
    output.mkdir(parents=True)
    sample_path = output / "samples.jsonl"
    sample_path.write_text("".join(_serialize(s) + "\n" for s in samples), encoding="utf-8")
    manifest = {"schema_version": 1, "source": str(source.resolve()), "source_hashes": hashes,
        "source_config": raw_config["runner"]["jev"], "seed": seed,
        "eligible_spans": len(candidates), "source_decisions": len(seen),
        "excluded": dict(excluded), "sample_count": len(samples),
        "tasks": dict(Counter(s["task_id"] for s in samples)),
        "samples_sha256": file_hash(sample_path), "combined_reproduction_verified": True,
        "scope": "Reconstructible combined inputs; not a correctness-labelled dataset"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "samples.jsonl"
    if file_hash(path) != manifest["samples_sha256"]:
        raise ValueError("Dataset hash mismatch")
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) != manifest["sample_count"] or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Invalid sample count or duplicate sample")
    for sample in samples:
        if set(sample["inputs"]) != set(MODES):
            raise ValueError("Missing input mode")
        for prepared in sample["inputs"].values():
            if prepared["request"] is not None and fingerprint(prepared["request"]) != prepared["metadata"]["input_sha256"]:
                raise ValueError("Input hash mismatch")
    return samples


def check_acceptance(dataset, path):
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("ready_for_experiment") is not True:
        raise ValueError("Offline acceptance has not passed")
    for name, key in [("samples.jsonl", "samples_sha256"), ("manifest.json", "dataset_manifest_sha256")]:
        if gate.get(key) != file_hash(Path(dataset) / name):
            raise ValueError("Acceptance dataset hash mismatch")
    project = Path(__file__).resolve().parents[1]
    hashes = gate.get("implementation_hashes", {})
    for name in ["experiments/jev_fixed_span.py", "experiments/jev_fixed_span_data.py"]:
        if hashes.get(name) != file_hash(project / name):
            raise ValueError("Acceptance implementation hash mismatch")
    return gate
