"""Sequential, durable AppWorld batch using the validated OpenJev deployment."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx
import yaml
from deploy.jev.openjev_preflight import OpenJevPreflight
from experiments.run_appworld import AppWorldPilotRunner


def load_manifest(path):
    ids = [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Task manifest must be nonempty and contain no duplicates")
    return ids


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_services(config, *, attempts=6, delay_seconds=5):
    origins = [(config["models"][kind]["api_base"].rstrip("/") + "/models",
                config["models"][kind].get("served_model_name") or config["models"][kind]["model_id"])
               for kind in ("large", "small")]
    origins.append((config["runner"]["jev"]["endpoint"].rstrip("/") + "/v1/models",
                    config["runner"]["jev"]["model_id"]))
    last_error = None
    with httpx.Client(trust_env=False, timeout=10) as client:
        for attempt in range(attempts):
            try:
                for url, expected in origins:
                    response = client.get(url)
                    response.raise_for_status()
                    if expected not in {item.get("id") for item in response.json().get("data", [])}:
                        raise RuntimeError("Expected model is not available from the service")
                return
            except Exception as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(delay_seconds)
    raise RuntimeError(
        f"Model service health check failed for {expected} at {url} after {attempts} attempts"
    ) from last_error


def main():
    gate = json.loads(Path(os.environ["JEV_ACCEPTANCE_FILE"]).read_text(encoding="utf-8"))
    if gate.get("ready_for_batch") is not True:
        raise SystemExit("Single-task acceptance has not passed; batch was not started")
    config_path = Path(os.environ["JEV_RUN_CONFIG"])
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if gate.get("config_sha256") != config_sha256:
        raise SystemExit("Batch configuration differs from the accepted single-task configuration")
    raw = yaml.safe_load(config_bytes)
    if raw.get("runner", {}).get("jev", {}).get("enabled") is not True:
        raise SystemExit("This deployment batch requires Jev enabled")
    ids = load_manifest(os.environ["JEV_TASK_MANIFEST"])
    output = Path(os.environ["JEV_RUN_OUTPUT"])
    if output.exists():
        raise SystemExit("Use a fresh batch output directory")
    check_services(raw)
    output.mkdir(parents=True)
    (output / "config.yaml").write_bytes(config_bytes)
    (output / "instances.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    experiment = f"jev-batch-{uuid4().hex}"
    write_json(output / "manifest.json", {
        "task_ids": ids, "experiment_name": experiment,
        "appworld_output": str(Path(os.environ["APPWORLD_ROOT"]) / "experiments/outputs" / experiment),
        "config_sha256": config_sha256,
        "acceptance_file": os.environ["JEV_ACCEPTANCE_FILE"],
    })
    preflight = OpenJevPreflight.from_directory(os.environ["JEV_MODEL_DIR"])
    runner = AppWorldPilotRunner(raw_config=raw, mode="classifier_route", agent_type="react_code",
                                task_ids=ids, output_dir=output, experiment_name=experiment,
                                jev_input_preflight=preflight)
    runner._appworld_agent_config["logger_config"]["verbose"] = True
    rows = []
    try:
        for task_id in ids:
            check_services(raw)
            row = runner.run_task(task_id)
            committed_rows = [*rows, row]
            write_json(output / "appworld-classifier_route-summary.json", committed_rows)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"completed": len(committed_rows), "total": len(ids),
                                         "task_id": task_id, "task_success": row.get("task_success"),
                                         "failure": row.get("failure")}, ensure_ascii=False) + "\n")
                stream.flush()
            rows.append(row)
            print(f"TASK_DONE {len(rows)}/{len(ids)} {task_id} success={row.get('task_success')}", flush=True)
    except Exception as error:
        write_json(output / "batch-error.json", {"type": type(error).__name__, "completed": len(rows),
                                                  "remaining": ids[len(rows):]})
        raise
    finally:
        runner.close()
    from experiments.report import summarize
    write_json(output / "report.json", summarize(rows))
    print(f"BATCH_COMPLETE {len(rows)}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
