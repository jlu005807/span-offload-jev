"""Local decision journal; audit I/O never interrupts a tool transaction."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


class JevAudit:
    def __init__(self, path: Path, *, config) -> None:
        self.path = Path(path)
        self.config = config
        self._stats: dict[str, dict] = {}

    def record(self, row: dict) -> None:
        workflow = row["workflow_run_id"]
        stats = self._stats.setdefault(workflow, self._empty())
        stats["decisions"] += 1
        stats["http_attempts"] += int(row["http_attempted"])
        stats["http_successes"] += int(row["response"] is not None)
        stats["http_errors"] += int(row["http_attempted"] and row["status"] != "ok")
        stats["input_unavailable"] += int(row["status"] == "input_unavailable")
        stats["budget_unavailable"] += int(row["status"] == "budget_unavailable")
        stats["client_total_ms"] += row["client_elapsed_ms"]
        stats["state_chars_total"] += row.get("input_metadata", {}).get("state_chars") or 0
        if row["intent"]:
            stats[f"{row['intent'].lower()}_intent"] += 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        except (OSError, TypeError, ValueError):
            stats["audit_errors"] += 1
            stats["routing_audit_complete"] = False

    def _empty(self) -> dict:
        return dict(mode=self.config.mode, input_mode=self.config.input_mode,
                    model_id=self.config.model_id, threshold=self.config.threshold,
                    expected_revision=self.config.expected_revision, revision_verified=False,
                    decisions=0, http_attempts=0, http_successes=0, http_errors=0,
                    input_unavailable=0, budget_unavailable=0, slm_intent=0, llm_intent=0,
                    client_total_ms=0.0, state_chars_total=0, audit_errors=0,
                    routing_audit_complete=True)

    def summary(self, workflow: str) -> dict:
        return deepcopy(self._stats.get(workflow, self._empty()))
