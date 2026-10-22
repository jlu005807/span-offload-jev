"""One native decision per eligible Span; existing M3 plans remain authoritative."""
from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from ...clock import real_time, real_utcnow
from ...contracts import ExecutionMode, FallbackPlan, RecoveryLevel
from ..routers import _ModeRouter, _guarded_slm_plan
from .input_builder import key_for
from .types import JevUnavailable, validate_response


class JevRouter(_ModeRouter):
    def __init__(self, large, small, *, config, context_store, client, audit,
                 guard=None, legacy=None) -> None:
        super().__init__(large, small)
        self.config, self.store, self.client, self.audit = config, context_store, client, audit
        self.guard, self.legacy = guard, legacy
        if config.mode == "shadow" and legacy is None:
            raise ValueError("Jev shadow requires a legacy router")

    def decide(self, profile, options=None, policy=None, snapshot=None, *, admission_facts=None):
        legacy = None
        if self.legacy is not None:
            legacy = self.legacy.decide(profile, options, policy, snapshot,
                                        admission_facts=admission_facts)
        timeout = legacy.timeout_ms if legacy is not None else int(policy.budget.max_latency_ms)
        p_small = self._predict(profile, timeout, legacy)
        if legacy is not None:
            return legacy
        if p_small is not None and p_small >= self.config.threshold:
            from ...contracts import GuardRef
            guard = self.guard or GuardRef(kind="guard", id="basic-tool-call", version="v1")
            return self._plan(profile, policy, snapshot, target=self._small_target(),
                              reason="jev_slm", execution_mode=ExecutionMode.GUARDED_SLM,
                              verifier_plan=_guarded_slm_plan(guard),
                              fallback_plan=FallbackPlan(target=self._large_target(),
                                                         recovery_mode=RecoveryLevel.NOT_NEEDED),
                              decision_meta_extra={"classifier_predictor": "jev_choice_v1"})
        return self._plan(profile, policy, snapshot, target=self._large_target(),
                          reason="jev_llm" if p_small is not None else "jev_unavailable_large",
                          decision_meta_extra={"classifier_predictor": "jev_choice_v1"})

    def _predict(self, profile, route_timeout_ms, legacy):
        started = real_time()
        prepared = self.store.take(key_for(profile))
        row = dict(zip(("workflow_run_id", "episode_id", "logical_invocation_id"), key_for(profile)))
        row.update(request_id=uuid4().hex, mode=self.config.mode, input_mode=self.config.input_mode,
                   threshold=self.config.threshold, model_id=self.config.model_id,
                   expected_revision=self.config.expected_revision, revision_verified=False,
                   request=prepared.request if prepared else None, response=None,
                   input_metadata=prepared.metadata if prepared else {},
                   http_attempted=False, status="input_unavailable", error="context_missing",
                   intent=None, legacy_intent=("SLM" if legacy.target.kind.value == "small_model"
                                               else "LLM") if legacy else None)
        score = None
        try:
            if prepared is None or prepared.error or prepared.request is None:
                raise JevUnavailable(prepared.error if prepared and prepared.error else "context_missing")
            if prepared.caller_deadline_at is None:
                raise JevUnavailable("deadline_missing")
            deadline = min(prepared.caller_deadline_at,
                           profile.produced_at + timedelta(milliseconds=route_timeout_ms))
            budget = min(self.config.request_timeout_ms,
                         int((deadline - real_utcnow()).total_seconds() * 1000)
                         - self.config.execution_reserve_ms)
            if budget <= 0:
                row["status"] = "budget_unavailable"
                raise JevUnavailable("budget_exhausted")
            response = self.client.predict(prepared.request, timeout_ms=budget)
            row["http_attempted"] = True
            response = validate_response(response, self.config.model_id)
            if real_utcnow() >= deadline - timedelta(milliseconds=self.config.execution_reserve_ms):
                raise JevUnavailable("late_result", attempted=True)
            score = response["answers"]["route"]["probabilities"]["SLM"]
            row.update(response=response, status="ok", error=None,
                       intent="SLM" if score >= self.config.threshold else "LLM")
        except JevUnavailable as error:
            row["http_attempted"] = row["http_attempted"] or getattr(error, "attempted", False)
            row["error"] = error.reason
            if row["http_attempted"]:
                row["status"] = "unavailable"
        except Exception:
            # Never expose transport exception text, which can contain authentication details.
            row.update(status="unavailable", error="client_error")
        finally:
            input_tokens = getattr(self.client, "last_input_tokens", None)
            if type(input_tokens) is int and input_tokens > 0:
                row.setdefault("input_metadata", {})["input_tokens"] = input_tokens
            row["client_elapsed_ms"] = max(0.0, (real_time() - started) * 1000)
            self.audit.record(row)
        return score
