from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.run_appworld import _prepare_runner_config
from traceoffload.contracts import AllowedTargets, ModelTarget, ModelTargetKind
from traceoffload.fakes import FakeAgentEnv, FakeEvaluator, FakeMiniSweModel, FakeModelGateway, FakeVerifier
from traceoffload.runner import PilotRunner
from test_jev_input import _context, _request, _msg, _profile
from test_jev_routing import Client


REPO = Path(__file__).resolve().parents[1]


def runner_config(*, enabled=True, input_mode="combined", mode="active"):
    raw = yaml.safe_load((REPO / "configs/arms/offload-active.yaml").read_text(encoding="utf-8"))
    raw["runner"]["l2_add4"]["model_path"] = str(REPO / "router/artifact/l2_add4_twostage_logit.json")
    raw["runner"]["history_store"]["backend"] = "memory"
    raw["runner"]["jev"] = dict(enabled=enabled, mode=mode, input_mode=input_mode,
                                endpoint="http://127.0.0.1:8900", model_id="jev-test")
    return _prepare_runner_config(raw)


def make_runner(tmp_path, cfg):
    return PilotRunner(config=cfg, mode="classifier_route", output_dir=tmp_path,
                       gateway=FakeModelGateway(), model=FakeMiniSweModel(),
                       env_factory=FakeAgentEnv, verifier=FakeVerifier(), evaluator=FakeEvaluator())


def targets(runner):
    return AllowedTargets(policy_version="test", targets=(
        ModelTarget(kind=ModelTargetKind.LARGE_MODEL, model_ref=runner._large),
        ModelTarget(kind=ModelTargetKind.SMALL_MODEL, model_ref=runner._small)))


def test_active_skips_legacy_artifact_even_when_enabled(tmp_path, monkeypatch):
    cfg = runner_config()
    cfg["runner"]["l2_add4"]["model_path"] = str(tmp_path / "does-not-exist.json")
    def forbidden(*args):
        pytest.fail("active loaded the legacy artifact")
    monkeypatch.setattr("traceoffload.runner.load_l2_add4_predictor", forbidden)
    runner = make_runner(tmp_path, cfg)
    assert runner._l2_add4_predictor is None
    assert not runner._m1_embedding_needed()
    assert runner._profile_observer_needed()
    assert runner._make_router().config.mode == "active"
    runner.close()


def test_jev_model_info_preserves_raw_configured_model_ids(tmp_path):
    cfg = runner_config()
    cfg["models"]["large"]["model_id"] = "provider/large-model"
    cfg["models"]["small"]["model_id"] = "provider/small-model"
    runner = make_runner(tmp_path, cfg)
    try:
        runner._set_skip_leading_messages(0)
        context, profile = _context(), _profile()
        prepared = runner._jev_input_builder.build(
            context, _request(_msg("user", "route this")), profile
        )
        assert prepared.request["state"]["model_info"]["large_model"] == "provider/large-model"
        assert prepared.request["state"]["model_info"]["small_model"] == "provider/small-model"
    finally:
        runner.close()


@pytest.mark.parametrize("input_mode,needs_features", [("context", False), ("four_features", True), ("combined", True)])
def test_active_profiler_only_builds_needed_numeric_features(tmp_path, input_mode, needs_features):
    runner = make_runner(tmp_path, runner_config(input_mode=input_mode))
    profiler = runner._make_profiler(targets(runner))
    assert hasattr(profiler, "_builder") is needs_features
    if needs_features:
        assert set(profiler._builder.feature_names) == {"prefix_n_A", "log_est_prompt_tokens"}
    runner.close()


def test_disabled_keeps_legacy_and_ignores_bad_unused_settings(tmp_path):
    cfg = runner_config(enabled=False)
    cfg["runner"]["jev"]["endpoint"] = "broken"
    runner = make_runner(tmp_path, cfg)
    assert runner._l2_add4_predictor is not None
    assert runner._jev_input_builder is None
    assert runner._jev_client is None
    assert runner._jev_audit is None
    runner.close()


def test_shadow_loads_legacy_and_combines_observers(tmp_path):
    runner = make_runner(tmp_path, runner_config(mode="shadow"))
    assert runner._l2_add4_predictor is not None
    assert runner._m1_embedding_needed()
    assert runner._make_router().legacy is not None
    runner.close()


def test_jev_observer_and_fewshot_registry_are_wired_without_embedding(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, runner_config())
    runner._set_skip_leading_messages(1)
    def forbidden(*args):
        pytest.fail("embedding called without a consumer")
    monkeypatch.setattr(runner, "_observe_m4_profile", forbidden)
    ctx, p = _context(), _profile()
    runner._observe_profile(ctx, _request(_msg("assistant", "DEMO"), _msg("user", "Real observation")), p)
    prepared = runner._jev_contexts.take((ctx.workflow_run_id, ctx.episode_id, ctx.logical_invocation_id))
    assert prepared.error is None
    assert "DEMO" not in str(prepared.request)
    runner.close()


def test_failed_task_finalization_keeps_stats_and_clears_inputs(tmp_path):
    from experiments.run_appworld import _jev_task_lifecycle
    runner = make_runner(tmp_path, runner_config())
    ctx = _context()
    from traceoffload.m2.jev.input_builder import PreparedInput
    key = (ctx.workflow_run_id, ctx.episode_id, ctx.logical_invocation_id)
    @_jev_task_lifecycle
    def fails(self, task_id):
        self._jev_task_workflow_id = ctx.workflow_run_id
        self._jev_contexts.put(key, PreparedInput(None, ctx.caller_deadline_at, "test", {}))
        self._jev_audit.record(dict(workflow_run_id=ctx.workflow_run_id, http_attempted=True,
                                   response=None, status="unavailable", intent=None,
                                   client_elapsed_ms=15, input_metadata={}))
        return {"instance_id": task_id, "task_success": False}
    row = fails(runner, "task-1")
    assert row["jev"]["http_attempts"] == 1
    assert row["jev"]["client_total_ms"] == 15
    assert runner._jev_contexts.take(key) is None
    runner.close()
