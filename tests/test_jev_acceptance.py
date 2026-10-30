from copy import deepcopy

import pytest

from deploy.jev.run_single_task import acceptance_errors, new_experiment_name


def valid():
    return {"workflows": [{"instance_id": "099174d_1", "task_success": True, "failure": None,
                           "jev": {"http_successes": 2, "routing_audit_complete": True},
                           "framework_model_calls_by_target": {"large_model": 2, "small_model": 1}}]}


def test_healthy_native_task_passes():
    assert acceptance_errors(valid(), jev_enabled=True) == []


@pytest.mark.parametrize("mutation", [
    lambda r: r["workflows"][0].update(failure="runtime error"),
    lambda r: r["workflows"][0]["jev"].update(http_successes=0),
    lambda r: r["workflows"][0]["jev"].update(routing_audit_complete=False),
    lambda r: r.update(workflows=[]),
    lambda r: r["workflows"][0].pop("task_success"),
    lambda r: r["workflows"][0].pop("framework_model_calls_by_target"),
])
def test_failed_or_degraded_run_cannot_open_batch_gate(mutation):
    result = deepcopy(valid())
    mutation(result)
    assert acceptance_errors(result, jev_enabled=True)


def test_disabled_backend_is_not_jev_acceptance():
    assert acceptance_errors(valid(), jev_enabled=False)


@pytest.mark.parametrize("reason", ["evaluation_failed", "context_length_exceeded", "agent_terminated"])
def test_integration_gate_does_not_select_only_successful_task_trajectories(reason):
    result = valid()
    result["workflows"][0].update(task_success=False, task_failure_reason=reason)
    assert acceptance_errors(result, jev_enabled=True) == []


def test_known_context_error_is_a_task_failure_even_if_outer_layer_raised():
    result = valid()
    result["workflows"][0].update(task_success=False, failure="ContextWindowExceededError",
                                  task_failure_reason="context_length_exceeded")
    assert acceptance_errors(result, jev_enabled=True) == []


def test_appworld_experiment_names_do_not_reuse_directory_basename():
    assert new_experiment_name("099174d_1") != new_experiment_name("099174d_1")
