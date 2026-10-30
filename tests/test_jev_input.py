"""Jev inputs preserve the observed prefix and each mode's information boundary."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib
import json
from types import SimpleNamespace

import pytest

from traceoffload.contracts import MessageRole, ModelMessage, ModelMessageToolCall
from traceoffload.m1.span_embedding import SpanActionsTextBuilder, SpanEmbeddingTextBuilderV2


def _module():
    return importlib.import_module("traceoffload.m2.jev.input_builder")


def _config(mode="context", **updates):
    values = dict(input_mode=mode, model_id="jev-router", max_history_spans=10,
                  max_state_chars=12000)
    values.update(updates)
    return SimpleNamespace(**values)


def _context(goal="Update the specified contact"):
    return SimpleNamespace(
        workflow_run_id="workflow-1", episode_id="episode-1",
        logical_invocation_id="invocation-1",
        workflow_context=SimpleNamespace(workflow_goal=SimpleNamespace(
            goal_text=goal, goal_ref=SimpleNamespace(kind="goal", id="private-ref", version="v1"))),
        caller_deadline_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        prior_execution_facts=None,
    )


def _profile(**updates):
    values = dict(workflow_run_id="workflow-1", episode_id="episode-1",
                  logical_invocation_id="invocation-1",
                  online_features={"prefix_n_A": 2.0, "log_est_prompt_tokens": 7.2})
    values.update(updates)
    return SimpleNamespace(**values)


def _request(*messages):
    return SimpleNamespace(messages=tuple(messages), native_payload={"untouched": [1, 2]})


def _msg(role, content, **kwargs):
    return ModelMessage(role=MessageRole(role), content=content, **kwargs)


def _builder(mode="context", *, skip=0, **kwargs):
    builder = _module().JevInputBuilder(_config(mode, **kwargs),
                                      large_model="executor-large", small_model="executor-small")
    builder.set_skip_leading_messages(skip)
    return builder


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize("mode,parts", [
    ("context", {"model_info", "context"}),
    ("four_features", {"model_info", "features"}),
    ("combined", {"model_info", "context", "features"}),
])
def test_modes_send_one_shared_question_and_actual_executor_identities(mode, parts):
    prepared = _builder(mode).build(_context(), _request(_msg("user", "Begin")), _profile())
    assert prepared.error is None
    body = prepared.request
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == "jev-router"
    assert set(body["state"]) == parts
    assert body["state"]["model_info"] == {
        "large_model": "executor-large", "small_model": "executor-small",
        "execution_policy": "The small model uses the existing guard and large-model fallback.",
    }
    assert list(body["questions"]) == ["route"]
    question = body["questions"]["route"]
    assert question["type"] == "choice"
    expected_instructions = (_module().ROUTE_CONTEXT_INSTRUCTIONS_V3 if mode == "context"
                             else _module().ROUTE_INSTRUCTIONS_V2)
    assert question["instructions"] == expected_instructions
    assert list(question["criteria"]) == ["LLM", "SLM"]
    assert all(ord(character) < 128 for character in question["instructions"])
    assert all(ord(character) < 128 for value in question["criteria"].values() for character in value)
    assert "executor-small" not in question["instructions"]
    if mode == "context":
        assert "four feature" not in question["instructions"].lower()
        assert "prefix_n_A" not in question["instructions"]
        assert "log_est_prompt_tokens" not in question["instructions"]
        assert "V2_goal_recent" not in question["instructions"]
        assert "C_bare_w10" not in question["instructions"]
        assert prepared.metadata["prompt_version"] == _module().CONTEXT_PROMPT_VERSION
    else:
        assert prepared.metadata["prompt_version"] == _module().PROMPT_VERSION


@pytest.mark.parametrize("version,expected_constant", [
    ("v1", "ROUTE_CONTEXT_INSTRUCTIONS_V3"),
    ("v2", "ROUTE_CONTEXT_INSTRUCTIONS_V2_BALANCED"),
    ("v3", "ROUTE_CONTEXT_INSTRUCTIONS_V3_PROACTIVE"),
])
def test_context_prompt_versions_keep_choice_criteria_identical(version, expected_constant):
    module = _module()
    prepared = _builder(prompt_version=version).build(
        _context(), _request(_msg("user", "Begin")), _profile())
    question = prepared.request["questions"]["route"]
    assert question["instructions"] == getattr(module, expected_constant)
    assert question["criteria"] == dict(module.ROUTE_CRITERIA_V2)
    assert prepared.metadata["prompt_version"] == module.CONTEXT_PROMPT_VERSIONS[version][1]
    assert prepared.metadata["prompt_sha256"] == sha256(_json({
        "instructions": question["instructions"], "criteria": question["criteria"]
    }).encode()).hexdigest()


def test_four_features_reuses_both_existing_text_recipes_and_ignores_extra_observations():
    context, profile = _context(), _profile()
    request = _request(_msg("assistant", "fewshot"), _msg("user", "fewshot output"),
                       _msg("assistant", "```python\napis.contacts.search_contacts(query='Li')\n```",
                            reasoning_content="Confirm the person"),
                       _msg("user", "PRIVATE TOOL RESULT: two contacts"))
    prepared = _builder("four_features", skip=2).build(context, request, profile)
    v2, actions = SpanEmbeddingTextBuilderV2(), SpanActionsTextBuilder()
    v2.set_skip_leading_messages(2)
    actions.set_skip_leading_messages(2)
    assert prepared.request["state"]["features"] == {
        **profile.online_features,
        "V2_goal_recent": v2.build(context, request, profile).text,
        "C_bare_w10": actions.build(context, request, profile).text,
    }
    assert "PRIVATE TOOL RESULT" not in _json(prepared.request)
    assert "fewshot" not in _json(prepared.request)


def test_no_prior_actions_is_a_valid_empty_feature():
    prepared = _builder("four_features").build(_context(), _request(_msg("user", "Start")), _profile())
    assert prepared.error is None
    assert prepared.request["state"]["features"]["C_bare_w10"] == ""


def test_four_feature_metadata_identifies_both_actual_source_windows():
    messages = [_msg("assistant", "template")]
    messages.extend(_msg("assistant", f"```python\nprint({i})\n```") for i in range(12))
    messages.extend(_msg("assistant", "") for _ in range(3))
    prepared = _builder("four_features", skip=1).build(_context(), _request(*messages), _profile())
    # V2 counts empty assistant spans, while C_bare counts only nonempty actions.
    assert prepared.metadata["feature_v2_message_indices"] == list(range(6, 16))
    assert prepared.metadata["feature_actions_message_indices"] == list(range(3, 13))
    assert prepared.metadata["history_message_start"] == 3
    assert prepared.metadata["history_message_end"] == 15


def test_context_does_not_construct_feature_builders(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("context-only must not construct feature builders")
    monkeypatch.setattr(_module(), "SpanEmbeddingTextBuilderV2", unexpected)
    monkeypatch.setattr(_module(), "SpanActionsTextBuilder", unexpected)
    prepared = _builder("context").build(_context(), _request(_msg("user", "Start")),
                                         _profile(online_features=None))
    assert prepared.error is None


def test_context_window_keeps_whole_action_observation_units_and_user_corrections():
    call = ModelMessageToolCall(tool_call_id="call-2", tool_name="contacts.search",
                               arguments={"query": "Li"})
    request = _request(
        _msg("assistant", "template"), _msg("user", "template result"),
        _msg("assistant", "old action"), _msg("user", "old result"),
        _msg("assistant", None, tool_calls=(call,)),
        _msg("tool", [{"type": "text", "text": "two contacts"}], tool_call_id="call-2"),
        _msg("user", "Correction: use the second contact"),
    )
    prepared = _builder(skip=2, max_history_spans=1).build(_context(), request, _profile())
    history = prepared.request["state"]["context"]["history"]
    assert len(history) == 1
    messages = history[0]["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "user"]
    assert messages[0]["tool_calls"][0]["arguments"] == {"query": "Li"}
    assert messages[1]["tool_call_id"] == "call-2"
    assert messages[1]["content"][0]["text"] == "two contacts"
    assert messages[2]["content"] == "Correction: use the second contact"
    assert "template" not in _json(prepared.request)
    assert "old action" not in _json(prepared.request)
    assert prepared.metadata["history_message_start"] == 4
    assert prepared.metadata["history_message_end"] == 6


def test_combined_deduplicates_only_fully_covered_source_text_and_keeps_results():
    context = _context()
    request = _request(
        _msg("assistant", "```python\napis.contacts.search_contacts(query='Li')\n```",
             reasoning_content="Find person"),
        _msg("user", "contacts.search_contacts returned FIRST"),
        _msg("assistant", "```python\napis.contacts.search_contacts(query='Li')\n```"),
        _msg("user", "contacts.search_contacts returned SECOND"),
        _msg("user", "Correction: use the second person"),
    )
    four = _builder("four_features").build(context, request, _profile())
    combined = _builder("combined").build(context, request, _profile())
    state = combined.request["state"]
    assert state["features"] == four.request["state"]["features"]
    assert "goal_text" not in state["context"]
    supplemental = _json(state["context"])
    assert "FIRST" in supplemental and "SECOND" in supplemental
    assert "Correction: use the second person" in supplemental
    assert "```python" not in supplemental
    assert "Find person" not in supplemental
    assert combined.metadata["deduplicated_items"] >= 3


def test_combined_keeps_long_text_when_features_only_contain_its_tail():
    content = "must-retain-content-head:" + "x" * 1600
    reasoning = "must-retain-reasoning-head:" + "r" * 900
    request = _request(_msg("assistant", content, reasoning_content=reasoning),
                       _msg("user", "Observed result"))
    prepared = _builder("combined").build(_context(), request, _profile())
    supplemental = _json(prepared.request["state"]["context"])
    assert content in supplemental
    assert reasoning in supplemental


def test_serialized_state_budget_removes_old_units_and_preserves_latest():
    context, profile = _context(), _profile()
    latest = [_msg("assistant", "latest action"), _msg("user", "latest result")]
    minimum = _builder().build(context, _request(*latest), profile)
    limit = len(_json(minimum.request["state"])) + 10
    request = _request(_msg("assistant", "old" * 100), _msg("user", "old output"), *latest)
    prepared = _builder(max_state_chars=limit).build(context, request, profile)
    assert prepared.error is None
    state = prepared.request["state"]
    assert "old output" not in _json(state)
    assert "latest action" in _json(state) and "latest result" in _json(state)
    assert state["context"]["goal_text"] == context.workflow_context.workflow_goal.goal_text
    assert prepared.metadata["dropped_history_units"] == 1
    assert prepared.metadata["state_chars"] == len(_json(state)) <= limit


@pytest.mark.parametrize("mode", ["context", "four_features", "combined"])
def test_over_limit_does_not_truncate_required_data_or_change_mode(mode):
    prepared = _builder(mode, max_state_chars=40).build(
        _context(), _request(_msg("assistant", "required latest action")), _profile())
    assert prepared.request is None
    assert prepared.error == "input_too_large"
    assert prepared.metadata["input_mode"] == mode


@pytest.mark.parametrize("bad", [None, {}, {"prefix_n_A": True, "log_est_prompt_tokens": 2},
                                  {"prefix_n_A": 1, "log_est_prompt_tokens": float("nan")},
                                  {"prefix_n_A": 1, "log_est_prompt_tokens": float("inf")}])
def test_four_features_rejects_missing_or_invalid_scalars(bad):
    prepared = _builder("four_features").build(_context(), _request(_msg("user", "Start")),
                                               _profile(online_features=bad))
    assert prepared.request is None
    assert prepared.error == "input_missing_features"


def test_context_marks_unknown_goal_without_using_reference_as_goal():
    prepared = _builder().build(_context(goal=None), _request(_msg("user", "Known observation")), _profile())
    assert prepared.error is None
    state = prepared.request["state"]["context"]
    assert state["goal_text"] is None
    assert state["goal_available"] is False
    assert "private-ref" not in _json(prepared.request)


def test_context_normalizes_whitespace_goal_as_unavailable():
    prepared = _builder().build(
        _context(goal="   "), _request(_msg("user", "Known observation")), _profile())

    assert prepared.error is None
    state = prepared.request["state"]["context"]
    assert state["goal_text"] is None
    assert state["goal_available"] is False
    serialized = _json(prepared.request)
    assert "024c982_1" not in serialized
    assert "AppWorld task 024c982_1" not in serialized


@pytest.mark.parametrize("mode", ["four_features", "combined"])
def test_features_do_not_treat_an_unresolved_goal_reference_as_task_text(mode):
    prepared = _builder(mode).build(_context(goal=None), _request(_msg("user", "Known result")), _profile())
    assert prepared.error == "input_missing_goal"
    assert prepared.request is None


def test_combined_budget_keeps_features_and_latest_observation_as_one_required_input():
    context, profile = _context(), _profile()
    request = _request(_msg("assistant", "old action"), _msg("user", "old observation" * 100),
                       _msg("assistant", "latest action"), _msg("user", "latest observation"))
    full = _builder("combined").build(context, request, profile)
    required = deepcopy(full.request["state"])
    required["context"]["history"] = required["context"]["history"][-1:]
    prepared = _builder("combined", max_state_chars=len(_json(required))).build(context, request, profile)
    assert prepared.error is None
    assert prepared.request["state"]["features"] == full.request["state"]["features"]
    assert prepared.request["state"] == required
    assert prepared.metadata["dropped_history_units"] == 1
    too_small = _builder("combined", max_state_chars=len(_json(required)) - 1).build(context, request, profile)
    assert too_small.error == "input_too_large"
    assert too_small.request is None


def test_combined_does_not_deduplicate_an_identical_message_outside_v2_window():
    # The same text is insufficient evidence: its source must be in V2's window.
    messages = []
    for number in range(11):
        messages.extend([_msg("assistant", "repeated action"), _msg("user", f"result {number}")])
    prepared = _builder("combined", max_history_spans=11).build(_context(), _request(*messages), _profile())
    history = prepared.request["state"]["context"]["history"]
    assert history[0]["messages"][0]["content"] == "repeated action"
    assert "content" not in history[-1]["messages"][0]


def test_identity_mismatch_and_missing_history_cannot_produce_a_request():
    builder = _builder()
    mismatch = builder.build(_context(), _request(_msg("user", "Start")),
                             _profile(logical_invocation_id="different-invocation"))
    assert mismatch.error == "input_identity_mismatch"
    missing = _builder(skip=2).build(_context(), _request(_msg("user", "Only one")), _profile())
    assert missing.error == "input_missing_history"


def test_empty_history_without_a_known_goal_is_unavailable():
    prepared = _builder(skip=1).build(_context(goal=None), _request(_msg("user", "template")), _profile())
    assert prepared.request is None
    assert prepared.error == "input_missing_context"


def test_unknown_template_skip_is_unavailable_and_negative_skip_is_rejected():
    builder = _module().JevInputBuilder(_config(), large_model="large", small_model="small")
    prepared = builder.build(_context(), _request(_msg("assistant", "may be fewshot")), _profile())
    assert prepared.error == "input_unknown_template_skip"
    assert prepared.request is None
    with pytest.raises(ValueError):
        builder.set_skip_leading_messages(-1)


def test_input_is_a_detached_prefix_snapshot_with_stable_hash_and_local_deadline():
    context = _context()
    request = _request(_msg("user", [{"type": "text", "text": "A known fact"}]))
    original = deepcopy(request)
    first = _builder().build(context, request, _profile())
    second = _builder().build(context, request, _profile())
    assert first.request == second.request
    assert first.metadata["input_sha256"] == sha256(_json(first.request).encode("utf-8")).hexdigest()
    assert first.metadata["input_sha256"] == second.metadata["input_sha256"]
    assert first.caller_deadline_at == context.caller_deadline_at
    assert "caller_deadline_at" not in _json(first.request)
    first.request["state"]["context"]["history"][0]["messages"][0]["content"][0]["text"] = "changed"
    assert request.messages == original.messages
    assert request.native_payload == original.native_payload


def test_context_store_uses_full_identity_takes_once_and_clears_only_one_workflow():
    module = _module()
    store = module.RoutingContextStore()
    context = _context()
    other_episode = _profile(episode_id="episode-2")
    other_workflow = _profile(workflow_run_id="workflow-2")
    prepared = module.PreparedInput(request=None, caller_deadline_at=None,
                                    error="input_missing_features", metadata={})
    for source in (context, other_episode, other_workflow):
        store.put(module.key_for(source), prepared)
    assert store.take(module.key_for(context)) is prepared
    assert store.take(module.key_for(context)) is None
    store.clear_workflow("workflow-1")
    assert store.take(module.key_for(other_episode)) is None
    assert store.take(module.key_for(other_workflow)) is prepared
    store.put(module.key_for(context), prepared)
    store.clear()
    assert store.take(module.key_for(context)) is None
