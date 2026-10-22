"""Configuration and real local HTTP coverage for the optional Jev client."""

from __future__ import annotations

import asyncio
import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import json
import math
import socket
import threading
import time

import pytest


def _types():
    return importlib.import_module("traceoffload.m2.jev.types")


def _config(**changes):
    raw = {"enabled": True, "mode": "active", "model_id": "local-jev", "endpoint": "http://127.0.0.1:8900"}
    raw.update(changes)
    return {"runner": {"jev": raw}}


def _load(raw=None):
    return _types().load_jev_config(raw if raw is not None else _config(), mode="classifier_route")


def _request():
    return {
        "model": "local-jev",
        "state": {"model_info": {"large_model": "large", "small_model": "small"}, "context": {"goal": "任务"}},
        "questions": {"route": {"type": "choice", "instructions": "Choose the next execution model.", "criteria": {"LLM": "large", "SLM": "small"}}},
    }


def _response():
    return {"model": "local-jev", "answers": {"route": {"type": "choice", "choice": "SLM", "probabilities": {"LLM": 0.2, "SLM": 0.8}}}, "usage": {"input_tokens": 42, "output_tokens": 0}}


def test_optional_package_exists():
    assert importlib.util.find_spec("traceoffload.m2.jev") is not None


def test_disabled_configuration_ignores_endpoint_secret_and_unknown_fields(monkeypatch):
    monkeypatch.delenv("JEV_UNSET_KEY", raising=False)
    assert _load({}) is None
    assert _load(_config(enabled=False, endpoint="invalid", api_key_env="JEV_UNSET_KEY", input_template="obsolete")) is None


def test_configuration_defaults_and_immutability():
    config = _load()
    assert config.mode == "active"
    assert config.input_mode == "combined"
    assert config.threshold == 0.8
    assert config.request_timeout_ms == 2000
    assert config.connect_timeout_ms == 500
    assert config.execution_reserve_ms == 1000
    assert config.max_history_spans == 10
    assert config.max_state_chars == 12000
    assert config.expected_revision is None
    with pytest.raises(FrozenInstanceError):
        config.threshold = 0.1


def test_active_configuration_does_not_validate_legacy_artifact():
    raw = _config()
    raw["runner"]["l2_add4"] = {"enabled": True, "model_path": "missing-file"}
    assert _load(raw).mode == "active"


def test_shadow_default_requires_add4_but_does_not_load_it():
    raw = _config()
    del raw["runner"]["jev"]["mode"]
    with pytest.raises(ValueError, match="add4"):
        _load(raw)
    raw["runner"]["l2_add4"] = {"enabled": True, "model_path": "missing-file"}
    assert _load(raw).mode == "shadow"


@pytest.mark.parametrize("changes", [
    {"enabled": "false"}, {"enabled": 0}, {"mode": "ACTIVE"}, {"input_mode": "context_v1"},
    {"input_template": "context_v1"}, {"threshold": True}, {"threshold": "0.8"},
    {"threshold": math.nan}, {"threshold": math.inf}, {"threshold": -0.1}, {"threshold": 1.1},
    {"request_timeout_ms": 0}, {"request_timeout_ms": 2.5}, {"connect_timeout_ms": True},
    {"connect_timeout_ms": 2001}, {"execution_reserve_ms": -1}, {"max_state_chars": 0},
    {"max_history_spans": 9}, {"model_id": ""}, {"model_id": 1}, {"expected_revision": 42},
    {"api_key_env": ""}, {"unexpected_typo": True},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        _load(_config(**changes))


@pytest.mark.parametrize("endpoint", [
    "http://localhost:8900", "http://service.internal", "ftp://127.0.0.1", "http://127.0.0.1/v1",
    "http://user:secret@127.0.0.1", "http://127.0.0.1?q=1", "http://127.0.0.1#fragment",
    "http://127.0.0.1:99999", "http://127.0.0.1?", "http://127.0.0.1#",
    "http://127.0.\n0.1", "http://127.0.0.1:\t8900",
])
def test_endpoint_is_an_ip_origin(endpoint):
    with pytest.raises(ValueError):
        _load(_config(endpoint=endpoint))


def test_ipv6_origin_and_context_window_are_supported():
    config = _load(_config(endpoint="http://[::1]:8900/", input_mode="context", max_history_spans=3))
    assert config.endpoint == "http://[::1]:8900"
    assert config.max_history_spans == 3


def test_modes_and_m4_dependencies_are_validated():
    with pytest.raises(ValueError, match="classifier_route"):
        _types().load_jev_config(_config(), mode="llm_only")
    raw = _config()
    raw["runner"]["shadow_route"] = {"enabled": True}
    with pytest.raises(ValueError, match="shadow_route"):
        _load(raw)
    raw["runner"].pop("shadow_route")
    raw["runner"]["m4_memory"] = {"enabled": True}
    with pytest.raises(ValueError, match="m1_online"):
        _load(raw)
    raw["runner"]["m4_memory"].update(embedding_mode="m1_online", template_version="span-embed-v2")
    assert _load(raw) is not None


def test_enabled_secret_missing_errors_do_not_contain_secret_values(monkeypatch):
    monkeypatch.delenv("JEV_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_TEST_KEY"):
        _load(_config(api_key_env="JEV_TEST_KEY"))
    monkeypatch.setenv("JEV_TEST_KEY", "private-test-token")
    config = _load(_config(api_key_env="JEV_TEST_KEY"))
    assert "private-test-token" not in repr(config)


def test_optional_client_import_does_not_import_httpx(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise AssertionError("httpx was imported on the disabled path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(importlib.import_module("traceoffload.m2.jev.client"))
    assert _load(_config(enabled=False)) is None


def test_valid_response_preserves_native_extensions_and_does_not_require_revision():
    response = _response()
    response["marks"] = {"source": "native"}
    response["answers"]["route"]["calibration"] = {"version": "native"}
    assert _types().validate_response(response, "local-jev") == response


@pytest.mark.parametrize("probabilities", [
    {"LLM": 0.2}, {"LLM": 0.2, "SLM": 0.8, "OTHER": 0}, {"LLM": False, "SLM": 1.0},
    {"LLM": "0.2", "SLM": 0.8}, {"LLM": math.nan, "SLM": 0.8},
    {"LLM": math.inf, "SLM": 0.8}, {"LLM": -0.1, "SLM": 1.1}, {"LLM": 0.3, "SLM": 0.8},
])
def test_invalid_response_probabilities_are_rejected(probabilities):
    response = _response()
    response["answers"]["route"]["probabilities"] = probabilities
    with pytest.raises(_types().JevUnavailable, match="invalid_response"):
        _types().validate_response(response, "local-jev")


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model="other"),
    lambda r: r.update(answers={}),
    lambda r: r["answers"]["route"].update(type="boolean"),
    lambda r: r["answers"]["route"].update(choice="OTHER"),
    lambda r: r.update(usage={"input_tokens": -1}),
    lambda r: r.update(usage={"input_tokens": True}),
    lambda r: r.update(usage={"input_tokens": math.inf}),
])
def test_response_identity_shape_and_usage_are_validated(mutation):
    response = _response()
    mutation(response)
    with pytest.raises(_types().JevUnavailable):
        _types().validate_response(response, "local-jev")


@contextmanager
def _server(*, status=200, payload=None, raw_body=None, delay=0.0, chunks=None, location=None):
    received = []
    response_bytes = raw_body if raw_body is not None else json.dumps(payload if payload is not None else _response()).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
            try:
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(sum(len(c) for c in chunks) if chunks else len(response_bytes)))
                if location:
                    self.send_header("Location", location)
                self.end_headers()
                if chunks:
                    for chunk in chunks:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        time.sleep(0.03)
                else:
                    self.wfile.write(response_bytes)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def _client(endpoint, *, preflight=lambda request: 42, **changes):
    cls = importlib.import_module("traceoffload.m2.jev.client").HttpJevClient
    return cls(_load(_config(endpoint=endpoint, **changes)), input_preflight=preflight)


def test_missing_or_failed_token_preflight_never_sends():
    with _server() as (endpoint, received):
        client = _client(endpoint, preflight=None)
        with pytest.raises(_types().JevUnavailable, match="input_unverified"):
            client.predict(_request(), timeout_ms=500)

        def too_long(_request):
            raise _types().JevUnavailable("input_too_long")

        with pytest.raises(_types().JevUnavailable, match="input_too_long"):
            _client(endpoint, preflight=too_long).predict(_request(), timeout_ms=500)
        assert received == []


def test_preflight_may_not_change_the_request_it_verifies():
    def mutating_preflight(request):
        request["state"].clear()
        return 1

    with _server() as (endpoint, received):
        with pytest.raises(_types().JevUnavailable, match="input_unverified") as caught:
            _client(endpoint, preflight=mutating_preflight).predict(_request(), timeout_ms=3000)
        assert caught.value.attempted is False
        assert received == []


def test_preflight_time_is_included_in_the_budget():
    def slow_preflight(_request):
        time.sleep(0.08)
        return 42

    with _server() as (endpoint, received):
        with pytest.raises(_types().JevUnavailable, match="timeout") as caught:
            _client(endpoint, preflight=slow_preflight).predict(_request(), timeout_ms=40)
        assert caught.value.attempted is False
        assert received == []


def test_successful_preflight_token_count_is_available_to_audit():
    with _server() as (endpoint, _received):
        client = _client(endpoint, preflight=lambda _request: 37)
        client.predict(_request(), timeout_ms=5000)
        assert client.last_input_tokens == 37


def test_failed_http_keeps_successful_preflight_token_count():
    with _server(status=500) as (endpoint, _received):
        client = _client(endpoint, preflight=lambda _request: 41)
        with pytest.raises(_types().JevUnavailable, match="http_error"):
            client.predict(_request(), timeout_ms=5000)
        assert client.last_input_tokens == 41


def test_invalid_native_request_is_rejected_before_transmission():
    with _server() as (endpoint, received):
        request = _request()
        request["questions"]["extra"] = request["questions"]["route"]
        with pytest.raises(_types().JevUnavailable, match="invalid_request") as caught:
            _client(endpoint).predict(request, timeout_ms=500)
        assert caught.value.attempted is False
        request = _request()
        request["state"]["oversized"] = "x" * 12000
        with pytest.raises(_types().JevUnavailable, match="input_too_long"):
            _client(endpoint).predict(request, timeout_ms=500)
        assert received == []


def test_deeply_nested_request_is_an_input_error():
    request = _request()
    nested = request["state"]
    for _ in range(1200):
        nested["nested"] = {}
        nested = nested["nested"]
    with _server() as (endpoint, received):
        with pytest.raises(_types().JevUnavailable, match="invalid_request"):
            _client(endpoint).predict(request, timeout_ms=500)
        assert received == []


def test_real_http_body_is_native_ordered_and_bearer_is_not_in_body(monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", "private-test-token")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    with _server() as (endpoint, received):
        request = _request()
        request["local_request_id"] = "local-only"
        before = deepcopy(request)
        # Cold optional imports and SSL context initialization on mounted WSL
        # storage can take seconds; this case checks serialization, not timing.
        response = _client(endpoint, api_key_env="JEV_TEST_KEY", expected_revision="deployment-record-only", request_timeout_ms=5000).predict(request, timeout_ms=5000)
        assert response == _response()
        assert request == before
        assert len(received) == 1
        assert received[0]["path"] == "/v1/systemone"
        assert received[0]["authorization"] == "Bearer private-test-token"
        body = received[0]["body"]
        assert set(json.loads(body)) == {"model", "state", "questions"}
        assert body.index(b'"LLM"') < body.index(b'"SLM"')
        assert b"private-test-token" not in body


def test_ip_literal_does_not_use_a_dns_executor(monkeypatch):
    with _server() as (endpoint, received):
        def forbidden_dns(*_args, **_kwargs):
            raise AssertionError("IP-only Jev requests must not start DNS work")

        monkeypatch.setattr(socket, "getaddrinfo", forbidden_dns)
        assert _client(endpoint, request_timeout_ms=5000).predict(_request(), timeout_ms=5000) == _response()
        assert len(received) == 1


@pytest.mark.parametrize("status", [302, 422, 429, 500, 529])
def test_http_failures_do_not_retry_or_follow_redirects(status):
    with _server(status=status, location="http://127.0.0.1:1/secret") as (endpoint, received):
        with pytest.raises(_types().JevUnavailable) as caught:
            _client(endpoint).predict(_request(), timeout_ms=500)
        assert caught.value.reason in {"http_redirect", "input_rejected", "service_busy", "http_error"}
        assert caught.value.attempted is True
        assert len(received) == 1


@pytest.mark.parametrize("body,reason", [(b"not-json", "invalid_response"), (b"x" * 65537, "response_too_large")])
def test_invalid_or_oversized_body_is_rejected(body, reason):
    with _server(raw_body=body) as (endpoint, _received):
        with pytest.raises(_types().JevUnavailable, match=reason):
            _client(endpoint).predict(_request(), timeout_ms=500)


def test_duplicate_json_keys_are_rejected():
    with _server(raw_body=b'{"model":"wrong","model":"local-jev","answers":{}}') as (endpoint, _received):
        with pytest.raises(_types().JevUnavailable, match="invalid_response"):
            _client(endpoint).predict(_request(), timeout_ms=500)


def test_slow_chunked_body_uses_total_deadline():
    body = json.dumps(_response()).encode()
    with _server(chunks=[body[index:index + 10] for index in range(0, len(body), 10)]) as (endpoint, received):
        started = time.perf_counter()
        with pytest.raises(_types().JevUnavailable, match="timeout"):
            _client(endpoint).predict(_request(), timeout_ms=120)
        assert time.perf_counter() - started < 1
        assert len(received) == 1


@pytest.mark.skipif(not hasattr(time, "CLOCK_MONOTONIC"), reason="The production clock requires Unix clock_gettime")
def test_frozen_appworld_clock_does_not_disable_timeout():
    from freezegun import freeze_time
    from traceoffload.clock import real_time

    with _server(delay=0.3) as (endpoint, received):
        client = _client(endpoint)
        with freeze_time("2020-01-01"):
            frozen_monotonic = time.monotonic()
            default_loop_clock = asyncio.BaseEventLoop.time
            started = real_time()
            with pytest.raises(_types().JevUnavailable, match="timeout"):
                client.predict(_request(), timeout_ms=80)
            assert 0.05 < real_time() - started < 1
            assert time.monotonic() == frozen_monotonic
            assert asyncio.BaseEventLoop.time is default_loop_clock
        assert len(received) == 1


def test_running_event_loop_caller_is_rejected_without_a_request():
    with _server() as (endpoint, received):
        async def caller():
            with pytest.raises(_types().JevUnavailable, match="unsupported_caller"):
                _client(endpoint).predict(_request(), timeout_ms=500)

        asyncio.run(caller())
        assert received == []


def test_connection_failure_has_a_sanitized_error(monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", "private-test-token")
    with _server() as (endpoint, _received):
        pass
    with pytest.raises(_types().JevUnavailable) as caught:
        _client(endpoint, api_key_env="JEV_TEST_KEY").predict(_request(), timeout_ms=100)
    assert caught.value.reason == "connection_error"
    assert "private-test-token" not in str(caught.value)


@pytest.mark.parametrize("secret", ["private\nheader", "private-机密"])
def test_invalid_bearer_values_are_sanitized_before_http(monkeypatch, secret):
    monkeypatch.setenv("JEV_TEST_KEY", secret)
    with _server() as (endpoint, received):
        with pytest.raises(_types().JevUnavailable, match="auth_unavailable") as caught:
            _client(endpoint, api_key_env="JEV_TEST_KEY").predict(_request(), timeout_ms=500)
        assert caught.value.attempted is False
        assert secret not in str(caught.value)
        assert received == []
