"""Bounded synchronous bridge to a native asynchronous System One HTTP call."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import os
from typing import Protocol

from ...clock import real_time
from .types import JevConfig, JevUnavailable, SystemOneRequest, SystemOneResponse, validate_response


MAX_RESPONSE_BYTES = 64 * 1024


class JevDecisionClient(Protocol):
    def predict(self, request: SystemOneRequest, *, timeout_ms: int) -> SystemOneResponse: ...


def _real_clock_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.time = real_time
    return loop


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite JSON value")


def _native_request(request: SystemOneRequest, config: JevConfig) -> tuple[dict, bytes]:
    """Take an isolated native-only snapshot without reordering the candidates."""
    if not isinstance(request, dict) or request.get("model") != config.model_id:
        raise JevUnavailable("invalid_request")
    questions = request.get("questions")
    if not isinstance(request.get("state"), dict) or not isinstance(questions, dict) or set(questions) != {"route"}:
        raise JevUnavailable("invalid_request")
    route = questions["route"]
    if not isinstance(route, dict) or route.get("type") != "choice" or not isinstance(route.get("instructions"), str) or not route["instructions"].strip():
        raise JevUnavailable("invalid_request")
    criteria = route.get("criteria")
    if not isinstance(criteria, dict) or list(criteria) != ["LLM", "SLM"] or any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
        raise JevUnavailable("invalid_request")
    native = {"model": request["model"], "state": request["state"], "questions": questions}
    try:
        state_text = json.dumps(native["state"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(state_text) > config.max_state_chars:
            raise JevUnavailable("input_too_long")
        body = json.dumps(native, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        snapshot = json.loads(body)
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        raise JevUnavailable("invalid_request") from None
    return snapshot, body


class HttpJevClient:
    """Only an explicitly verified complete-input preflight permits transmission.

    The callback must use the chosen service's tokenizer, full native template
    and validated safe limit. It raises JevUnavailable on unsafe inputs and
    returns the exact positive token count on success. Character limits alone
    cannot supply this guarantee. HTTP remains an optional, lazy dependency.
    """

    def __init__(self, config: JevConfig, *, input_preflight: Callable[[dict], int] | None = None) -> None:
        self._config = config
        self._input_preflight = input_preflight
        self.last_input_tokens: int | None = None

    def predict(self, request: SystemOneRequest, *, timeout_ms: int) -> SystemOneResponse:
        self.last_input_tokens = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise JevUnavailable("unsupported_caller")
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise JevUnavailable("budget_exhausted")
        if self._input_preflight is None:
            raise JevUnavailable("input_unverified")
        deadline = real_time() + min(timeout_ms, self._config.request_timeout_ms) / 1000
        native, body = _native_request(request, self._config)
        try:
            input_tokens = self._input_preflight(native)
            if json.dumps(native, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") != body:
                raise JevUnavailable("input_unverified")
        except JevUnavailable:
            raise
        except Exception:
            raise JevUnavailable("input_unverified") from None
        if type(input_tokens) is not int or input_tokens <= 0:
            raise JevUnavailable("input_unverified")
        self.last_input_tokens = input_tokens
        if real_time() >= deadline:
            raise JevUnavailable("timeout")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._config.api_key_env is not None:
            secret = os.environ.get(self._config.api_key_env)
            if not secret or not secret.isascii() or not secret.isprintable() or any(character.isspace() for character in secret):
                raise JevUnavailable("auth_unavailable")
            headers["Authorization"] = f"Bearer {secret}"
        attempted = {"value": False}
        try:
            with asyncio.Runner(loop_factory=_real_clock_loop) as runner:
                response = runner.run(self._predict(body, headers, deadline, attempted))
            if real_time() >= deadline:
                raise JevUnavailable("timeout", attempted=attempted["value"])
            return response
        except JevUnavailable as error:
            error.attempted = attempted["value"]
            raise

    async def _predict(self, body: bytes, headers: dict, deadline: float, attempted: dict) -> SystemOneResponse:
        try:
            import httpx
        except ImportError:
            raise JevUnavailable("dependency_missing") from None
        try:
            async with asyncio.timeout_at(deadline):
                remaining = deadline - real_time()
                if remaining <= 0:
                    raise JevUnavailable("timeout")
                timeout = httpx.Timeout(remaining, connect=min(remaining, self._config.connect_timeout_ms / 1000))
                async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=timeout) as client:
                    if real_time() >= deadline:
                        raise JevUnavailable("timeout")
                    attempted["value"] = True
                    async with client.stream("POST", self._config.endpoint + "/v1/systemone", content=body, headers=headers) as response:
                        status = response.status_code
                        if 300 <= status < 400:
                            raise JevUnavailable("http_redirect")
                        if status == 422:
                            raise JevUnavailable("input_rejected")
                        if status in {429, 529}:
                            raise JevUnavailable("service_busy")
                        if not 200 <= status < 300:
                            raise JevUnavailable("http_error")
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
                                raise JevUnavailable("response_too_large")
                            data.extend(chunk)
                        try:
                            raw = json.loads(data, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
                        except (ValueError, UnicodeError, RecursionError):
                            raise JevUnavailable("invalid_response") from None
                        validated = validate_response(raw, self._config.model_id)
                        if real_time() >= deadline:
                            raise JevUnavailable("timeout")
                        return validated
        except (TimeoutError, httpx.TimeoutException):
            raise JevUnavailable("timeout") from None
        except httpx.RequestError:
            raise JevUnavailable("connection_error") from None


__all__ = ["HttpJevClient", "JevDecisionClient", "MAX_RESPONSE_BYTES"]
