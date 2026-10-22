"""Small local types and strict validation for native System One decisions."""

from __future__ import annotations

from dataclasses import dataclass, fields
import ipaddress
import math
import os
import re
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


SystemOneRequest = dict[str, Any]
SystemOneResponse = dict[str, Any]


class JevUnavailable(Exception):
    """A bounded, safe error category; never retain headers or response bodies."""

    def __init__(self, reason: str, *, attempted: bool = False) -> None:
        self.reason = reason
        self.attempted = attempted
        super().__init__(reason)


@dataclass(frozen=True)
class JevConfig:
    model_id: str
    endpoint: str
    expected_revision: str | None = None
    api_key_env: str | None = None
    mode: Literal["shadow", "active"] = "shadow"
    input_mode: Literal["context", "four_features", "combined"] = "combined"
    threshold: float = 0.8
    request_timeout_ms: int = 2000
    connect_timeout_ms: int = 500
    execution_reserve_ms: int = 1000
    max_history_spans: int = 10
    max_state_chars: int = 12000


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _enabled(block: dict, path: str) -> bool:
    value = block.get("enabled", False)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.enabled must be a boolean")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{path} must be a nonempty string without surrounding whitespace")
    return value


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _origin(value: Any) -> str:
    endpoint = _text(value, "runner.jev.endpoint")
    try:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or "?" in endpoint
            or "#" in endpoint
            or "\\" in endpoint
            or any(character.isspace() or ord(character) < 32 for character in endpoint)
            or "%" in parsed.hostname
        ):
            raise ValueError
        ipaddress.ip_address(parsed.hostname)
        port = parsed.port  # Accessing the property validates the port.
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError("runner.jev.endpoint must be an http/https IP origin without credentials, path, query or fragment") from None
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def load_jev_config(full_config: dict, *, mode: str) -> JevConfig | None:
    """Read Jev before loading legacy predictors; disabled settings stay untouched."""
    runner = _mapping(full_config.get("runner", {}), "runner")
    raw = _mapping(runner.get("jev", {}), "runner.jev")
    if not _enabled(raw, "runner.jev"):
        return None
    if mode != "classifier_route":
        raise ValueError("runner.jev requires classifier_route")
    if "input_template" in raw:
        raise ValueError("runner.jev.input_template is obsolete; use input_mode")
    unknown = set(raw) - {field.name for field in fields(JevConfig)} - {"enabled"}
    if unknown:
        raise ValueError("runner.jev contains unknown settings")
    jev_mode = raw.get("mode", "shadow")
    input_mode = raw.get("input_mode", "combined")
    if jev_mode not in ("active", "shadow"):
        raise ValueError("runner.jev.mode must be shadow or active")
    if input_mode not in ("context", "four_features", "combined"):
        raise ValueError("runner.jev.input_mode must be context, four_features or combined")
    if jev_mode == "shadow" and not _enabled(_mapping(runner.get("l2_add4", {}), "runner.l2_add4"), "runner.l2_add4"):
        raise ValueError("Jev shadow requires runner.l2_add4.enabled=true")
    if _enabled(_mapping(runner.get("shadow_route", {}), "runner.shadow_route"), "runner.shadow_route"):
        raise ValueError("runner.jev and runner.shadow_route cannot both be enabled")
    m4 = _mapping(runner.get("m4_memory", {}), "runner.m4_memory")
    if _enabled(m4, "runner.m4_memory"):
        if m4.get("embedding_mode") != "m1_online" or m4.get("template_version") != "span-embed-v2":
            raise ValueError("Jev with M4 requires m1_online and span-embed-v2")
    values = {field.name: raw.get(field.name, field.default) for field in fields(JevConfig)}
    values["model_id"] = _text(raw.get("model_id"), "runner.jev.model_id")
    values["endpoint"] = _origin(raw.get("endpoint"))
    threshold = values["threshold"]
    if not _finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("runner.jev.threshold must be a finite probability")
    values["threshold"] = float(threshold)
    for name in ("request_timeout_ms", "connect_timeout_ms", "execution_reserve_ms", "max_history_spans", "max_state_chars"):
        if type(values[name]) is not int or values[name] <= 0:
            raise ValueError(f"runner.jev.{name} must be a positive integer")
    if values["connect_timeout_ms"] > values["request_timeout_ms"]:
        raise ValueError("runner.jev.connect_timeout_ms exceeds request_timeout_ms")
    if input_mode != "context" and values["max_history_spans"] != 10:
        raise ValueError("four_features and combined require max_history_spans=10")
    for name in ("expected_revision", "api_key_env"):
        if values[name] is not None:
            values[name] = _text(values[name], f"runner.jev.{name}")
    key_name = values["api_key_env"]
    if key_name is not None:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_name) is None:
            raise ValueError("runner.jev.api_key_env must name an environment variable")
        if not os.environ.get(key_name):
            raise ValueError(f"runner.jev requires environment variable {key_name}")
    return JevConfig(**values)


def validate_response(raw: Any, model_id: str) -> SystemOneResponse:
    """Validate the required native fields while preserving service extensions."""
    if not isinstance(raw, dict) or raw.get("model") != model_id:
        raise JevUnavailable("model_mismatch" if isinstance(raw, dict) and raw.get("model") else "invalid_response")
    answers = raw.get("answers")
    route = answers.get("route") if isinstance(answers, dict) else None
    if not isinstance(route, dict) or route.get("type") != "choice":
        raise JevUnavailable("invalid_response")
    probabilities = route.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != {"LLM", "SLM"}:
        raise JevUnavailable("invalid_response")
    if any(not _finite_number(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise JevUnavailable("invalid_response")
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise JevUnavailable("invalid_response")
    if "choice" in route and route["choice"] not in ("LLM", "SLM"):
        raise JevUnavailable("invalid_response")
    if "usage" in raw:
        usage = raw["usage"]
        if not isinstance(usage, dict) or any(not _finite_number(value) or value < 0 for value in usage.values()):
            raise JevUnavailable("invalid_response")
    return raw


__all__ = ["JevConfig", "JevUnavailable", "SystemOneRequest", "SystemOneResponse", "load_jev_config", "validate_response"]
