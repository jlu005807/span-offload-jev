"""可运行骨架的小型确定性 pre-commit 守卫。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..clock import real_time
from ..mechanism_counters import MechanismCounters
from ..interfaces import SpanIdentity
from ..contracts import (
    GuardOutcome,
    GuardRef,
    GuardResult,
    ModelResponse,
    ParseStatus,
    ToolExecutionResult,
    ToolExecutionStatus,
    VerifierStep,
)


def _value_at_path(value: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted argument path without executing user supplied code."""

    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _contains_nul(value: Any) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, Mapping):
        return any(_contains_nul(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_nul(item) for item in value)
    return False


@dataclass(frozen=True)
class ToolRule:
    """Versioned, tool-specific argument and result policy.

    The rule is deliberately data-only.  Complex shell or domain validation is
    selected by ``verifier_ref`` and implemented by a registered verifier, not
    by evaluating arbitrary configuration code.
    """

    tool_name: str
    required_argument_paths: tuple[str, ...] = ()
    non_empty_string_paths: tuple[str, ...] = ()
    allowed_argument_keys: tuple[str, ...] | None = None
    max_argument_bytes: int | None = None
    reject_nul: bool = True
    accepted_statuses: tuple[ToolExecutionStatus, ...] = (ToolExecutionStatus.SUCCESS,)
    accepted_returncodes: tuple[int | None, ...] = (None, 0)
    verifier_ref: GuardRef | None = None


@dataclass(frozen=True)
class ToolPolicy:
    """Effective whitelist used by M2's verifier and M3's result checks."""

    policy_id: str = "tool-policy"
    version: str = "v1"
    allowed_tools: tuple[str, ...] = ("bash",)
    require_tool_calls: bool = False
    max_calls: int | None = None
    rules: tuple[ToolRule, ...] = ()

    def rule_for(self, tool_name: str) -> ToolRule | None:
        if tool_name not in self.allowed_tools:
            return None
        rule = next((rule for rule in self.rules if rule.tool_name == tool_name), None)
        if rule is not None:
            return rule
        # Preserve the old bash contract for programmatic ``ToolPolicy()``
        # callers while all configured tools use explicit registry rules.
        if tool_name == "bash":
            return ToolRule(
                tool_name="bash",
                required_argument_paths=("command",),
                non_empty_string_paths=("command",),
                allowed_argument_keys=("command",),
            )
        return None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        default_allowed_tools: tuple[str, ...] = ("bash",),
        default_require_tool_calls: bool = False,
    ) -> "ToolPolicy":
        """Parse a strict data mapping used by config adapters.

        Unknown keys are ignored at this compatibility boundary because the
        top-level config loader owns schema validation.  Invalid values fail
        closed with ``ValueError`` rather than silently widening the whitelist.
        """

        if not isinstance(raw, Mapping):
            raise ValueError("tool policy must be a mapping")
        raw_allowed = raw.get("allowed_tools", default_allowed_tools)
        if not isinstance(raw_allowed, Sequence) or isinstance(raw_allowed, str):
            raise ValueError("tool policy allowed_tools must be a sequence")
        allowed_tools = tuple(str(item) for item in raw_allowed)
        if not allowed_tools or any(not item.strip() for item in allowed_tools):
            raise ValueError("tool policy allowed_tools cannot be empty")
        tools = raw.get("tools", {})
        if tools is None:
            tools = {}
        if not isinstance(tools, Mapping):
            raise ValueError("tool policy tools must be a mapping")

        def _tuple_strings(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
            if value is None:
                return default
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ValueError("tool policy string list must be a sequence")
            result = tuple(str(item) for item in value)
            if any(not item for item in result):
                raise ValueError("tool policy string list cannot contain empty values")
            return result

        def _statuses(value: Any) -> tuple[ToolExecutionStatus, ...]:
            values = _tuple_strings(value, default=(ToolExecutionStatus.SUCCESS.value,))
            try:
                return tuple(ToolExecutionStatus(item) for item in values)
            except ValueError as error:
                raise ValueError("tool policy contains an unknown tool status") from error

        def _returncodes(value: Any) -> tuple[int | None, ...]:
            if value is None:
                return (None, 0)
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ValueError("accepted_returncodes must be a sequence")
            result: list[int | None] = []
            for item in value:
                if item is None:
                    result.append(None)
                elif isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError("accepted_returncodes must contain integers or null")
                else:
                    result.append(item)
            if not result:
                raise ValueError("accepted_returncodes cannot be empty")
            return tuple(result)

        rules: list[ToolRule] = []
        for tool_name in allowed_tools:
            raw_rule = tools.get(tool_name, {})
            if raw_rule is None:
                raw_rule = {}
            if not isinstance(raw_rule, Mapping):
                raise ValueError(f"tool policy rule for {tool_name!r} must be a mapping")
            input_config = raw_rule.get("input", {})
            if input_config is None:
                input_config = {}
            if not isinstance(input_config, Mapping):
                raise ValueError(f"tool policy input for {tool_name!r} must be a mapping")
            required = _tuple_strings(
                raw_rule.get(
                    "required_argument_paths",
                    input_config.get(
                        "required_paths",
                        ("command",) if tool_name == "bash" else None,
                    ),
                )
            )
            non_empty = _tuple_strings(
                raw_rule.get(
                    "non_empty_string_paths",
                    ("command",) if tool_name == "bash" else None,
                )
            )
            allowed_keys_value = raw_rule.get("allowed_argument_keys")
            allowed_keys = (
                _tuple_strings(allowed_keys_value)
                if allowed_keys_value is not None
                else (("command",) if tool_name == "bash" else None)
            )
            max_bytes = raw_rule.get(
                "max_argument_bytes",
                input_config.get("max_serialized_bytes"),
            )
            if max_bytes is not None:
                if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
                    raise ValueError("max_argument_bytes must be a positive integer")
            result_config = raw_rule.get("result", {})
            if result_config is None:
                result_config = {}
            if not isinstance(result_config, Mapping):
                raise ValueError(f"tool policy result for {tool_name!r} must be a mapping")
            verifier_ref = None
            ref_value = raw_rule.get("verifier_ref")
            if isinstance(ref_value, Mapping):
                verifier_ref = GuardRef(
                    kind=str(ref_value.get("kind", "guard")),
                    id=str(ref_value.get("id", f"{tool_name}-policy")),
                    version=str(ref_value.get("version", "v1")),
                )
            rules.append(
                ToolRule(
                    tool_name=tool_name,
                    required_argument_paths=required,
                    non_empty_string_paths=non_empty,
                    allowed_argument_keys=allowed_keys,
                    max_argument_bytes=max_bytes,
                    reject_nul=bool(raw_rule.get("reject_nul", True)),
                    accepted_statuses=_statuses(result_config.get("accepted_statuses")),
                    accepted_returncodes=_returncodes(result_config.get("accepted_returncodes")),
                    verifier_ref=verifier_ref,
                )
            )
        max_calls = raw.get("max_calls")
        if max_calls is not None:
            if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
                raise ValueError("tool policy max_calls must be a positive integer")
        return cls(
            policy_id=str(raw.get("policy_id", "tool-policy")),
            version=str(raw.get("version", "v1")),
            allowed_tools=allowed_tools,
            require_tool_calls=bool(raw.get("require_tool_calls", default_require_tool_calls)),
            max_calls=max_calls,
            rules=tuple(rules),
        )

    @classmethod
    def from_registry(
        cls,
        entries: Sequence[Any],
        *,
        allowed_tools: tuple[str, ...] | None = None,
        require_tool_calls: bool = False,
        policy_id: str = "tool-policy",
        version: str = "v1",
    ) -> "ToolPolicy":
        """Build a policy from adapter registry entries and their schemas."""

        names = (
            tuple(str(entry.tool_name) for entry in entries)
            if allowed_tools is None
            else tuple(allowed_tools)
        )
        rules: list[ToolRule] = []
        by_name = {str(entry.tool_name): entry for entry in entries}
        for name in names:
            entry = by_name.get(name)
            schema = getattr(entry, "input_schema", None) if entry is not None else None
            required = ()
            allowed_keys = None
            if isinstance(schema, Mapping):
                raw_required = schema.get("required", ())
                if isinstance(raw_required, Sequence) and not isinstance(raw_required, str):
                    required = tuple(str(item) for item in raw_required)
                properties = schema.get("properties", {})
                if isinstance(properties, Mapping):
                    allowed_keys = tuple(str(item) for item in properties)
            non_empty = ("command",) if name == "bash" and "command" in required else ()
            rules.append(
                ToolRule(
                    tool_name=name,
                    required_argument_paths=required,
                    non_empty_string_paths=non_empty,
                    allowed_argument_keys=allowed_keys,
                )
            )
        return cls(
            policy_id=policy_id,
            version=version,
            allowed_tools=tuple(names),
            require_tool_calls=require_tool_calls,
            rules=tuple(rules),
        )


class BasicToolCallVerifier:
    """检查候选能否安全交回 mini-swe-agent。

    这是刻意的格式与白名单守卫，不是语义正确性 evaluator。
    工具命令仍只由用户选择的环境执行；默认手动 runner 在显式启用本地
    执行之前使用记录环境。
    """

    def __init__(
        self,
        *,
        allowed_tools: tuple[str, ...] | None = None,
        require_tool_calls: bool | None = None,
        tool_policy: ToolPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        effective_tools = ("bash",) if allowed_tools is None else tuple(allowed_tools)
        if tool_policy is None:
            self._policy = ToolPolicy(
                allowed_tools=effective_tools,
                require_tool_calls=bool(require_tool_calls),
                rules=tuple(
                    ToolRule(
                        tool_name=name,
                        required_argument_paths=("command",) if name == "bash" else (),
                        non_empty_string_paths=("command",) if name == "bash" else (),
                    )
                    for name in effective_tools
                ),
            )
        elif isinstance(tool_policy, ToolPolicy):
            self._policy = tool_policy
            if allowed_tools is not None or require_tool_calls is not None:
                self._policy = replace(
                    self._policy,
                    allowed_tools=effective_tools
                    if allowed_tools is not None
                    else self._policy.allowed_tools,
                    require_tool_calls=(
                        self._policy.require_tool_calls
                        if require_tool_calls is None
                        else bool(require_tool_calls)
                    ),
                )
        else:
            self._policy = ToolPolicy.from_mapping(
                tool_policy,
                default_allowed_tools=effective_tools,
                default_require_tool_calls=bool(require_tool_calls),
            )

    def check(
        self,
        step: VerifierStep,
        *,
        candidate: ModelResponse,
        model_attempt_id: str,
        span_context: SpanIdentity | None = None,
    ) -> GuardResult:
        """执行一步验证；内部异常一律折叠为 ERROR 终态。

        ``span_context`` 只对**需要跨 span 历史的**子类有用（M4 失败记忆）；
        纯形状检查用不到它，收下即忽略，只为满足 `Verifier` 协议。
        """
        started = real_time()
        try:
            outcome = self._outcome(candidate)
        except Exception:
            outcome = GuardOutcome.ERROR
        return GuardResult(
            guard_ref=step.guard_ref,
            stage=step.stage,
            subject_model_attempt_id=model_attempt_id,
            outcome=outcome,
            latency_ms=max(0, round((real_time() - started) * 1000)),
            cost=0.0,
        )

    def _outcome(self, candidate: ModelResponse) -> GuardOutcome:
        """判定链：解析状态 -> 调用数量 -> 工具规则 -> 参数内容。"""
        if candidate.parse_status in (ParseStatus.INVALID, ParseStatus.UNSUPPORTED):
            return GuardOutcome.REJECT
        if self._policy.require_tool_calls and not candidate.tool_calls:
            return GuardOutcome.REJECT
        if self._policy.max_calls is not None and len(candidate.tool_calls) > self._policy.max_calls:
            return GuardOutcome.REJECT
        for call in candidate.tool_calls:
            rule = self._policy.rule_for(call.tool_name)
            if rule is None:
                return GuardOutcome.REJECT
            if not isinstance(call.arguments, Mapping):
                return GuardOutcome.REJECT
            if (
                rule.allowed_argument_keys is not None
                and set(call.arguments) - set(rule.allowed_argument_keys)
            ):
                return GuardOutcome.REJECT
            if any(_value_at_path(call.arguments, path) is None for path in rule.required_argument_paths):
                return GuardOutcome.REJECT
            if any(
                not isinstance(_value_at_path(call.arguments, path), str)
                or not _value_at_path(call.arguments, path).strip()
                for path in rule.non_empty_string_paths
            ):
                return GuardOutcome.REJECT
            if rule.reject_nul and _contains_nul(call.arguments):
                return GuardOutcome.REJECT
            if rule.max_argument_bytes is not None:
                try:
                    argument_bytes = len(
                        json.dumps(call.arguments, ensure_ascii=True, sort_keys=True).encode()
                    )
                except (TypeError, ValueError):
                    return GuardOutcome.REJECT
                if argument_bytes > rule.max_argument_bytes:
                    return GuardOutcome.REJECT
            # A rule with no required paths deliberately permits non-tool-call
            # payload shapes; the registry, not a universal ``command`` key,
            # defines what is valid for that tool.
        return GuardOutcome.PASS

    def _result_outcome(self, result: ToolExecutionResult) -> GuardOutcome:
        rule = self._policy.rule_for(result.tool_name)
        if rule is None:
            return GuardOutcome.REJECT
        if result.status not in rule.accepted_statuses:
            return GuardOutcome.REJECT
        if result.returncode not in rule.accepted_returncodes:
            return GuardOutcome.REJECT
        return GuardOutcome.PASS

    def check_tool_result(
        self,
        step: VerifierStep,
        *,
        result: ToolExecutionResult,
        model_attempt_id: str,
    ) -> GuardResult:
        """Validate a result using the result policy for its actual tool."""

        started = real_time()
        try:
            outcome = self._result_outcome(result)
        except Exception:
            outcome = GuardOutcome.ERROR
        return GuardResult(
            guard_ref=step.guard_ref,
            stage=step.stage,
            subject_model_attempt_id=model_attempt_id,
            subject_tool_call_id=result.tool_call_id,
            outcome=outcome,
            latency_ms=max(0, round((real_time() - started) * 1000)),
            cost=0.0,
        )


class M4FailureMemoryVerifier(BasicToolCallVerifier):
    """形状检查 + **M4 失败记忆**。只挂在 SLM 那条路上，大模型走 DIRECT 不过守卫。

    为什么是**继承**而不是替换
    --------------------------
    形状/白名单检查（工具名、参数键、必填、非空）是现成的、有用的，
    丢不掉。M4 是加在它**后面**的一道：先问「这个动作形状合法吗」，
    再问「这个动作在相似的场景下失败过吗」。

    判定顺序是契约的一部分
    ----------------------
    1. 形状不过 → 直接返回 REJECT，**短路，不查 M4**（形状都不合法，查历史没意义）
    2. 形状过了 → 调 ``lookup``；有命中 → REJECT
    3. 没命中 / 没配 ``lookup`` / ``lookup`` 抛异常 → PASS

    第 3 条里的最后一项是 **fail-open**：守卫不能因为记忆库坏了而卡住路由。
    这个库是「历史证据」，不是「正确性依赖」—— 读不到就等于没证据，放行。

    ``lookup`` 注入
    ---------------
    守卫只负责**顺序**，不负责去查什么。``lookup`` 收候选响应、返回命中的
    邻居（用真值判空即可，本类不看它的内容）。span 向量、记忆库句柄这些
    接线留在调用方，守卫保持可单测。
    """

    def __init__(
        self,
        *,
        allowed_tools: tuple[str, ...] | None = None,
        require_tool_calls: bool | None = None,
        tool_policy: ToolPolicy | Mapping[str, Any] | None = None,
        lookup: Callable[[ModelResponse, SpanIdentity | None], Sequence[Any]] | None = None,
        counters: MechanismCounters | None = None,
    ) -> None:
        super().__init__(
            allowed_tools=allowed_tools,
            require_tool_calls=require_tool_calls,
            tool_policy=tool_policy,
        )
        self._lookup = lookup
        #: T2 计数（spec §3.2）。默认空接收器 —— 不接线时零开销、行为不变。
        self._counters = counters if counters is not None else MechanismCounters.null()

    def check(
        self,
        step: VerifierStep,
        *,
        candidate: ModelResponse,
        model_attempt_id: str,
        span_context: SpanIdentity | None = None,
    ) -> GuardResult:
        """先形状、后记忆；任何一步不确定都放行。"""

        self._counters.bump("guard_checks")
        result = super().check(
            step,
            candidate=candidate,
            model_attempt_id=model_attempt_id,
            span_context=span_context,
        )
        if result.outcome != GuardOutcome.PASS:
            # ⚠️ **形状拒绝和记忆拒绝必须分开数。**
            # 形状拒绝多 ⇒ 小模型输出格式差；记忆拒绝多 ⇒ M4 记忆真在起作用。
            # 合成一个数，「M4 记忆这道值多少」就还是答不出来。
            self._counters.bump("guard_shape_rejects")
            return result
        if self._lookup is None:
            return result
        try:
            hits = self._lookup(candidate, span_context)
        except Exception:  # noqa: BLE001 - 记忆库坏了不能卡住路由
            # fail-open **不能**记成记忆拒绝 —— 库坏了不等于记忆拦下来了。
            return result
        if not hits:
            return result
        self._counters.bump("guard_memory_rejects")
        return GuardResult(
            guard_ref=result.guard_ref,
            stage=result.stage,
            subject_model_attempt_id=result.subject_model_attempt_id,
            outcome=GuardOutcome.REJECT,
            latency_ms=result.latency_ms,
            cost=result.cost,
        )


__all__ = [
    "BasicToolCallVerifier",
    "M4FailureMemoryVerifier",
    "ToolPolicy",
    "ToolRule",
]
