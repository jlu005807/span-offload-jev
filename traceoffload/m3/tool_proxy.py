"""M3 的工具事务边界。

Tool Proxy 是 agent framework 与真实环境之间唯一的控制点。它不理解
mini-swe-agent 的消息循环，只处理标准化 ``ToolAction``，并把完整环境结果
写入 ``ToolExecutionResult``。真实部署可以注入 sandbox、snapshot 或 rollback
实现；没有这些能力时，未知副作用不会被误判成安全操作。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..clock import real_time
from ..contracts import (
    ActualEffect,
    CommitStatus,
    EffectScope,
    ResumeStatus,
    RollbackStatus,
    ToolExecutionResult,
    ToolExecutionStatus,
    VersionedRef,
)
from ..interfaces import ToolAction, ToolProxyResult


def _effect(scope: EffectScope) -> ActualEffect:
    if scope == EffectScope.READ_ONLY:
        return ActualEffect.READ_ONLY
    if scope in (EffectScope.REVERSIBLE_WRITE, EffectScope.IRREVERSIBLE_WRITE):
        return ActualEffect.WRITE
    return ActualEffect.UNKNOWN


def _safe_id(value: str, *, default: str = "unknown") -> str:
    """把 provider/exception 名称压成契约允许的 Identifier。"""

    text = "".join(char if (char.isalnum() or char in "._:-") else "-" for char in value)
    text = text.strip("-")
    return text[:128] or default


def _summary(output: Mapping[str, Any] | Any) -> str | None:
    """返回便于检索的结果摘要；完整结果同时保存在 ``raw_output``。"""

    if output is None:
        return "output=none"
    if isinstance(output, Mapping):
        value = output.get("output")
        exception = output.get("exception_info")
        returncode = output.get("returncode")
        length = len(value) if isinstance(value, str) else len(str(value or ""))
        return (
            f"output_chars={length}; returncode={returncode!s}; "
            f"exception={'present' if exception else 'none'}"
        )
    else:
        value = output
    return f"output_chars={len(value) if isinstance(value, str) else len(str(value))}"


def _as_output(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"output": value, "returncode": 0, "exception_info": ""}


def _jsonable(value: Any) -> Any:
    """保留完整结果，同时把非 JSON 原生值转换成可写入 trace 的表示。"""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


def _raw_output(output: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in output.items()}


class CallableToolProxy:
    """用回调包装现有 Environment 的通用 Tool Proxy。

    参数回调都位于 adapter/coordinator 装配层，因此不需要改动原始 agent
    源码。真实 SLM 路径由 ``begin_transaction`` 建立回滚边界，随后调用
    ``execute``。
    """

    def __init__(
        self,
        *,
        execute: Callable[[Mapping[str, Any], str], Any],
        effect_scope: Callable[[ToolAction], EffectScope] | None = None,
        begin_transaction: Callable[[str, ToolAction], bool | None] | None = None,
        commit: Callable[[str], bool | None] | None = None,
        rollback: Callable[[str], RollbackStatus | str] | None = None,
        resume_check: Callable[[str], ResumeStatus | str] | None = None,
        close_span: Callable[[object], bool | None] | None = None,
    ) -> None:
        self._execute = execute
        self._effect_scope = effect_scope or (lambda action: EffectScope.UNKNOWN)
        self._begin_transaction_callback = begin_transaction
        self._commit_callback = commit
        self._rollback_callback = rollback
        self._resume_callback = resume_check
        self._close_span_callback = close_span
        self._transactions: dict[str, ActualEffect] = {}
        self._transaction_aliases: dict[str, str] = {}
        # Guarded SLM actions are authorized before the Agent can see them.
        # Keep the binding here as a second enforcement point at execution
        # time, so a framework callback cannot silently replace tool args.
        self._authorized_actions: dict[str, ToolAction] = {}
        self._authorized_families: dict[str, str] = {}
        self._executed_transactions: set[str] = set()
        self._last_lifecycle_fault: str | None = None

    def _scope(self, action: ToolAction) -> EffectScope:
        try:
            value = self._effect_scope(action)
            return value if isinstance(value, EffectScope) else EffectScope(str(value))
        except Exception:
            return EffectScope.UNKNOWN

    def effect_scope(self, action: ToolAction) -> EffectScope:
        """Expose the adapter's pre-execution effect declaration to M3."""

        return self._scope(action)

    def close_span(self, request) -> bool:
        """Episode 结束清理；无 checkpoint 后端时为幂等 no-op。

        callback 返回 None 视为成功（旧实现/缺省），只有明确的 False 视为失败。
        """
        if self._close_span_callback is None:
            return True
        return self._close_span_callback(request) is not False

    def can_rollback(self, action: ToolAction) -> bool:
        """Whether M3 may expose a mutating action to the guarded SLM path."""

        scope = self._scope(action)
        if scope in (EffectScope.READ_ONLY,):
            return True
        return (
            scope == EffectScope.REVERSIBLE_WRITE
            and self._begin_transaction_callback is not None
            and self._rollback_callback is not None
        )

    def begin_transaction(self, action: ToolAction, *, transaction_id: str) -> bool:
        """Create the real-environment rollback boundary before execution.

        Both checkpoint creation and rollback must be explicit.  A rollback
        callback alone cannot prove that a pre-execution restoration point
        exists and therefore is never enough for guarded admission.
        """

        scope = self._scope(action)
        if scope in (EffectScope.IRREVERSIBLE_WRITE, EffectScope.UNKNOWN):
            return False
        if scope == EffectScope.READ_ONLY:
            self._transactions.setdefault(transaction_id, ActualEffect.READ_ONLY)
            self._transaction_aliases[action.tool_call_id] = transaction_id
            self._bind_authorized_action(transaction_id, action)
            return True
        if self._begin_transaction_callback is None or self._rollback_callback is None:
            self._last_lifecycle_fault = "begin_callback_missing"
            return False
        try:
            accepted = self._begin_transaction_callback(transaction_id, action)
            if accepted is False:
                self._last_lifecycle_fault = "begin_callback_refused"
                return False
        except Exception as error:
            self._last_lifecycle_fault = f"begin_callback_error:{type(error).__name__}"
            return False
        self._transactions[transaction_id] = ActualEffect.WRITE
        self._transaction_aliases[action.tool_call_id] = transaction_id
        self._bind_authorized_action(transaction_id, action)
        return True

    def commit(self, transaction_id: str) -> bool:
        """Finalize a write transaction; release is owned by ``close_span``."""

        actual_id = self._resolve_transaction_id(transaction_id)
        effect = self._transactions.get(actual_id, ActualEffect.UNKNOWN)
        if effect not in (ActualEffect.NONE, ActualEffect.READ_ONLY) and self._commit_callback is None:
            self._last_lifecycle_fault = "commit_callback_missing"
            return False
        try:
            if self._commit_callback is not None:
                committed = self._commit_callback(actual_id)
                if committed is False:
                    self._last_lifecycle_fault = "commit_callback_refused"
                    return False
        except Exception as error:
            self._last_lifecycle_fault = f"commit_callback_error:{type(error).__name__}"
            return False
        # Do not forget a write handle here.  The episode-level close/release
        # operation is the only point allowed to discard it.
        if effect in (ActualEffect.NONE, ActualEffect.READ_ONLY):
            self._forget_transaction(actual_id)
        return True

    def release_transaction(self, transaction_id: str) -> bool:
        """Drop an authorization handle after a read-only/rejected action."""

        self._forget_transaction(self._resolve_transaction_id(transaction_id))
        return True

    def finish_recovery(self, transaction_ids: tuple[str, ...]) -> bool:
        """Release handles retained while a workflow recovery lock was active."""

        for transaction_id in transaction_ids:
            self.release_transaction(transaction_id)
        return True

    def execute(self, action: ToolAction, *, transaction_id: str) -> ToolProxyResult:
        return self._execute_with(
            action,
            transaction_id=transaction_id,
            execute=lambda: self._execute(action.arguments, action.cwd),
        )

    def _execute_with(
        self,
        action: ToolAction,
        *,
        transaction_id: str,
        execute: Callable[[], Any],
    ) -> ToolProxyResult:
        started = real_time()
        actual_id = self._resolve_transaction_id(transaction_id)
        expected = self._authorized_actions.get(actual_id)
        guarded_transaction = expected is not None
        if expected is None and self._is_authorized_family(transaction_id):
            return self._rejected_result(
                action,
                transaction_id=transaction_id,
                error_type="transaction-not-authorized",
            )
        if expected is not None:
            if not _same_tool_action(expected, action):
                return self._rejected_result(
                    action,
                    transaction_id=transaction_id,
                    error_type="tool-action-mismatch",
                )
            if actual_id in self._executed_transactions:
                return self._rejected_result(
                    action,
                    transaction_id=transaction_id,
                    error_type="duplicate-tool-execution",
                )
            self._executed_transactions.add(actual_id)
        scope = self._scope(action)
        effect = _effect(scope)
        self._transactions[actual_id] = effect
        self._transaction_aliases[action.tool_call_id] = actual_id
        try:
            output = _as_output(execute())
        except BaseException as error:
            # mini-swe-agent uses Submitted as a control-flow exception. It must
            # still reach the original Agent loop and is not a tool failure.
            if type(error).__name__ in {"Submitted", "Exit"}:
                raise
            # 工具基础设施异常（阶段 2 修正）：confirmed=False 让收据进入
            # missing/untrusted 回滚分支；fault_class 显式标 env_fault，不再靠
            # 异常类名猜测。error_type 仅留作审计，不参与故障判定。
            output = {"output": "", "returncode": -1, "exception_info": str(error)}
            record = ToolExecutionResult(
                transaction_id=transaction_id,
                tool_call_id=action.tool_call_id,
                tool_name=_safe_id(action.tool_name),
                status=ToolExecutionStatus.FAILED,
                effect=effect,
                commit_status=(
                    CommitStatus.NOT_ATTEMPTED
                    if effect in (ActualEffect.NONE, ActualEffect.READ_ONLY)
                    else CommitStatus.UNKNOWN
                ),
                returncode=-1,
                latency_ms=max(0, round((real_time() - started) * 1000)),
                output_summary=_summary(output),
                raw_output=_raw_output(output),
                result_ref=VersionedRef(
                    kind="tool_result",
                    id=f"real-{_safe_id(action.tool_call_id)}",
                    version="v1",
                ),
                error_type=_safe_id(type(error).__name__),
                fault_class="env_fault",
            )
            return ToolProxyResult(record=record, output=output, confirmed=False)
        returncode = _returncode(output)
        status = ToolExecutionStatus.SUCCESS if returncode in (None, 0) else ToolExecutionStatus.FAILED
        # 命令结果观察（阶段 2）：returncode 是正确性信号，不参与回滚判定；
        # 127 = shell 的 command_not_found。执行层异常分支不设置（不可确认）。
        command_status = (
            "exit_0"
            if returncode in (None, 0)
            else "command_not_found"
            if returncode == 127
            else "nonzero"
        )
        commit = (
            CommitStatus.NOT_ATTEMPTED
            if effect in (ActualEffect.NONE, ActualEffect.READ_ONLY)
            else CommitStatus.NOT_ATTEMPTED
            if guarded_transaction and status == ToolExecutionStatus.SUCCESS
            else CommitStatus.COMMITTED
            if status == ToolExecutionStatus.SUCCESS
            else CommitStatus.PARTIAL
        )
        record = ToolExecutionResult(
            transaction_id=transaction_id,
            tool_call_id=action.tool_call_id,
            tool_name=_safe_id(action.tool_name),
            status=status,
            effect=effect,
            commit_status=commit,
            returncode=returncode,
            latency_ms=max(0, round((real_time() - started) * 1000)),
            output_summary=_summary(output),
            raw_output=_raw_output(output),
            result_ref=VersionedRef(
                kind="tool_result",
                id=f"real-{_safe_id(action.tool_call_id)}",
                version="v1",
            ),
            error_type=("tool-nonzero" if status != ToolExecutionStatus.SUCCESS else None),
            command_status=command_status,
        )
        return ToolProxyResult(record=record, output=output)

    def rollback(self, transaction_id: str) -> RollbackStatus:
        transaction_id = self._resolve_transaction_id(transaction_id)
        effect = self._transactions.get(transaction_id, ActualEffect.UNKNOWN)
        if effect in (ActualEffect.NONE, ActualEffect.READ_ONLY):
            self._forget_transaction(transaction_id)
            return RollbackStatus.NOT_NEEDED
        if self._rollback_callback is not None:
            try:
                status = _rollback_value(self._rollback_callback(transaction_id))
            except Exception as error:
                self._last_lifecycle_fault = f"rollback_callback_error:{type(error).__name__}"
                return RollbackStatus.UNKNOWN
            if status == RollbackStatus.SUCCEEDED:
                self._forget_transaction(transaction_id)
            else:
                self._last_lifecycle_fault = "rollback_callback_refused"
            return status
        self._last_lifecycle_fault = "rollback_callback_missing"
        return RollbackStatus.UNKNOWN

    def consume_lifecycle_fault(self) -> str | None:
        fault = self._last_lifecycle_fault
        self._last_lifecycle_fault = None
        return fault

    def _bind_authorized_action(self, transaction_id: str, action: ToolAction) -> None:
        self._authorized_actions[transaction_id] = ToolAction(
            tool_call_id=action.tool_call_id,
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            cwd=action.cwd,
        )
        self._authorized_families[transaction_id] = _transaction_family(
            transaction_id,
            action.tool_call_id,
        )
        self._executed_transactions.discard(transaction_id)

    def _forget_transaction(self, transaction_id: str) -> None:
        self._transactions.pop(transaction_id, None)
        self._authorized_actions.pop(transaction_id, None)
        self._authorized_families.pop(transaction_id, None)
        self._executed_transactions.discard(transaction_id)
        for tool_call_id, actual_id in tuple(self._transaction_aliases.items()):
            if actual_id == transaction_id:
                self._transaction_aliases.pop(tool_call_id, None)

    def _rejected_result(
        self,
        action: ToolAction,
        *,
        transaction_id: str,
        error_type: str,
    ) -> ToolProxyResult:
        output = {
            "output": "",
            "returncode": -1,
            "exception_info": error_type,
        }
        record = ToolExecutionResult(
            transaction_id=transaction_id,
            tool_call_id=_safe_id(action.tool_call_id),
            tool_name=_safe_id(action.tool_name),
            status=ToolExecutionStatus.REJECTED,
            effect=ActualEffect.UNKNOWN,
            commit_status=CommitStatus.BLOCKED,
            returncode=-1,
            output_summary=_summary(output),
            raw_output=output,
            result_ref=VersionedRef(
                kind="tool_result",
                id=f"rejected-{_safe_id(action.tool_call_id)}",
                version="v1",
            ),
            error_type=error_type,
        )
        return ToolProxyResult(record=record, output=output)

    def resume_check(self, transaction_id: str) -> ResumeStatus:
        transaction_id = self._resolve_transaction_id(transaction_id)
        if self._resume_callback is not None:
            return _resume_value(self._resume_callback(transaction_id))
        effect = self._transactions.get(transaction_id, ActualEffect.UNKNOWN)
        if effect in (ActualEffect.NONE, ActualEffect.READ_ONLY):
            return ResumeStatus.NOT_NEEDED
        return ResumeStatus.UNKNOWN

    def _resolve_transaction_id(self, transaction_id: str) -> str:
        if transaction_id in self._transactions:
            return transaction_id
        # Coordinator may canonicalize an adapter transaction ID after the
        # environment callback; preserve the tool-call lineage for recovery.
        for tool_call_id, actual_id in self._transaction_aliases.items():
            if transaction_id.endswith(f"-{tool_call_id}"):
                requested_family = _transaction_family(transaction_id, tool_call_id)
                actual_family = _transaction_family(actual_id, tool_call_id)
                # Only resolve aliases within the same execution request. A
                # recovery/direct episode may reuse a tool_call_id while
                # owning a new transaction namespace.
                if requested_family == actual_family or not transaction_id.startswith(
                    "transaction-"
                ):
                    return actual_id
        return transaction_id

    def _is_authorized_family(self, transaction_id: str) -> bool:
        return any(
            transaction_id == authorized_family
            or transaction_id.startswith(f"{authorized_family}-")
            for authorized_family in self._authorized_families.values()
        )


__all__ = ["CallableToolProxy"]
def _returncode(output: Mapping[str, Any]) -> int | None:
    value = output.get("returncode")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _same_tool_action(left: ToolAction, right: ToolAction) -> bool:
    """Compare the model-authorized tool fields; adapter cwd may be normalized."""

    return left.tool_name == right.tool_name and dict(left.arguments) == dict(right.arguments)


def _transaction_family(transaction_id: str, tool_call_id: str) -> str:
    """Return the canonical execution-request prefix for an action ID."""

    suffix = f"-{tool_call_id}" if tool_call_id else ""
    if suffix and transaction_id.endswith(suffix):
        return transaction_id[: -len(suffix)]
    return transaction_id


def _rollback_value(value: RollbackStatus | str) -> RollbackStatus:
    try:
        return value if isinstance(value, RollbackStatus) else RollbackStatus(str(value))
    except ValueError:
        return RollbackStatus.UNKNOWN


def _resume_value(value: ResumeStatus | str) -> ResumeStatus:
    try:
        return value if isinstance(value, ResumeStatus) else ResumeStatus(str(value))
    except ValueError:
        return ResumeStatus.UNKNOWN

