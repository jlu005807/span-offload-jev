"""TraceOffload 组件接缝的 Protocol 定义。

每个跨组件边界都使用 ``traceoffload.contracts`` 中的契约模型；这些 Protocol
只声明哪些契约对象流入/流出每个组件，使 fakes、真实 M1-M4 实现与
Coordinator 可以互相替换。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import AwareDatetime

from .contracts import (
    Attribution,
    ExecutionRequest,
    ExecutionResult,
    FeedbackSnapshot,
    GuardResult,
    InvocationContext,
    ModelRequest,
    ModelResponse,
    ModelTarget,
    OutcomeEvent,
    RouteOption,
    RoutePlan,
    RoutePolicy,
    RuntimeProfile,
    SpanRecord,
    ToolExecutionResult,
    EffectRisk,
    RecoveryLevel,
    ResumeStatus,
    RollbackStatus,
    StateAccessMode,
    VerifierStep,
)


class GatewayError(RuntimeError):
    """模型网关调用在提供方或传输层失败。"""


class GatewayTimeout(GatewayError):
    """模型网关调用超过期限被中止。"""


@dataclass(frozen=True)
class ModelCallResult:
    """一次物理模型完成及其记录指标。

    响应正文、token 数、时延与成本会成为 ExecutionResult 中的一条
    ModelAttempt，并可随完整 SpanRecord 持久化。
    """

    response: ModelResponse
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    cost: float | None


@dataclass(frozen=True)
class ToolAction:
    """Agent adapter 传给 Tool Proxy 的最小、可审计动作。"""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    cwd: str = ""


@dataclass(frozen=True)
class ToolProxyResult:
    """Tool Proxy 的原始输出与完整事务记录。

    ``confirmed`` distinguishes a result emitted by the real Tool Proxy from
    a framework-only observation fallback.  Guarded SLM may commit only the
    former; an unconfirmed observation is retained for audit and forces the
    conservative rollback path.
    """

    record: ToolExecutionResult
    output: Mapping[str, Any]
    confirmed: bool = True


class ToolProxy(Protocol):
    """M3 统一使用的真实工具、事务和恢复边界。"""

    def execute(self, action: ToolAction, *, transaction_id: str) -> ToolProxyResult: ...

    def effect_scope(self, action: ToolAction) -> EffectScope: ...

    def can_rollback(self, action: ToolAction) -> bool: ...

    def begin_transaction(self, action: ToolAction, *, transaction_id: str) -> bool: ...

    def commit(self, transaction_id: str) -> bool: ...

    def rollback(self, transaction_id: str) -> RollbackStatus: ...

    def resume_check(self, transaction_id: str) -> ResumeStatus: ...

    def release_transaction(self, transaction_id: str) -> bool: ...

    def finish_recovery(self, transaction_ids: tuple[str, ...]) -> bool: ...

    def consume_lifecycle_fault(self) -> str | None: ...


class CodeAgentAdapter(Protocol):
    """One integration object for a code-agent framework and its environment.

    A concrete adapter supplies a framework-specific model/environment hook and
    one ToolProxy.  When its environment has snapshot capabilities, that proxy
    is a real span sandbox; otherwise the adapter reports conservative runtime
    facts so M2 can force DIRECT execution.
    """

    adapter_id: str

    @property
    def tool_proxy(self) -> ToolProxy: ...

    @property
    def state_access_mode(self) -> StateAccessMode: ...

    @property
    def expected_effect(self) -> EffectRisk: ...

    @property
    def recovery_capability(self) -> RecoveryLevel: ...

    @property
    def sandbox_capabilities(self) -> Mapping[str, Any]: ...

    def framework_metadata(self) -> Mapping[str, Any]: ...


class ModelGateway(Protocol):
    """Run one completion through the target Agent integration.

    The core routes and records the call. Serialization, transport, retries,
    response parsing and native errors remain owned by the integration.
    """

    def complete(
        self,
        target: ModelTarget,
        request: ModelRequest,
        *,
        model_attempt_id: str,
        deadline_at: AwareDatetime,
    ) -> ModelCallResult: ...


class Profiler(Protocol):
    """M1：为一次调用构建执行前运行时画像。"""

    def build_profile(
        self,
        context: InvocationContext,
        request: ModelRequest,
    ) -> RuntimeProfile: ...


class Router(Protocol):
    """M2：在路由策略下选择不可变执行方案。

    ``admission_facts`` 是路由前可得的预算事实（截止余量/剩余步数）；
    仅生产路由器消费，旧调用者与测试可不传。
    """

    def decide(
        self,
        profile: RuntimeProfile,
        options: Sequence[RouteOption],
        policy: RoutePolicy,
        snapshot: FeedbackSnapshot | None,
        *,
        admission_facts: object | None = None,
    ) -> RoutePlan: ...


class Executor(Protocol):
    """M3：把路由方案运行到终态执行结果。"""

    def execute_step(self, request: ExecutionRequest) -> ExecutionResult: ...


@dataclass(frozen=True)
class SpanIdentity:
    """一个 span 的身份三元组：``(workflow_run_id, episode_id, logical_invocation_id)``。

    为什么需要它：``ModelResponse`` 里**没有任何 span 标识**（只有 ``response_id``
    / ``model_attempt_id`` / ``tool_calls`` / …）。所以一个想在「当前 span」上
    查历史的 verifier（比如 M4 失败记忆）没有任何办法自己找到自己 ——
    身份只能由 executor 从 ``ExecutionRequest.route_plan`` 里取出来、**显式传下去**。

    三个 id 与 ``RoutePlan`` 上同名，语义也一致。

    ⚠️ ``episode_id`` 单独用**不够**：它只在一次 episode 内唯一，而 M4 的记忆
    是跨 episode 的。``_m4_record_key`` 用的是 ``(workflow_run_id,
    logical_invocation_id)`` 这一对 —— 见 `runner._m4_record_key`。
    """

    workflow_run_id: str
    episode_id: str
    logical_invocation_id: str


class Verifier(Protocol):
    """对一个候选模型响应执行一步验证。"""

    def check(
        self,
        step: VerifierStep,
        *,
        candidate: ModelResponse,
        model_attempt_id: str,
        span_context: SpanIdentity | None = None,
    ) -> GuardResult:
        """对一个候选响应跑一步验证。

        ``span_context`` 是**当前 span 的身份**，由 M3 从
        ``ExecutionRequest.route_plan`` 取出后传入。``ModelResponse`` 里没有
        span 标识，所以需要在「当前 span」上查历史的实现（M4 失败记忆）
        只能靠这个参数找到自己。不需要它的实现收下即忽略。
        """
        ...


class Evaluator(Protocol):
    """为完成的执行产生延迟的、带修订的结果标签。"""

    def evaluate_span(
        self,
        record: SpanRecord,
        *,
        accepted_attempt_id: str | None,
    ) -> tuple[OutcomeEvent, ...]: ...

    def evaluate_workflow_final(
        self,
        records: Sequence[SpanRecord],
        *,
        attribution: Sequence[Attribution] | None = None,
    ) -> OutcomeEvent: ...


class ProbeBudgetLedger(Protocol):
    """持久化探针预算台账：跨 workflow/进程累计探针消耗（文档 A7）。

    预算按 experiment_id 作用域隔离：新实验用新 experiment_id；同实验重试才允许
    decision_id 幂等。decision_id 全局唯一——已存在但 experiment/workflow 不一致
    时 try_consume 返回 False（不能幂等成功，也不能重复扣减）。
    """

    def workflow_used(self, experiment_id: str, workflow_run_id: str) -> int: ...

    def total_used(self, experiment_id: str) -> int: ...

    def try_consume(
        self,
        experiment_id: str,
        workflow_run_id: str,
        decision_id: str,
        reason: str,
        *,
        workflow_limit: int | None,
        total_limit: int | None,
    ) -> bool: ...


class FeedbackStore(Protocol):
    """M4：从存储的证据发布新的版本化反馈快照。"""

    def publish_snapshot(self) -> FeedbackSnapshot: ...


__all__ = [
    "Evaluator",
    "CodeAgentAdapter",
    "Executor",
    "FeedbackStore",
    "GatewayError",
    "GatewayTimeout",
    "ModelCallResult",
    "ToolAction",
    "ToolProxy",
    "ToolProxyResult",
    "ModelGateway",
    "Profiler",
    "Router",
    "Verifier",
]
