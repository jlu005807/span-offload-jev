"""M1 运行时画像：为一次调用构建执行前描述。

``RuntimeProfiler.build_profile`` 只从不可变的 ``InvocationContext`` 与已解析的
``ModelRequest`` 推导画像。从这些输入无法观察的安全相关字段写成 ``unknown``
并记入 ``uncertain_fields``；unknown 值永远不会被降级为低风险值。
冻结 pilot 不使用先验反馈快照（无 learned prior），因此画像不依赖它们。
"""

from __future__ import annotations

import re
import json
from typing import Any, Literal
from collections.abc import Sequence

from ..contracts import (
    Component,
    DownstreamDependency,
    EffectRisk,
    Identifier,
    InvocationContext,
    MessageRole,
    ModelRequest,
    OutputContract,
    OutputKind,
    ProducerRef,
    RecoveryLevel,
    RuntimeProfile,
    AllowedTargets,
    ModelRef,
    ModelTarget,
    ModelTargetKind,
    SpanDescriptor,
    UncertaintyReason,
    UncertainField,
    WorkflowContext,
)

# descriptor schema 版本：进入 descriptor_key，schema 变更即换新键
DESCRIPTOR_SCHEMA_VERSION = "2"

# last_observation_class 的判定正则（阶段 4）：纯文本特征，非语义分类；
# 按优先级 test_feedback > error_trace > code_view > plain 取最高档。
_OBS_TEST_FEEDBACK = re.compile(r"\b\d+\s+passed\b|\bFAILED\b|\bERROR\b|\bOK\b|Ran\s+\d+\s+tests?")
_OBS_ERROR_TRACE = re.compile(r"Traceback|Error:|Exception|raise |Failed to")
_OBS_CODE_VIEW = re.compile(r"^\s*(def |class |import |from )|/testbed/.*\.py", re.M)


class RuntimeProfiler:
    """冻结 pilot 的确定性 M1 画像器（纯函数，输入不可变）。"""

    def __init__(self, *, allowed_targets: AllowedTargets | None = None) -> None:
        # 只有 adapter/config 明确提供数据策略时才填充；默认 unknown，
        # 避免从黑盒 framework metadata 猜测数据是否可发送给 SLM。
        self._allowed_targets = allowed_targets

    def build_profile(
        self,
        context: InvocationContext,
        request: ModelRequest,
    ) -> RuntimeProfile:
        workflow = context.workflow_context
        descriptor = SpanDescriptor(
            subtask_type=_classify_subtask(request, workflow),
            estimated_input_tokens=_estimate_input_tokens(request),
            output_contract=workflow.output_contract,
            allowed_targets=getattr(self, "_allowed_targets", None) or "unknown",
            downstream_dependency=workflow.downstream_contract.dependency_level,
            effect_risk=workflow.state_context.expected_effect,
            available_verifiers=workflow.verifier_catalog,
            recovery_level=workflow.recovery_capability,
        )
        return RuntimeProfile(
            schema_version=context.schema_version,
            message_id=f"message-profile-{context.episode_id}",
            producer_ref=ProducerRef(component=Component.M1, version="0.1.0"),
            produced_at=context.produced_at,
            workflow_run_id=context.workflow_run_id,
            workflow_plan_ref=context.workflow_plan_ref,
            workflow_plan_revision=context.workflow_plan_revision,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            request_ref=context.request_ref,
            descriptor_key=_descriptor_key(workflow, request, descriptor),
            descriptor=descriptor,
            uncertain_fields=_uncertain_fields(descriptor),
            last_observation_class=_classify_last_observation(request),
        )


def _estimate_input_tokens(request: ModelRequest) -> int:
    """输入 token 估计：总字符数 // 4 的确定性启发式（真实 token 数由 attempt 记录）。"""
    total_chars = sum(_content_length(message.content) for message in request.messages) + len(
        request.input_text or ""
    )
    return max(0, total_chars // 4)


def _classify_subtask(
    request: ModelRequest,
    workflow: WorkflowContext,
) -> Identifier | Literal["unknown"]:
    """从可观察消息边界做保守的在线子任务分类。

    这不是语义分类器：它只区分当前请求是否在解释刚收到的工具观察，
    或在生成下一次工具参数。无法由协议事实确认的语义仍保持 unknown。
    """
    if workflow.recovery_mode == "rollback_recovery":
        return "rollback_recovery"
    if request.messages and request.messages[-1].role == MessageRole.TOOL:
        return "tool_result_interpretation"
    if request.tool_definitions or workflow.available_tools:
        return "tool_argument_generation"
    contract = workflow.output_contract
    if isinstance(contract, OutputContract):
        return {
            OutputKind.JSON: "structured_generation",
            OutputKind.CODE_PATCH: "code_patch_generation",
            OutputKind.SQL: "sql_generation",
            OutputKind.PLAN: "plan_generation",
        }.get(contract.output_kind, "unknown")
    return "unknown"


_ObservationClass = Literal["test_feedback", "error_trace", "code_view", "plain", "unknown"]


def _classify_last_observation(request: ModelRequest) -> _ObservationClass:
    """判定最后一条工具观察的文本特征档（阶段 4 路由特征）。

    纯确定性文本特征，非语义分类；同一观察按优先级
    test_feedback > error_trace > code_view > plain 取最高档。
    请求中没有工具消息时为 unknown（本请求不含观察）。
    """
    for message in reversed(request.messages):
        if message.role != MessageRole.TOOL:
            continue
        content = _content_for_classification(message.content)
        if _OBS_TEST_FEEDBACK.search(content):
            return "test_feedback"
        if _OBS_ERROR_TRACE.search(content):
            return "error_trace"
        if _OBS_CODE_VIEW.search(content):
            return "code_view"
        return "plain"
    return "unknown"


def _content_length(content: str | list[object] | None) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=True, separators=(",", ":")))


def _content_for_classification(content: str | list[object] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    fragments: list[str] = []
    for item in content:
        if isinstance(item, str):
            fragments.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            fragments.append(item["text"])
    return "".join(fragments)


def _descriptor_key(
    workflow: WorkflowContext,
    request: ModelRequest,
    descriptor: SpanDescriptor,
) -> str:
    """规范键：含可影响路由/经验统计的可观察 descriptor 字段。"""
    output_version = (
        descriptor.output_contract.version
        if isinstance(descriptor.output_contract, OutputContract)
        else "none"
    )
    tool_refs = list(workflow.available_tools)
    tool_refs.extend(definition.schema_ref for definition in request.tool_definitions)
    tool_versions = sorted({f"{ref.kind}:{ref.version}" for ref in tool_refs})
    tool_key = ",".join(tool_versions) if tool_versions else "none"
    verifier_key = ",".join(
        sorted(
            f"{guard.kind}:{guard.id}:{guard.version}"
            for guard in descriptor.available_verifiers
        )
    ) or "none"
    return "|".join(
        (
            f"descriptor/v{DESCRIPTOR_SCHEMA_VERSION}",
            f"subtask:{descriptor.subtask_type}",
            f"output:{output_version}",
            f"tools:{tool_key}",
            f"risk:{descriptor.effect_risk.value}",
            f"downstream:{descriptor.downstream_dependency.value}",
            f"verifiers:{verifier_key}",
            f"recovery:{descriptor.recovery_level.value}",
        )
    )


def _uncertain_fields(descriptor: SpanDescriptor) -> tuple[UncertainField, ...]:
    """descriptor 中为 unknown 的字段逐一记入 uncertain_fields（reason=MISSING）。"""
    fields: list[UncertainField] = []
    if descriptor.subtask_type == "unknown":
        fields.append(
            UncertainField(field_path="descriptor.subtask_type", reason=UncertaintyReason.MISSING)
        )
    if descriptor.allowed_targets == "unknown":
        fields.append(
            UncertainField(field_path="descriptor.allowed_targets", reason=UncertaintyReason.MISSING)
        )
    if descriptor.downstream_dependency == DownstreamDependency.UNKNOWN:
        fields.append(
            UncertainField(
                field_path="descriptor.downstream_dependency",
                reason=UncertaintyReason.MISSING,
            )
        )
    if descriptor.effect_risk == EffectRisk.UNKNOWN:
        fields.append(
            UncertainField(field_path="descriptor.effect_risk", reason=UncertaintyReason.MISSING)
        )
    if descriptor.recovery_level == RecoveryLevel.UNKNOWN:
        fields.append(
            UncertainField(field_path="descriptor.recovery_level", reason=UncertaintyReason.MISSING)
        )
    return tuple(fields)


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "RuntimeProfiler",
]

class ClassifierFeatureProfiler:
    """在 RuntimeProfiler 外面包一层，把在线特征塞进 RuntimeProfile。

    M1 本身是单遍的；"两阶段"只出现在这层包装里：先基础 profile，再补特征。
    只在 classifier_route 模式下使用。
    """

    def __init__(self, base: Any, builder: Any) -> None:
        self._base = base
        self._builder = builder

    def build_profile(
        self,
        context: InvocationContext,
        request: ModelRequest,
    ) -> RuntimeProfile:
        profile = self._base.build_profile(context, request)
        features = self._builder.build(context, request, profile)
        return profile.model_copy(
            update={
                "online_features": features,
                "feature_version": self._builder.feature_version,
            }
        )

