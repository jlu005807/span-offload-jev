"""响应层：模型响应相关对象。

由 Agent 模型集成（真实）/ FakeModelGateway（测试）产生，经 M3 存入
RequestResponseStore。accepted_output_ref 和 result_ref 仍控制访问权限，
但在线 trace 会保留完整响应正文。
"""

from __future__ import annotations

from pydantic import Field

from .base import (
    ContractModel,
    FinishReason,
    Identifier,
    JsonObject,
    JsonValue,
    ParseStatus,
    VersionedRef,
)
from .requests import ModelMessageToolCall


class ModelResponseToolCall(ModelMessageToolCall):
    """当前模型响应中供 TraceOffload 审计和路由的工具调用视图。"""


class ModelResponse(ContractModel):
    """一次 model_attempt 的实际模型返回。

    生产者：Agent 模型集成；消费者：M3（候选/接受判定）、Coordinator SpanRecord。
    能力引用控制谁可以解析响应，但不再剥离响应正文或 ``raw_response_ref``。
    """

    response_id: Identifier = Field(min_length=1, max_length=128)
    model_attempt_id: Identifier = Field(min_length=1, max_length=128)
    output_text: str | None = None
    reasoning_content: str | None = None
    structured_output: dict[str, JsonValue] | list[JsonValue] | None = None
    tool_calls: tuple[ModelResponseToolCall, ...] = ()
    finish_reason: FinishReason | Identifier
    parse_status: ParseStatus
    raw_response_ref: VersionedRef | None = None
    # Exact Agent-owned objects.  The integration returns these unchanged so
    # the wrapper never has to synthesize a second Agent message format.
    native_message: JsonObject | None = None
    native_error_message: JsonObject | None = None
    native_response: JsonValue | None = None

__all__ = [
    "ModelResponseToolCall",
    "ModelResponse",
]
