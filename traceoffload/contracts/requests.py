"""请求层：Agent 发出的模型请求相关对象。

由 integrations 的 wrapper 从 mini-swe-agent 消息转换而来，经 Coordinator
存入 RequestResponseStore，供 M1（画像视图）与 M3（执行视图）解析；实际在线
SpanRecord 同时保留完整请求，能力引用只控制访问权限。
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import (
    ContractModel,
    EffectScope,
    Identifier,
    JsonObject,
    JsonValue,
    MessageRole,
    VersionedRef,
    VersionId,
)


class ModelMessageToolCall(ContractModel):
    """历史 assistant 工具调用，保留在后续模型请求中。

    用途：OpenAI 兼容 API 只有在前面 assistant 消息保留对应 call ID 与参数时，
    tool-result 消息才有意义。把它放进共享请求契约，防止适配器在两个
    LM span 之间丢失该关联（跨轮 tool call ID 保留）。
    """

    tool_call_id: Identifier = Field(min_length=1, max_length=128)
    tool_name: Identifier = Field(min_length=1, max_length=128)
    arguments: JsonObject
    # Keep the provider's JSON text for exact trace inspection. Transport uses
    # ModelRequest.native_payload and never reconstructs history from this view.
    raw_arguments: str | None = None
    # A malformed argument is handled by the agent's FormatError loop.  It is
    # intentionally represented in the response contract rather than raised by
    # the transport gateway.
    argument_error: str | None = None


class ModelMessage(ContractModel):
    """模型请求中的一条消息。

    role 为 tool 时通过 tool_call_id 关联历史调用；role 为 assistant 时
    通过 tool_calls 回放它发过的工具调用（多轮对话必需）。
    """

    role: MessageRole
    # OpenAI-compatible providers use ``content: null`` for assistant tool
    # calls. Keep that distinction in the normalized inspection view too.
    content: str | list[JsonValue] | None
    # Thinking providers may return this alongside an assistant message.
    reasoning_content: str | None = None
    name: Identifier | None = Field(default=None, min_length=1, max_length=128)
    tool_call_id: Identifier | None = Field(default=None, min_length=1, max_length=128)
    tool_calls: tuple[ModelMessageToolCall, ...] = ()
    @model_validator(mode="after")
    def assistant_metadata_is_scoped(self) -> ModelMessage:
        if self.role != MessageRole.ASSISTANT and self.reasoning_content is not None:
            raise ValueError("reasoning_content is only valid on assistant messages")
        return self


class Attachment(ContractModel):
    """附件：通过版本化引用指向实际内容，不复制正文。"""

    kind: Identifier = Field(min_length=1, max_length=64)
    ref: VersionedRef
    mime_type: str | None = Field(default=None, min_length=1, max_length=128)
    summary: str | None = None


class ToolDefinition(ContractModel):
    """工具定义：工具名 + schema 引用 + 副作用范围（事前声明）。

    用途：传给 M1 做子任务画像、供 Agent 集成描述工具协议、
    声明工具的副作用范围供 M2 风险评估。
    """

    tool_name: Identifier = Field(min_length=1, max_length=128)
    schema_ref: VersionedRef
    description: str | None = None
    effect_scope: EffectScope


class ModelParameters(ContractModel):
    """解码参数（temperature/top_p/top_k/max_output_tokens/seed/stop）。

    由 pilot 配置的 decoding 段构造；Agent 集成按其原生客户端约定使用。
    """

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: tuple[str, ...] = ()


class RequestScope(ContractModel):
    """请求的隔离与数据策略边界：只放受限引用/稳定标识，不放凭据。

    tenant_ref 必须与请求能力引用的 tenant_ref 一致（storage 校验）。
    """

    tenant_ref: VersionedRef | None = None
    session_ref: VersionedRef | None = None
    data_policy_ref: VersionedRef


class ModelRequest(ContractModel):
    """原始不可变模型请求：一次逻辑 LM 调用的完整输入。

    生产者：Agent（wrapper 转换）；消费者：M1（画像视图）、M3（执行视图）、
    Agent 模型集成（真实调用）和 Coordinator 的完整在线 SpanRecord。经 request_ref 受限解析，
    能力引用控制访问权限，但不会从记录中删去正文。
    """

    request_version: VersionId = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$",
    )
    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    # Opaque Agent-owned request captured before TraceOffload routing.  The
    # core may inspect/record it, but must never reconstruct it for transport.
    # Each Agent integration defines the shape (mini-swe-agent uses its raw
    # message list; another Agent may use a different JSON payload).
    native_payload: JsonValue | None = None
    input_text: str | None = None
    attachments: tuple[Attachment, ...] = ()
    tool_definitions: tuple[ToolDefinition, ...] = ()
    output_contract_ref: VersionedRef | None = None
    model_parameters: ModelParameters
    request_scope: RequestScope


__all__ = [
    "ModelMessageToolCall",
    "ModelMessage",
    "Attachment",
    "ToolDefinition",
    "ModelParameters",
    "RequestScope",
    "ModelRequest",
]
