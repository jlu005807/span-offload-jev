"""M1 在线语义输入构建：把执行前可观察事实转成稳定文本。

该模块只做确定性文本投影，不做网络调用、不修改 Agent 消息历史。
输入只允许是 ``InvocationContext``（含 prior_execution_facts）、``ModelRequest``
和 M1 产出的 ``RuntimeProfile``；当前 model 输出、tool 结果、OutcomeEvent
都属于未来信息，禁止进入本模块。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from ..contracts import (
    InvocationContext,
    MessageRole,
    ModelMessage,
    ModelRequest,
    OutputContract,
    PriorExecutionFacts,
    PriorToolOutcome,
    RuntimeProfile,
)
# ⚠️ 只用**纯函数**那两份（`span_features/` 是离线/在线共用的单一实现），
# 不引 `components/` 里的任何东西 —— 那样会绕回 runner，成为循环 import。
from .constants import FeatureConstants
from .signature import signature_inputs

TEMPLATE_VERSION = "span-embed-v0"
SUCCESSFUL_ROUND_LIMIT = 5
MAX_TEXT_CHARS = 8_000
PER_MESSAGE_CHARS = 1_200
FALLBACK_RECENT_MESSAGES = 10


@dataclass(frozen=True)
class SpanEmbeddingInput:
    """一次 Span 的 embedding 文本输入及其审计元数据。"""

    workflow_run_id: str
    episode_id: str
    logical_invocation_id: str
    template_version: str
    text: str
    text_sha256: str
    message_count: int
    rendered_message_count: int
    successful_round_count: int
    selected_success_round_count: int
    target_model: str | None = None
    native_payload_present: bool = False
    prior_facts_present: bool = False


@dataclass(frozen=True)
class _ToolRound:
    """消息序列中的一个工具调用轮次：assistant tool_calls 及其后续 tool 消息。"""

    start: int
    end: int
    call_ids: tuple[str, ...]


class SpanEmbeddingTextBuilder:
    """把 M1 时刻的事实投影为 ``SpanEmbeddingInput``。

    最近成功轮次规则的 v0 口径：

    - 一轮 = assistant tool_calls + 对应 tool 消息；
    - 成功 = 该轮所有 tool_call_id 的 ``PriorToolOutcome.success`` 均为 True；
    - 从最新一轮往前累计最多 5 个成功轮；
    - 窗口从最旧的入选成功轮开始，期间失败轮和非 tool-call 轮原样保留。
    """

    def __init__(
        self,
        *,
        template_version: str = TEMPLATE_VERSION,
        successful_round_limit: int = SUCCESSFUL_ROUND_LIMIT,
        max_text_chars: int = MAX_TEXT_CHARS,
        per_message_chars: int = PER_MESSAGE_CHARS,
        fallback_recent_messages: int = FALLBACK_RECENT_MESSAGES,
    ) -> None:
        if successful_round_limit < 1:
            raise ValueError("successful_round_limit must be positive")
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")
        if per_message_chars < 1:
            raise ValueError("per_message_chars must be positive")
        if fallback_recent_messages < 1:
            raise ValueError("fallback_recent_messages must be positive")
        self._template_version = template_version
        self._successful_round_limit = successful_round_limit
        self._max_text_chars = max_text_chars
        self._per_message_chars = per_message_chars
        self._fallback_recent_messages = fallback_recent_messages

    def build(
        self,
        context: InvocationContext,
        request: ModelRequest,
        profile: RuntimeProfile,
        *,
        target_model: str | None = None,
    ) -> SpanEmbeddingInput:
        messages = list(request.messages)
        prior = context.prior_execution_facts
        outcomes = {
            outcome.tool_call_id: outcome
            for outcome in (prior.tool_outcomes if prior is not None else ())
        }
        rounds = _tool_rounds(messages)
        window_start, successful, selected = _select_window(
            messages,
            rounds,
            outcomes,
            successful_round_limit=self._successful_round_limit,
            fallback_recent_messages=self._fallback_recent_messages,
        )
        rendered_messages = messages[window_start:]
        header = _header_lines(context, request, profile, prior, target_model)
        body = [self._render_message(message) for message in rendered_messages]
        while body and sum(len(line) + 1 for line in header + body) > self._max_text_chars:
            body.pop(0)
        text = "\n".join([*header, *body])
        if len(text) > self._max_text_chars:
            text = text[: self._max_text_chars]
        return SpanEmbeddingInput(
            workflow_run_id=context.workflow_run_id,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            template_version=self._template_version,
            text=text,
            text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            message_count=len(messages),
            rendered_message_count=len(rendered_messages),
            successful_round_count=successful,
            selected_success_round_count=selected,
            target_model=target_model,
            native_payload_present=request.native_payload is not None,
            prior_facts_present=prior is not None,
        )

    def _render_message(self, message: ModelMessage) -> str:
        return _render_model_message(message, per_message_chars=self._per_message_chars)


def _tool_rounds(messages: Sequence[ModelMessage]) -> list[_ToolRound]:
    rounds: list[_ToolRound] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != MessageRole.ASSISTANT or not message.tool_calls:
            index += 1
            continue
        start = index
        call_ids = tuple(call.tool_call_id for call in message.tool_calls)
        index += 1
        while index < len(messages) and messages[index].role == MessageRole.TOOL:
            index += 1
        rounds.append(_ToolRound(start=start, end=index - 1, call_ids=call_ids))
    return rounds


def _round_success(
    round_: _ToolRound,
    outcomes: Mapping[str, PriorToolOutcome],
) -> bool:
    if not round_.call_ids:
        return False
    return all(
        outcome is not None and outcome.success
        for outcome in (outcomes.get(call_id) for call_id in round_.call_ids)
    )


def _select_window(
    messages: Sequence[ModelMessage],
    rounds: Sequence[_ToolRound],
    outcomes: Mapping[str, PriorToolOutcome],
    *,
    successful_round_limit: int,
    fallback_recent_messages: int,
) -> tuple[int, int, int]:
    """返回 (window_start, 总成功轮数, 选中的成功轮数)。"""

    successful_rounds = [
        round_ for round_ in rounds if _round_success(round_, outcomes)
    ]
    success_starts: list[int] = []
    for round_ in reversed(successful_rounds):
        success_starts.append(round_.start)
        if len(success_starts) >= successful_round_limit:
            break
    if success_starts:
        window_start = success_starts[-1]
    else:
        window_start = max(0, len(messages) - fallback_recent_messages)
    # 如果窗口正好紧跟一条 user/system/developer 消息，把它带上；否则不
    # 反向穿透工具轮，避免把更早的未入选轮次重新带进窗口。
    if window_start > 0 and messages[window_start - 1].role in (
        MessageRole.USER,
        MessageRole.SYSTEM,
        MessageRole.DEVELOPER,
    ):
        window_start -= 1
    return window_start, len(successful_rounds), len(success_starts)


def _header_lines(
    context: InvocationContext,
    request: ModelRequest,
    profile: RuntimeProfile,
    prior: PriorExecutionFacts | None,
    target_model: str | None,
) -> list[str]:
    workflow = context.workflow_context
    goal = workflow.workflow_goal.goal_text
    if not goal and workflow.workflow_goal.goal_ref is not None:
        goal = (
            f"{workflow.workflow_goal.goal_ref.kind}:"
            f"{workflow.workflow_goal.goal_ref.id}:"
            f"{workflow.workflow_goal.goal_ref.version}"
        )
    output_ref = request.output_contract_ref
    output_ref_text = (
        f"{output_ref.kind}:{output_ref.id}:{output_ref.version}"
        if output_ref is not None
        else "none"
    )
    verifier_text = (
        "; ".join(
            f"stage={record.stage.value}, outcome={record.outcome.value}, "
            f"guard={record.guard_ref.kind}:{record.guard_ref.id}:{record.guard_ref.version}"
            for record in (prior.verifier_results if prior is not None else ())
        )
        or "none"
    )
    state = prior.state_observation if prior is not None else None
    state_text = (
        "effect={effect}, commit={commit}, recovery_required={recovery}, "
        "rollback={rollback}, resume={resume}, fault={fault}".format(
            effect=state.effect.value,
            commit=state.commit_status.value,
            recovery=str(state.recovery_required).lower(),
            rollback=state.rollback_status.value,
            resume=state.resume_status.value,
            fault=state.fault_class or "none",
        )
        if state is not None
        else "none"
    )
    return [
        f"[AGENT] id={workflow.agent_id}; role={workflow.agent_role.value}; "
        f"node={workflow.node_context.node_id}/{workflow.node_context.node_type}; "
        f"step={workflow.node_context.step_index}",
        f"[GOAL] {_truncate(goal or 'none', 1000)}",
        f"[REQUEST] input_text={_truncate(request.input_text or 'none', 1000)}; "
        f"output_contract_ref={output_ref_text}; "
        f"native_payload={'present' if request.native_payload is not None else 'none'}; "
        f"message_count={len(request.messages)}; "
        f"estimated_input_tokens={profile.descriptor.estimated_input_tokens}",
        f"[MODEL] target={target_model or 'unknown'} (M1 pre-routing)",
        _descriptor_line(profile),
        f"[TOOLS] {_tools_text(request, workflow)}",
        f"[LAST_OBSERVATION_CLASS] {profile.last_observation_class}",
        f"[PRIOR_VERIFICATION] {verifier_text}",
        f"[PRIOR_STATE] {state_text}",
        "[RECENT_CONTEXT]",
    ]


def _descriptor_line(profile: RuntimeProfile) -> str:
    descriptor = profile.descriptor
    return (
        "[DESCRIPTOR] "
        f"subtask={descriptor.subtask_type}; "
        f"output={_output_kind(descriptor.output_contract)}; "
        f"risk={descriptor.effect_risk.value}; "
        f"recovery={descriptor.recovery_level.value}; "
        f"downstream={descriptor.downstream_dependency.value}"
    )


def _output_kind(output_contract: object) -> str:
    if isinstance(output_contract, OutputContract):
        return output_contract.output_kind.value
    return "none"


def _tools_text(request: ModelRequest, workflow: object) -> str:
    definitions = [
        f"{tool.tool_name}(schema={tool.schema_ref.id}:{tool.schema_ref.version}, "
        f"effect={tool.effect_scope.value})"
        for tool in request.tool_definitions
    ]
    if definitions:
        return "; ".join(definitions)
    available = getattr(workflow, "available_tools", ())
    if available:
        return "; ".join(
            f"{ref.kind}:{ref.id}:{ref.version}" for ref in available
        )
    return "none"


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _render_model_message(message: ModelMessage, *, per_message_chars: int) -> str:
    parts: list[str] = []
    content = _content_text(message.content)
    if content:
        parts.append(content)
    if message.reasoning_content:
        parts.append(f"[reasoning] {message.reasoning_content}")
    for call in message.tool_calls:
        arguments = json.dumps(
            call.arguments,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        parts.append(
            f"[tool_call name={call.tool_name} id={call.tool_call_id} args={arguments}]"
        )
    if message.name:
        parts.append(f"[name={message.name}]")
    if message.tool_call_id:
        parts.append(f"[tool_call_id={message.tool_call_id}]")
    rendered = " | ".join(part for part in parts if part)
    return _truncate(f"{message.role.value}: {rendered}".rstrip(), per_message_chars)


def build_embedding_text_from_span_record(
    span_record: object,
    *,
    max_text_chars: int = MAX_TEXT_CHARS,
    per_message_chars: int = PER_MESSAGE_CHARS,
    recent_messages: int = 12,
) -> str:
    """SpanRecord 的离线/后台 fallback 文本投影。

    该函数只用于 M4 worker 在缺省 embedding_text 时构建输入；如果上层已经
    提供 M1 的完整 ``SpanEmbeddingInput.text``，应优先使用它。
    """

    route = getattr(span_record, "route_plan", None)
    workflow_run_id = getattr(route, "workflow_run_id", "workflow-unknown")
    episode_id = getattr(route, "episode_id", "episode-unknown")
    logical_invocation_id = getattr(route, "logical_invocation_id", "invocation-unknown")
    target = getattr(route, "target", None)
    model_ref = getattr(target, "model_ref", None)
    target_kind = getattr(getattr(target, "kind", None), "value", "unknown")
    model_id = getattr(model_ref, "id", "unknown")
    execution_mode = getattr(getattr(route, "execution_mode", None), "value", "unknown")
    lines = [
        f"[SPAN] workflow={workflow_run_id}; episode={episode_id}; "
        f"invocation={logical_invocation_id}",
        f"[ROUTE] target={target_kind}:{model_id}; mode={execution_mode}",
    ]
    profile = getattr(span_record, "runtime_profile", None)
    if profile is not None:
        lines.append(_descriptor_line(profile))
    request = getattr(span_record, "model_request", None)
    messages = tuple(getattr(request, "messages", ()) or ())
    lines.append(f"[REQUEST] message_count={len(messages)}")
    lines.append("[RECENT_CONTEXT]")
    if messages:
        for message in messages[-recent_messages:]:
            lines.append(
                _render_model_message(message, per_message_chars=per_message_chars)
            )
    else:
        lines.append("none")
    text = "\n".join(lines)
    if len(text) > max_text_chars:
        text = text[-max_text_chars:]
    return text


# ============================================================================
# v1 配方（0923）—— GOAL 在前 + (思考 + 输出 + 工具名/参数) × K
# ============================================================================
#
# 与 v0 的差别（v0 见上 `SpanEmbeddingTextBuilder`）：
#   | | v0 | v1 |
#   |---|---|---|
#   | 骨架 | 10 行 header + 窗口内**全部** messages | `[GOAL]` + **前 K 个 span** |
#   | 工具返回内容 | **包含**（占 80% 字符） | **剔除** |
#   | `[TOOLS]` 语义 | 可用工具的 **schema** | 本 span **实际调用的 API 名 + 参数** |
#   | 截断方向 | **头保留** | **尾保留** |
#   | 窗口 | 最近 ≤5 个成功轮（实测恒走兜底） | 最近 **K 个 assistant 消息** |
#
# **格式固定**：每个 span 块恒为三行 `[REASONING] / [OUTPUT] / [TOOLS]`，
# 顺序不变、标签不变。缺内容写 `none`，**不留空行差异** —— 这样不同 span 的
# 文本在结构上逐位可比。
TEMPLATE_VERSION_V1 = "span-embed-v1"
SPAN_WINDOW_K = 6
REASONING_TAIL_CHARS_V1 = 800
ASSISTANT_TAIL_CHARS_V1 = 1500
TOOL_ARGS_CHARS = 240
GOAL_CHARS = 2000

_API_CALL_RE = re.compile(r"apis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+\-]*\n(.*?)```", re.S)
_WS_RE = re.compile(r"\s+")


def _tail_v1(value: str, limit: int) -> str:
    """取尾部。短文本原样返回。"""
    text = value or ""
    return text if len(text) <= limit else text[-limit:]


def _fenced_code(text: str) -> str:
    """把消息里所有 ``` 围栏块的代码拼起来（AppWorld 的代码只在围栏里）。"""
    return "\n".join(m.group(1) for m in _FENCE_RE.finditer(text or ""))


def _balanced_args(code: str, open_index: int, limit: int) -> str:
    """从 `(` 起做括号配对，取出实参并压成单行；超长或未闭合则在 limit 处截断。"""
    if open_index >= len(code) or code[open_index] != "(":
        return ""
    depth = 0
    for index in range(open_index, len(code)):
        char = code[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                inner = code[open_index + 1 : index]
                return _WS_RE.sub(" ", inner).strip()[:limit]
    inner = code[open_index + 1 : open_index + 1 + limit]
    return _WS_RE.sub(" ", inner).strip()


def api_calls_with_args(code: str, *, max_args_chars: int = TOOL_ARGS_CHARS) -> list[str]:
    """抽出 `app.api(参数)` 形式的调用串。

    工具名口径与离线 `classfierTrain/aw3_build.py:api_calls_of_code` **同一条正则**
    （`apis\\.(\\w+)\\.(\\w+)\\s*\\(`），保证在线/离线可比；参数是本地新增，
    只压成单行 + 截断，不改名。
    """
    out: list[str] = []
    for match in _API_CALL_RE.finditer(code or ""):
        args = _balanced_args(code, match.end() - 1, max_args_chars)
        out.append(f"{match.group(1)}.{match.group(2)}({args})")
    return out


def _tools_line(message: ModelMessage) -> str:
    """一个 span 的工具名 + 参数。

    两条来源，按优先级：
    1. ``message.tool_calls``（结构化）—— AppWorld 上**实测恒为空**
    2. assistant ``content`` 的代码围栏里 ``apis.app.api(...)``—— AppWorld 走这条
    结果按首次出现去重，`none` 表示这个 span 没调 API（纯思考 / 纯回答）。
    """
    names: list[str] = []
    for call in message.tool_calls:
        arguments = call.arguments
        rendered = (
            json.dumps(arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            if not isinstance(arguments, str)
            else arguments
        )
        names.append(f"{call.tool_name}({_WS_RE.sub(' ', rendered).strip()[:TOOL_ARGS_CHARS]})")
    names.extend(api_calls_with_args(_fenced_code(_content_text(message.content))))
    if not names:
        return "none"
    return ", ".join(dict.fromkeys(names))


def _span_block(index: int, message: ModelMessage) -> str:
    """一个 span 的固定三行块。[index] 用相对位置标注（1 = 最早的入选 span）。"""
    reasoning = _tail_v1(message.reasoning_content or "", REASONING_TAIL_CHARS_V1)
    output = _tail_v1(_content_text(message.content), ASSISTANT_TAIL_CHARS_V1)
    return (
        f"[SPAN {index}]\n"
        f"[REASONING] {reasoning or 'none'}\n"
        f"[OUTPUT] {output or 'none'}\n"
        f"[TOOLS] {_tools_line(message)}"
    )


def build_v1_text(
    messages: Sequence[ModelMessage],
    goal: str,
    *,
    span_window_k: int = SPAN_WINDOW_K,
) -> str:
    """v1 文本：`[GOAL]` 在前，后面是最近 K 个 assistant 消息（= K 个 span）。

    选窗口的依据：**一个 span = 一次模型调用 = 一条 assistant 消息**。
    工具返回（`role == tool`）不进文本 —— 离线实测它占 v0 文本的 80% 字符。
    """
    assistants = [m for m in messages if m.role is MessageRole.ASSISTANT]
    window = assistants[-span_window_k:] if span_window_k > 0 else []
    head = f"[GOAL] {_tail_v1(goal or '', GOAL_CHARS).strip() or 'none'}"
    blocks = [_span_block(i + 1, m) for i, m in enumerate(window)]
    return "\n\n".join([head, *blocks]) if blocks else head


def _goal_text(workflow: object) -> str:
    goal = getattr(getattr(workflow, "workflow_goal", None), "goal_text", None) or ""
    if goal:
        return str(goal)
    goal_ref = getattr(getattr(workflow, "workflow_goal", None), "goal_ref", None)
    if goal_ref is not None:
        return f"{goal_ref.kind}:{goal_ref.id}:{goal_ref.version}"
    return ""


class SpanEmbeddingTextBuilderV1:
    """v1 构建器。接口与 `SpanEmbeddingTextBuilder` 一致，便于互换。

    **不修改 v0** —— `template_version` 是审计字段，改了会让已落库的历史记录
    无法解释。要换配方就换类（并换 `template_version`）。
    """

    def __init__(
        self,
        *,
        template_version: str = TEMPLATE_VERSION_V1,
        span_window_k: int = SPAN_WINDOW_K,
    ) -> None:
        if span_window_k < 1:
            raise ValueError("span_window_k must be positive")
        self._template_version = template_version
        self._span_window_k = span_window_k

    def build(
        self,
        context: InvocationContext,
        request: ModelRequest,
        profile: RuntimeProfile,
        *,
        target_model: str | None = None,
    ) -> SpanEmbeddingInput:
        del profile  # v1 不读 profile 派生字段（v0 的 header 里用了）
        messages = list(request.messages)
        text = build_v1_text(
            messages,
            _goal_text(context.workflow_context),
            span_window_k=self._span_window_k,
        )
        assistants = [m for m in messages if m.role is MessageRole.ASSISTANT]
        return SpanEmbeddingInput(
            workflow_run_id=context.workflow_run_id,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            template_version=self._template_version,
            text=text,
            text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            message_count=len(messages),
            rendered_message_count=min(len(assistants), self._span_window_k),
            successful_round_count=0,
            selected_success_round_count=0,
            target_model=target_model,
            native_payload_present=request.native_payload is not None,
            prior_facts_present=context.prior_execution_facts is not None,
        )


def build_embedding_text_from_span_record_v1(
    span_record: object,
    *,
    span_window_k: int = SPAN_WINDOW_K,
) -> str:
    """后补路径的 v1 文本。与 `SpanEmbeddingTextBuilderV1.build` 同配方、同格式。

    ⚠️ 与 v0 的 `build_embedding_text_from_span_record` 的区别：那个的骨架是
    `[SPAN]/[ROUTE]/…`，与主路径**完全不同**；这个刻意保持一致，
    两条路对同一个 span 产出**逐字相同**的文本。
    """
    request = getattr(span_record, "model_request", None)
    messages = tuple(getattr(request, "messages", ()) or ())
    route = getattr(span_record, "route_plan", None)
    goal = ""
    workflow_context = getattr(span_record, "workflow_context", None)
    if workflow_context is not None:
        goal = _goal_text(workflow_context)
    if not goal:
        goal = str(getattr(route, "workflow_goal_text", "") or "")
    return build_v1_text(messages, goal, span_window_k=span_window_k)


__all__ = [
    "SpanEmbeddingInput",
    "SpanEmbeddingTextBuilder",
    "SpanEmbeddingTextBuilderV1",
    "SpanEmbeddingTextBuilderV2",
    "build_embedding_text_from_span_record",
    "build_embedding_text_from_span_record_v1",
    "build_embedding_text_from_span_record_v2",
    "build_v1_text",
    "build_v2_text",
    "api_calls_with_args",
    "TEMPLATE_VERSION_V1",
    "TEMPLATE_VERSION_V2",
]


# ---------------------------------------------------------------------------
# v2 配方 —— 与离线 `classfierTrain/aw3_build.py:span_text_v2` 同配方
# ---------------------------------------------------------------------------
# 为什么要有 v2：v4 的 L2 预测器（`classfierTrain/out/router/l2_new_v4_*.json`）是在
# **`V2_goal_recent`** 文本的 embedding 上训的，不是 `V1_span`。
# 两个配方的文本**零字符相同**（实测 761 个 key 无一相同），所以在线必须
# 产出 v2 才能让离线训好的向量对上 —— 对不上不会报错，只会静默失准。
#
# ⚠️ 改动这里的任何一处，都必须同步改 `aw3_build.py` 的对应处，否则
#    在线/离线文本漂移。漂移**不报错**，只是模型悄悄变差。
TEMPLATE_VERSION_V2 = "span-embed-v2"
#: 离线 `aw3_build.RECENT_WINDOW_K_V2`（v1 是 5 条；v2 每 span 变小后窗口放大到 10）
SPAN_WINDOW_K_V2 = 10
#: 离线 `aw3_build.REASONING_TAIL_CHARS` / `ASSISTANT_TAIL_CHARS`
REASONING_TAIL_CHARS_V2 = 800
ASSISTANT_TAIL_CHARS_V2 = 1500


def _tools_line_v2(message: ModelMessage) -> str:
    """v2 的工具行：**只留 `app.api`，不带参数**，按首次出现去重，空则 `none`。

    与 V1 的 `_tools_line` 的关键差别：v1 是 `app.api(参数)` 且参数截断 240 字符；
    v2 把参数整个去掉。依据（`aw3_build.span_text_v2` 的注释）：v1 文本里工具
    输出占 67%，去掉后 v2 最大值从 152,713 降到 17,859 字符。
    """
    code = _fenced_code(_content_text(message.content))
    names = [f"{m.group(1)}.{m.group(2)}" for m in _API_CALL_RE.finditer(code)]
    return ", ".join(dict.fromkeys(names)) or "none"


def _span_block_v2(message: ModelMessage) -> str:
    """v2 的单个 span 块：`[TOOLS] / [REASONING] / [ASSISTANT]` 三行，顺序固定。

    ⚠️ 空值**不写 `none`**（与 v1 不同）：思考/输出为空时就是 `[REASONING] ` /
    `[ASSISTANT] `（**带尾空格**）。这是 `aw3_build.span_text_v2` 的行为，
    逐字对齐靠的就是这些细节。
    """
    reasoning = _tail_v1(message.reasoning_content or "", REASONING_TAIL_CHARS_V2)
    content = _tail_v1(_content_text(message.content), ASSISTANT_TAIL_CHARS_V2)
    return (
        f"[TOOLS] {_tools_line_v2(message)}\n"
        f"[REASONING] {reasoning}\n"
        f"[ASSISTANT] {content}"
    )


def build_v2_text(
    messages: Sequence[ModelMessage],
    goal: str,
    *,
    span_window_k: int = SPAN_WINDOW_K_V2,
    skip_leading_messages: int = 0,
) -> str:
    """v2 文本 = `[GOAL] <instruction>` + 最近 K 个 span 块。

    与 `build_v1_text` 的差别只有两处：**块格式**（v1 是 `[SPAN i]/[REASONING]/
    [OUTPUT]/[TOOLS]`，v2 是 `[TOOLS]/[REASONING]/[ASSISTANT]`）和**窗口大小**
    （6 → 10）。`[GOAL]` 段**不截断**（v1 截断 2000）。

    "一个 span = 一次模型调用 = 一条 assistant 消息" 这条两边一致，沿用 v1 的选窗。

    ⚠️ **`skip_leading_messages` 不是可选项，是必需的。** AppWorld 的 prompt 模板
    自带一段**完整的 few-shot 示例轨迹**（实测 23 条消息，其中 **10 条 assistant**，
    内容是另一个 task 的完整 rollout，以 `complete_task` 结尾）。它由
    `SimplifiedReActCodeAgent.initialize` 烘进 `self.messages`，所以**第一次调用
    `generate()` 时就已经有 10 条 assistant 消息了**。

    离线的 `prefix_steps` **只记真实执行的步、不含示例**。不跳过的话在线窗口会被
    示例塞满（窗口正好是 10）——早期步更是**整窗全是示例**，而离线同一步只有
    `[GOAL]`。实测线上真机：11 次调用每次都是 10 个示例块，逐字比对离线文本
    **0/11 一致**。这就是"在线/离线不同分布"的根因。
    """
    if skip_leading_messages < 0:
        raise ValueError("skip_leading_messages must be non-negative")
    considered = messages[skip_leading_messages:]
    assistants = [m for m in considered if m.role is MessageRole.ASSISTANT]
    window = assistants[-span_window_k:] if span_window_k > 0 else []
    # ⚠️ 离线是 `f"[GOAL] {instr}\n\n" + "\n\n".join(blocks)` —— **空窗口时也带
    #    那个 `\n\n`**。写成 `"\n\n".join([head, *blocks])` 在非空时等价，但空窗口
    #    会少两个字符（实测真是机第一次调用的长 96 vs 离线 98）。
    head = f"[GOAL] {goal or ''}\n\n"
    return head + "\n\n".join(_span_block_v2(m) for m in window)


class SpanEmbeddingTextBuilderV2:
    """v2 构建器。接口与 `SpanEmbeddingTextBuilderV1` 一致，便于互换。

    **不修改 v1** —— `template_version` 是审计字段，改了会让已落库的历史记录
    无法解释。要换配方就换类（并换 `template_version`）。
    """

    def __init__(
        self,
        *,
        template_version: str = TEMPLATE_VERSION_V2,
        span_window_k: int = SPAN_WINDOW_K_V2,
        skip_leading_messages: int = 0,
    ) -> None:
        if span_window_k < 1:
            raise ValueError("span_window_k must be positive")
        self._template_version = template_version
        self._span_window_k = span_window_k
        self._skip_leading_messages = int(skip_leading_messages)

    def set_skip_leading_messages(self, count: int) -> None:
        """设置要跳过的**模板消息**数（AppWorld 的 `num_instruction_messages`）。

        Agent 建好之前拿不到这个数（它由 `initialize()` 烘进 agent），所以做成
        可变属性 —— runner 在 agent 就绪后注入。**不设就是 0，那会让窗口被
        prompt 的 few-shot 示例填满**，见 `build_v2_text` 的警告。
        """
        if count < 0:
            raise ValueError("skip count must be non-negative")
        self._skip_leading_messages = int(count)

    def build(
        self,
        context: InvocationContext,
        request: ModelRequest,
        profile: RuntimeProfile,
        *,
        target_model: str | None = None,
    ) -> SpanEmbeddingInput:
        del profile  # 与 v1 一致：不读 profile 派生字段
        messages = list(request.messages)
        text = build_v2_text(
            messages,
            _goal_text(context.workflow_context),
            span_window_k=self._span_window_k,
            skip_leading_messages=self._skip_leading_messages,
        )
        assistants = [m for m in messages if m.role is MessageRole.ASSISTANT]
        return SpanEmbeddingInput(
            workflow_run_id=context.workflow_run_id,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            template_version=self._template_version,
            text=text,
            text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            message_count=len(messages),
            rendered_message_count=min(len(assistants), self._span_window_k),
            successful_round_count=0,
            selected_success_round_count=0,
            target_model=target_model,
            native_payload_present=request.native_payload is not None,
            prior_facts_present=context.prior_execution_facts is not None,
        )


def build_embedding_text_from_span_record_v2(
    span_record: object,
    *,
    span_window_k: int = SPAN_WINDOW_K_V2,
) -> str:
    """后补路径的 v2 文本。与 `SpanEmbeddingTextBuilderV2.build` 同配方、同格式。"""

    request = getattr(span_record, "model_request", None)
    messages = tuple(getattr(request, "messages", ()) or ())
    route = getattr(span_record, "route_plan", None)
    goal = ""
    workflow_context = getattr(span_record, "workflow_context", None)
    if workflow_context is not None:
        goal = _goal_text(workflow_context)
    if not goal:
        goal = str(getattr(route, "workflow_goal_text", "") or "")
    return build_v2_text(messages, goal, span_window_k=span_window_k)


# ---------------------------------------------------------------------------
# ④ `C_bare_w10`：最近 N 个「动作签名」的文本（add4 预测器的第二段嵌入）
# ---------------------------------------------------------------------------

#: 离线 `classfierTrain/03_特征提取/sf_actions_variants.VARIANTS` 里 `C_bare_w10`
#: 的窗口。⚠️ **改这个数 = 换向量空间** ⇒ 已导出的 `emb_C_bare_w10` 训练产物与
#: 预测器 artifact 全部作废。`tests/components/test_l2_add4_predictor.py` 用 AST
#: 抽离线那个常量来钉住它。
C_BARE_WINDOW = 10

#: 判不出签名的动作在文本里的占位。离线 `sf_actions_variants._render` 逐字如此
#: （`a.signature if a.signature is not None else "<unparsed>"`）。
C_BARE_UNPARSED = "<unparsed>"

#: 审计字段：这份文本的配方名（进 `SpanEmbeddingInput.template_version`）。
C_BARE_TEMPLATE_VERSION = "C_bare_w10"


def build_c_bare_text(
    messages: Sequence[ModelMessage],
    *,
    window: int = C_BARE_WINDOW,
    skip_leading_messages: int = 0,
    constants: FeatureConstants | None = None,
) -> str:
    """`C_bare_w10` 文本 = 最近 `window` 个动作签名的换行连接。

    这条规则是**实测**出来的，不是推的：拿 `runs/验证集` 的 380 个 cell，在
    `prefix_trace.json` 的 `next_input_context.messages` 上跑本函数，与离线
    `out/actions_variants/actions_texts.jsonl` 的 `C_bare_w10` **逐字比对**，
    跳过 20~23 条消息时 **380/380 相同**（不跳过只有 37%）。

    三条容易写错的：
    * **空的 assistant 消息要跳过。** 离线那边「一个 tool execution 一行」，
      没执行的动作根本不产生行。`042a9fc_2@k1` 离线是**空串**（k=1 没有前缀步），
      不跳过就会拿 AppWorld 模板里的 few-shot 示例填满整个窗口。
    * **判不出签名 → `<unparsed>`**，不是跳过 —— 离线明确占一行。
    * **一行都没有 → 空串。** 上层必须把它翻译成 1024 维权零向量
      （离线的 embed 阶段就是 `np.zeros` 且不发 HTTP 请求），
      **不能**把空串发给嵌入模型。
    """
    constants = constants or FeatureConstants()
    if skip_leading_messages < 0:
        raise ValueError("skip_leading_messages must be non-negative")
    lines: list[str] = []
    for message in messages[skip_leading_messages:]:
        if message.role is not MessageRole.ASSISTANT:
            continue
        code = _fenced_code(_content_text(message.content))
        if not code.strip():
            continue
        signature = signature_inputs(code, constants).signature
        lines.append(signature if signature else C_BARE_UNPARSED)
    chosen = lines[-window:] if window > 0 else []
    return "\n".join(chosen)


class SpanActionsTextBuilder:
    """④ 的构建器。接口与 `SpanEmbeddingTextBuilderV2` 一致，便于 runner 统一注入。

    **与 V2 相互独立**：V2 服务 M4 记忆与 v4 预测器（那 52 维），本类服务 add4
    预测器（116 维里的第 ④ 段）。两者共用的只是
    `set_skip_leading_messages` 的**注入时机**，不是文本、不是向量空间。

    与 `SpanEmbeddingTextBuilderV2` 同样**不读 `profile`** —— 这条文本只由
    Agent 自己的消息决定，掺进派生字段会让在线/离线不可比。
    """

    def __init__(
        self,
        *,
        template_version: str = C_BARE_TEMPLATE_VERSION,
        window: int = C_BARE_WINDOW,
        skip_leading_messages: int = 0,
        constants: FeatureConstants | None = None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._template_version = template_version
        self._window = int(window)
        self._skip_leading_messages = int(skip_leading_messages)
        self._constants = constants or FeatureConstants()

    def set_skip_leading_messages(self, count: int) -> None:
        """与 V2 同义同值：跳过 AppWorld prompt 模板里的 few-shot 消息。

        ⚠️ runner 必须把**同一个数**注入到本对象 —— 漏掉它，k=1 的窗口会被
        示例动作填满，而离线在 k=1 是空串。这是**静默**的：p 照样落在 [0,1]，
        只是错了。
        """
        if count < 0:
            raise ValueError("skip count must be non-negative")
        self._skip_leading_messages = int(count)

    def build(
        self,
        context: InvocationContext,
        request: ModelRequest,
        profile: RuntimeProfile,
        *,
        target_model: str | None = None,
    ) -> SpanEmbeddingInput:
        del profile  # 与 v1/v2 一致：只读 request 的消息
        messages = list(request.messages)
        text = build_c_bare_text(
            messages,
            window=self._window,
            skip_leading_messages=self._skip_leading_messages,
            constants=self._constants,
        )
        return SpanEmbeddingInput(
            workflow_run_id=context.workflow_run_id,
            episode_id=context.episode_id,
            logical_invocation_id=context.logical_invocation_id,
            template_version=self._template_version,
            text=text,
            text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            message_count=len(messages),
            # 这个文本的「渲染了几条」= 有几行非空动作，不是消息数。
            rendered_message_count=sum(1 for line in text.split("\n") if line),
            successful_round_count=0,
            selected_success_round_count=0,
            target_model=target_model,
            native_payload_present=request.native_payload is not None,
            prior_facts_present=context.prior_execution_facts is not None,
        )
