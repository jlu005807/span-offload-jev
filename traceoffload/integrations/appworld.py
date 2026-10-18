"""Seamless TraceOffload integration for AppWorld's simplified agents.

The integration stays outside ``appworld_agents`` and hooks only two native
boundaries:

* ``agent.language_model.generate(...)`` -- logical LM span entrance.
* ``world.batch_execute(codes)``           -- tool phase exit.

The native Agent loop, prompts, message history and action parsing are never
modified.  The actual model transport uses the native ``LanguageModel`` and the
original message objects; ``ModelRequest``/``ModelResponse`` are only the
framework-neutral audit/control view used by M1-M4.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MethodType
from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime

from traceoffload.clock import real_utcnow
from traceoffload.effects import EffectVerdict
from traceoffload.coordinator import Coordinator, CoordinatorError
from traceoffload.m3.integration import AdapterSession, AdapterSessionSpec, ToolRegistryEntry
from traceoffload.m3.tool_proxy import CallableToolProxy
from traceoffload.contracts import (
    EffectRisk,
    EffectScope,
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelMessageToolCall,
    ModelParameters,
    ModelRequest,
    ModelResponse,
    ModelResponseToolCall,
    ModelTarget,
    ModelTargetKind,
    ParseStatus,
    RecoveryLevel,
    RequestScope,
    StateAccessMode,
    ToolDefinition,
    VersionedRef,
    WorkflowGoal,
)
from traceoffload.interfaces import ModelCallResult, ToolAction, ToolProxyResult

def _real_time() -> float:
    """Monotonic clock that bypasses AppWorld/freezegun time freezing."""

    return time.clock_gettime(time.CLOCK_MONOTONIC)


_CURRENT_WORKFLOW_RUN_ID: str | None = None
_ROLE_MAP = {
    "system": MessageRole.SYSTEM,
    "developer": MessageRole.DEVELOPER,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
}
_LOGGER = logging.getLogger(__name__)

#: **整条 AppWorld 链路上唯一的工具名。**
#:
#: 模型输出（react_code 的代码 / function_calling 的 API 名 / full_code 的代码）
#: 先由 `_default_code_builder` 按类型吸收差异，再由集成**自己合成**一个 tool call
#: —— 名字永远是它。所以：
#:
#:   · 策略白名单认的是它（`run_appworld_traceoffload.py`）
#:   · `_pending_from_response` 按它筛，筛不到就**整体绕过工具阶段**
#:   · `tool_definitions_for` / `batch_execute` 也用同一个
#:
#: 这几个位置原先各自硬编码同一个字符串，**全靠手工同步**。收敛成常量之后，
#: `tests/integrations/test_appworld_batch_tool_name.py` 再钉住「集成侧 == 策略侧」。
BATCH_TOOL_NAME = "execute_code_batch"
#: **不能**放行的内建函数 —— 派生安全名单时从这里排除。
#:
#: 这一半不能靠派生：它们要么能做 I/O，要么能改执行环境，要么能凭空取到
#: 全局对象。**必须显式列出**，而且每条都有测试钉住
#: （`tests/integrations/test_appworld_analyzer_safe_calls.py`）。
#:
#: `open` 尤其重要：AppWorld 的沙箱在**运行期**把 `open()` 降成只读模式，
#: 但分析器判的是**文本** —— 不能拿运行期的性质去放行静态分析。
_UNSAFE_BUILTINS = frozenset({
    # I/O 与交互
    "open", "input", "breakpoint",
    # 动态执行 / 导入
    "eval", "exec", "compile", "__import__",
    # 作用域与对象改写
    "globals", "locals", "vars", "setattr", "delattr",
    # 交互式解释器里由 site 注入的
    "exit", "quit", "help",
})

#: 纯函数内建 —— **从 `builtins` 派生**，不是手维护的短名单。
#:
#: 手维护的名单每漏一项，后果都一样且都不报错：
#:
#:     漏项 → `unknown_call:` → 准入闸无条件拒 `unknown`
#:          → **整个 SLM 批次回退大模型**
#:
#: 已经撞过两次（`songs.extend(...)` 落在 6 个不同任务上、`next()` 落在另外几个），
#: 两次都不是「某个任务特殊」。⇒ 改成派生：`dir(builtins)` 就是全集，不会烂。
_SAFE_CALL_NAMES = frozenset(
    name
    for name in dir(builtins)
    if not name.startswith("_")
    and name not in _UNSAFE_BUILTINS
    # `True`/`False`/`None` 也在 `dir(builtins)` 里，但它们不是函数，
    # 写成 `True(x)` 是个取常量的调用 —— 收进来只会让名单和事实对不上。
    and callable(getattr(builtins, name))
)

#: 手写的小集合：标准库里**确定无副作用**的几个模块函数。
#: 与上面不同，这一类无法派生（模块里混着有副作用的函数），所以显式列出。


#: 能**安全判为只读**的标准库模块 —— 即「调用它不会有世界副作用」。
#:
#: 派生方式：拿 AppWorld 自己的 `ALLOWED_MODULE_NAMES`（那是它的可导入表），
#: **减掉能碰文件系统 / 起进程 / 改环境的那些**。为什么必须减：
#: AppWorld 那张表说的是「能 import」，不是「无副作用」—— 里面就有 `os` / `pathlib` / `io`。
#:
#: ⚠️ 这份名单**必须能追溯到 AppWorld 的表**，理由不是洁癖：它是判「写/读」的
#: 依据之一，为了让某个任务跑通而往这里加一个名字，等于给那个任务开洞。
#: `tests/integrations/test_appworld_analyzer_safe_calls.py` 有一条测试专门盯这件事。
_PURE_MODULES = frozenset({
    "abc", "array", "ast", "base64", "bisect", "calendar", "collections",
    "collections.abc", "colorsys", "contextlib", "contextvars", "copy", "csv",
    "dataclasses", "datetime", "decimal", "difflib", "dis", "email", "encodings",
    "enum", "errno", "fractions", "functools", "genericpath", "getopt", "graphlib",
    "hashlib", "heapq", "hmac", "html", "itertools", "json", "keyword", "marshal",
    "math", "numbers", "opcode", "operator", "optparse", "pendulum", "pprint",
    "pydoc_data", "queue", "quopri", "random", "re", "reprlib", "shlex",
    "sre_compile", "sre_constants", "sre_parse", "statistics", "string",
    "stringprep", "struct", "textwrap", "time", "token", "traceback", "types",
    "typing", "urllib", "uuid", "weakref", "xml", "yaml",
})

#: 显式**不纯**的模块根名 —— 派生 `_PURE_MODULES` 时被减掉的那些。
#: 列在这里是为了让「为什么 os 不在里面」有处可查，而不是靠读 diff 反推。
_IMPURE_MODULES = frozenset({
    "appworld", "builtins", "fnmatch", "inspect", "io", "os", "pathlib",
    "pydoc", "unittest",
})


@dataclass
class _Bindings:
    """一个 batch 里「哪些名字是什么」的静态视图。

    判读写的规则只需要三个事实，全部可以从 AST 上看出来、不需要执行：

        ``api_aliases``    `s = apis.spotify` ⇒ `s` 是 API 对象，`s.login()` 是写
        ``modules``        `import math` ⇒ `math.sqrt()` 是纯模块调用
        ``assigned``       在**本 batch 内被赋值过**的名字 ⇒ 它指的是本地数据

    第三条是收窄「不是 API 就只读」那条规则的那道闸：**从没被赋值过的名字**
    （函数参数、外部注入的东西）不属于本地数据 ⇒ 保持 unknown。
    """

    api_aliases: dict[str, str] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=dict)
    assigned: set[str] = field(default_factory=set)
    imported_functions: dict[str, str] = field(default_factory=dict)


def _target_names(node: ast.AST) -> list[str]:
    """赋值目标里的名字（支持 `a = ...` 与 `a, b = ...`）。"""

    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for element in node.elts for name in _target_names(element)]
    return []


def _api_object_path(value: ast.AST | None) -> str | None:
    """这个表达式是不是**一个 API 对象**？是就返回它在 `apis` 下的路径。

    关键区分：`apis.spotify`（对象）vs `apis.spotify.show_song()`（**数据**）。
    所以只有「`apis` 的属性链、且链上不含 Call」才算别名。

    `getattr(apis, "spotify")` 这种绕过属性链的写法归到 `""`（= `apis` 本身）——
    保守：后面 `x.login()` 会按 `login` 去查文档，查不到就是 unknown。
    """

    if isinstance(value, ast.Name) and value.id == "apis":
        return ""
    if isinstance(value, ast.Attribute):
        parts = []
        node: ast.AST = value
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id == "apis":
            return ".".join(reversed(parts))
        return None
    if isinstance(value, ast.Call):
        # 调用结果是数据 —— 但**参数里出现 apis** 说明可能是 getattr 之类的搬运。
        if any(
            isinstance(sub, ast.Name) and sub.id == "apis"
            for sub in ast.walk(value)
            if sub is not value
        ) and not _is_api_rooted_call(value):
            return ""
    return None


def _is_api_rooted_call(node: ast.Call) -> bool:
    """`apis.x.y(...)` —— 接收者是 apis 属性链的调用，它的**结果**是数据。"""

    func = node.func
    while isinstance(func, ast.Attribute):
        func = func.value
    return isinstance(func, ast.Name) and func.id == "apis"


def _collect_bindings(tree: ast.AST) -> _Bindings:
    """一次遍历收集「哪些名字是本 batch 自己绑定的」。

    绑定来源不止赋值：`def` / `class` / `for` / `with ... as` / `except ... as` /
    推导式的目标变量都会引入名字。少认一种，那个名字就会被当成「不知道是什么」
    ⇒ unknown ⇒ 整批回退。

    ⚠️ **函数参数不算**：`def helper(api): api.login()` 里的 `api` 完全可能就是
    API 对象。判不出来时不放宽（fail closed）。
    """

    bindings = _Bindings()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.assigned.add(node.name)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bindings.assigned.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bindings.assigned.update(_target_names(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bindings.assigned.add(node.name)
        elif isinstance(node, ast.comprehension):
            bindings.assigned.update(_target_names(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                bindings.modules[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            for alias in node.names:
                bindings.imported_functions[alias.asname or alias.name] = module
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            value = node.value
            path = _api_object_path(value)
            for target in targets:
                for name in _target_names(target):
                    bindings.assigned.add(name)
                    if path is not None:
                        bindings.api_aliases[name] = path
    return bindings


_METHOD_TO_CATEGORY = {
    "GET": "read_only",
    "HEAD": "read_only",
    "OPTIONS": "read_only",
    "POST": "declared_write",
    "PUT": "declared_write",
    "PATCH": "declared_write",
    "DELETE": "declared_write",
}
_WRITE_CATEGORIES = {"declared_write", "opaque_exec", "verify_exec"}


class AppWorldTraceOffloadError(RuntimeError):
    """Raised when AppWorld/TraceOffload integration cannot continue safely."""


def _identifier(value: Any, fallback: str = "unknown") -> str:
    text = str(value) if value is not None else fallback
    text = "".join(ch if (ch.isalnum() or ch in "._:-") else "-" for ch in text)
    text = text.strip("-._:") or fallback
    return text[:128]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return _json_safe(asdict(value))
        except Exception:
            pass
    return str(value)


def _content_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(value)


def _native_usage_counts(message: Mapping[str, Any]) -> tuple[int | None, int | None, float | None]:
    usage = message.get("standardized_usage")
    tokens = getattr(usage, "tokens", None)
    cost = getattr(usage, "cost", None)
    input_tokens = None
    output_tokens = None
    if tokens is not None:
        raw_input = getattr(tokens, "input_cache_hit_or_miss", None)
        raw_output = getattr(tokens, "output", None)
        if isinstance(raw_input, (int, float)):
            input_tokens = int(raw_input)
        if isinstance(raw_output, (int, float)):
            output_tokens = int(raw_output)
    total_cost = None
    if cost is not None:
        raw_cost = getattr(cost, "total", None)
        if isinstance(raw_cost, (int, float)):
            total_cost = float(raw_cost)
    return input_tokens, output_tokens, total_cost


def _empty_standardized_usage() -> Any:
    try:
        from appworld_agents.code.common.usage_tracker import Usage

        return Usage()
    except Exception:
        return None


def _message_tool_calls(message: Mapping[str, Any]) -> tuple[ModelMessageToolCall, ...]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise ValueError("message tool_calls must be a list")
    converted: list[ModelMessageToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise ValueError("message tool call must be a mapping")
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("message tool call requires an OpenAI-style function object")
        name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        call_id = raw_call.get("id", raw_call.get("call_id"))
        argument_error = None
        arguments: Any = raw_arguments
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                argument_error = f"Error parsing tool call arguments: {error}."
                arguments = {}
        if not isinstance(name, str) or not isinstance(call_id, str):
            raise ValueError("message tool call requires id and name")
        if not isinstance(arguments, Mapping):
            arguments = {}
        converted.append(
            ModelMessageToolCall(
                tool_call_id=call_id,
                tool_name=name,
                arguments=dict(arguments),
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
                argument_error=argument_error,
            )
        )
    return tuple(converted)


def _to_model_request(
    messages: list[dict[str, Any]],
    parameters: ModelParameters,
    *,
    tool_definitions: tuple[ToolDefinition, ...] = (),
    native_payload: Any = None,
) -> ModelRequest:
    converted: list[ModelMessage] = []
    for message in messages:
        role = _ROLE_MAP.get(str(message.get("role", "")))
        if role is None:
            raise ValueError(f"unsupported message role: {message.get('role')!r}")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError("TraceOffload AppWorld integration requires string/list/null content")
        reasoning_content = message.get("reasoning_content")
        name = message.get("name")
        tool_call_id = message.get("tool_call_id")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise ValueError("reasoning_content must be a string when supplied")
        if reasoning_content is not None and role != MessageRole.ASSISTANT:
            raise ValueError("reasoning_content is only valid on assistant messages")
        converted.append(
            ModelMessage(
                role=role,
                content=content,
                reasoning_content=reasoning_content,
                name=name if isinstance(name, str) else None,
                tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                tool_calls=_message_tool_calls(message),
            )
        )
    if not converted:
        raise ValueError("ModelRequest requires at least one message")
    return ModelRequest(
        request_version="1.0",
        messages=tuple(converted),
        native_payload=native_payload,
        tool_definitions=tool_definitions,
        model_parameters=parameters,
        request_scope=RequestScope(
            data_policy_ref=VersionedRef(kind="policy", id="appworld-simplified", version="v1")
        ),
    )


def _native_payload_snapshot(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control_at: int | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "messages": _json_safe(messages),
        "tools": _json_safe(tools),
        "cache_control_at": cache_control_at,
        "kwargs": _json_safe(kwargs),
    }


@dataclass
class PendingToolPhase:
    """Tool phase expected after an accepted model response."""

    tool_call_id: str
    expected_codes: tuple[str, ...]


def _pending_from_response(response: ModelResponse | None) -> PendingToolPhase | None:
    if response is None or not response.tool_calls:
        return None
    for call in response.tool_calls:
        if call.tool_name != BATCH_TOOL_NAME:
            continue
        codes = call.arguments.get("codes")
        if isinstance(codes, list):
            return PendingToolPhase(
                tool_call_id=call.tool_call_id,
                expected_codes=tuple(str(code) for code in codes),
            )
    return None


# ---------------------------------------------------------------------------
# Code extraction / API-call classification
# ---------------------------------------------------------------------------


def _extract_react_code(agent: Any, native_message: Mapping[str, Any]) -> list[str] | None:
    extractor = getattr(agent, "extract_code_and_fix_content", None)
    if not callable(extractor):
        return None
    content = _content_text(native_message.get("content")) or ""
    try:
        code, _fixed = extractor(content)
    except Exception:
        return None
    return [code] if isinstance(code, str) and code.strip() else None


def _extract_full_code(agent: Any, native_message: Mapping[str, Any]) -> list[str] | None:
    content = _content_text(native_message.get("content")) or ""
    try:
        from appworld.common.code_tools import remove_code_blocks

        code = remove_code_blocks(content)
    except Exception:
        return None
    return [code] if isinstance(code, str) and code.strip() else None


def _extract_function_calling_code(
    agent: Any, native_message: Mapping[str, Any], world: Any
) -> list[str] | None:
    raw_calls = native_message.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    separator = str(getattr(agent, "app_api_separator", "__"))
    limit = getattr(world, "max_api_calls_per_interaction", None)
    if isinstance(limit, int) and limit > 0:
        raw_calls = raw_calls[:limit]
    codes: list[str] = []
    for tool_call in raw_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        function_name = function.get("name")
        if not isinstance(function_name, str) or function_name.count(separator) != 1:
            continue
        app_name, api_name = function_name.split(separator, 1)
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments_str = str(json.loads(raw_arguments))
        except (json.JSONDecodeError, TypeError):
            arguments_str = "{}"
        codes.append(f"print({app_name}{separator}{api_name}(**{arguments_str}))")
    return codes or None


def _default_code_builder(
    agent: Any, native_message: Mapping[str, Any], world: Any
) -> list[str] | None:
    """Rebuild the exact code list the native Simplified Agent will execute."""

    if "error" in native_message:
        return None
    class_name = agent.__class__.__name__.lower()
    if hasattr(agent, "extract_code_and_fix_content"):
        return _extract_react_code(agent, native_message)
    if "functioncalling" in class_name or "function_calling" in class_name:
        return _extract_function_calling_code(agent, native_message, world)
    return _extract_full_code(agent, native_message)


class AppWorldCodeEffectAnalyzer:
    """Conservative Python/AST effect analyzer for AppWorld generated code."""

    def __init__(self) -> None:
        self._method_by_api: dict[str, str] = {}
        self._world: Any = None

    def bind_world(self, world: Any) -> None:
        self._world = world
        self._method_by_api = {}
        task = getattr(world, "task", None)
        docs = getattr(task, "api_docs", None)
        if docs is None:
            _LOGGER.warning(
                "appworld analyzer: world.task.api_docs 取不到 ⇒ 本 world 的 "
                "**每一条 apis.* 调用都会判成 unknown**（_method_by_api 为空）。"
                "后果：executor 的准入闸无条件拒 unknown ⇒ 整个 SLM 批次回退大模型。"
                "world=%s task=%s",
                type(world).__name__,
                type(task).__name__,
            )
            return
        try:
            for app_name in docs:
                apis = docs[app_name]
                for api_name in apis:
                    doc = apis[api_name]
                    if isinstance(doc, Mapping):
                        method = str(doc.get("method", "GET")).upper()
                    else:
                        method = "GET"
                    self._method_by_api[f"{app_name}.{api_name}"] = method
        except Exception as error:
            # ⚠️ 这里**清空**已建好的映射。半个 map 比空 map 更坏：一部分
            # apis.* 能过、一部分不能，看起来像"某些 API 有问题"，而不是
            # "文档绑定失败了" —— 而那两种情况的处置完全相反。
            self._method_by_api = {}
            _LOGGER.warning(
                "appworld analyzer: 读 api_docs 失败 ⇒ _method_by_api 被清空，"
                "**每一条 apis.* 调用都会判成 unknown** ⇒ 整个 SLM 批次回退大模型。"
                "原始错误：%r",
                error,
            )

    def _verdict_for_method(self, method: str, reason: str) -> EffectVerdict:
        category = _METHOD_TO_CATEGORY.get(method.upper(), "unknown")
        if category == "unknown":
            return _unknown_verdict(f"{reason}:{method}")
        scope = (
            EffectScope.READ_ONLY
            if category == "read_only"
            else EffectScope.REVERSIBLE_WRITE
        )
        return EffectVerdict(
            category=category,
            scope=scope,
            write_targets=(reason,) if category != "read_only" else (),
            requires_checkpoint=category != "read_only",
            confidence="certain",
            reason=f"{reason}:{method}",
            grammar_version="appworld-python-ast",
            policy_version="appworld-code-v1",
        )

    def _classify_call(self, node: ast.Call, bindings: _Bindings) -> EffectVerdict:
        """一个调用是写还是读 —— **判据是数据，不是名单**。

        ```
        写   ⟺ 调用了 API，且该 API 在 api_docs 里的 method 非只读
        只读 ⟸ 不是 API 调用（分析器自己的模型：只有 apis/requester 能触达世界）
        ```

        `api_docs` 由每个 task 自带、有限、可枚举 —— 这就是「泛化」的来源：
        不靠我们维护一张方法名表，靠 AppWorld 自己的 API 目录。

        剩下的三种「认不出接收者是谁」才保持 unknown：

        * API 名查不到（可能拼错、也可能文档没绑定上）
        * 接收者是一个**从没被赋值过**的名字（函数参数、外部注入）—— 它可能是 API
        * 调用了**不纯的模块**（`os` / `pathlib` / `io`）—— 不是 API，但能碰文件系统
        """

        func = node.func
        dotted = _dotted_name(func)

        # --- apis.<app>.<api>(...) 以及 `s = apis.spotify; s.<api>(...)` ---
        api_path = self._api_path_of(func, bindings)
        if api_path is not None:
            method = self._method_by_api.get(api_path)
            if method is None:
                return _unknown_verdict(f"unknown_api:{api_path}")
            return self._verdict_for_method(method, f"api:{api_path}")

        # --- requester.<method>(...) ---
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "requester"
        ):
            method = func.attr.upper()
            if method not in _METHOD_TO_CATEGORY:
                return _unknown_verdict(f"unknown_requester_method:{method}")
            return self._verdict_for_method(method, f"requester:{method}")

        # --- app__api(...)（直接函数形式）---
        if isinstance(func, ast.Name) and func.id.count("__") == 1:
            app_name, api_name = func.id.split("__", 1)
            name = f"{app_name}.{api_name}"
            method = self._method_by_api.get(name)
            if method is None:
                return _unknown_verdict(f"unknown_direct_api:{name}")
            return self._verdict_for_method(method, f"direct:{name}")

        # --- 模块调用：只有**纯**模块才放行 ---
        module = self._module_root_of(func, bindings)
        if module is not None:
            if module in _PURE_MODULES:
                return _read_verdict(f"pure_module_call:{dotted}")
            return _unknown_verdict(f"impure_module_call:{dotted}")

        # --- 裸名：内建函数、或从纯模块 import 进来的函数 ---
        if isinstance(func, ast.Name):
            if func.id in _SAFE_CALL_NAMES:
                return _read_verdict(f"safe_call:{func.id}")
            imported_from = bindings.imported_functions.get(func.id)
            if imported_from is not None and imported_from in _PURE_MODULES:
                return _read_verdict(f"pure_import:{func.id}")
            if imported_from is not None:
                return _unknown_verdict(f"impure_import:{func.id}")
            # 本 batch 自己 `def` 出来的函数 —— 它的副作用由**函数体里那些调用**
            # 各自判定（`ast.walk` 会走到），这里不必再保守。
            if func.id in bindings.assigned:
                return _read_verdict(f"local_function:{func.id}")
            return _unknown_verdict(f"unknown_call:{func.id}")

        # --- 本地接收者：接收者一定是**本 batch 内赋值过**的名字（或调用结果）---
        if self._receiver_is_local(func, bindings):
            return _read_verdict(f"local_call:{dotted}")
        return _unknown_verdict(f"unknown_call:{dotted or type(func).__name__}")

    def _api_path_of(self, func: ast.AST, bindings: _Bindings) -> str | None:
        """接收者是 `apis` 或某个 API 别名时，返回它在 `apis` 下的路径。"""

        chain: list[str] = []
        node: ast.AST = func
        while isinstance(node, ast.Attribute):
            chain.append(node.attr)
            node = node.value
        if not chain or not isinstance(node, ast.Name):
            return None
        attrs = ".".join(reversed(chain))
        if node.id == "apis":
            return attrs
        alias_path = bindings.api_aliases.get(node.id)
        if alias_path is None:
            return None
        return f"{alias_path}.{attrs}" if alias_path else attrs

    def _module_root_of(self, func: ast.AST, bindings: _Bindings) -> str | None:
        """接收者是不是一个 import 进来的模块（`math.sqrt` / `datetime.now`）。"""

        node: ast.AST = func
        while isinstance(node, ast.Attribute):
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        if node.id in bindings.modules:
            return bindings.modules[node.id]
        if node.id in _PURE_MODULES or node.id in _IMPURE_MODULES:
            return node.id
        return None

    @staticmethod
    def _receiver_is_local(func: ast.AST, bindings: _Bindings) -> bool:
        """接收者指的是本地数据吗？

        `songs.extend(...)` 里 `songs` **在本 batch 内被赋值过** ⇒ 本地。
        从没赋值过的名字（函数参数、外部注入）返回 False —— 它可能是 API 对象，
        必须保持 unknown（fail closed）。
        """

        node: ast.AST = func
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id in bindings.assigned
        # 接收者是调用结果 / 字面量 / 下标 —— AppWorld 的 API 不返回 API 对象，
        # 所以这些一律是数据。
        return isinstance(node, (ast.Call, ast.Subscript, ast.Constant, ast.List, ast.Dict, ast.Tuple))

    def analyze(self, code: str, cwd: str = "", workspace_root: str = "") -> EffectVerdict:
        del cwd, workspace_root
        if not code or not code.strip():
            return _read_verdict("empty_code")
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return _unknown_verdict(f"python_syntax_error:{error.msg}")
        bindings = _collect_bindings(tree)
        verdicts: list[EffectVerdict] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                verdicts.append(self._classify_call(node, bindings))
        if not verdicts:
            return _read_verdict("no_effectful_calls")
        merged = verdicts[0]
        for verdict in verdicts[1:]:
            merged = _merge_verdicts(merged, verdict)
        return merged


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _read_verdict(reason: str) -> EffectVerdict:
    return EffectVerdict(
        category="read_only",
        scope=EffectScope.READ_ONLY,
        confidence="certain",
        reason=reason,
        grammar_version="appworld-python-ast",
        policy_version="appworld-code-v1",
    )


def _unknown_verdict(reason: str) -> EffectVerdict:
    """判不出来 —— **按写处理**（要建 checkpoint），不是按拒处理。

    判不出来 ≠ 做不了。AppWorld 的 checkpoint 是 ``world.save_state()`` ——
    **整个世界状态**，所以任何触达世界的改动都能被回滚。于是「未知」的正确处置是
    「建快照 → 执行 → 出事就回滚」，而不是「不执行、整批回退大模型」。

    ⚠️ ``category`` 保持 ``"unknown"``，只把 ``scope`` 改成写。这是刻意的：
    ``category`` 是**事实**（我们确实不知道这是什么），
    ``scope`` 是**处置**（当作可回滚的写来做）。改掉事实，M4 / 审计 /
    误判统计就再也看不见这一类了。

    ⇒ 于是判据变干净：**只有明确的读操作不建 checkpoint**。
    ``executor`` 的准入闸认得这个组合 —— 它按 ``scope`` 建快照，
    而 ``category`` 仍是 unknown 供事后统计。

    ⚠️ 这一条**只在 AppWorld 侧**成立（那里快照覆盖一切）。通用执行器
    （SWE-bench 那条路）对 ``scope=UNKNOWN`` 仍然是拒绝 ——
    见 ``tests/components/test_executor.py::TestUnknownScopePolicy``。
    """

    return EffectVerdict(
        category="unknown",
        scope=EffectScope.REVERSIBLE_WRITE,
        risk_flags=frozenset({reason}),
        confidence="unknown",
        reason=reason,
        grammar_version="appworld-python-ast",
        policy_version="appworld-code-v1",
    )


def _merge_verdicts(left: EffectVerdict, right: EffectVerdict) -> EffectVerdict:
    rank = {
        "read_only": 0,
        "declared_write": 1,
        "opaque_exec": 2,
        "verify_exec": 2,
        "external": 3,
        "irreversible": 4,
        "unknown": 5,
    }
    worst = max((left, right), key=lambda item: rank.get(item.category, 5))
    if worst.category == "unknown":
        return worst
    if left.category == "unknown" or right.category == "unknown":
        return left if left.category == "unknown" else right
    category = worst.category
    return EffectVerdict(
        category=category,
        scope=EffectScope.READ_ONLY if category == "read_only" else EffectScope.REVERSIBLE_WRITE,
        write_targets=left.write_targets + right.write_targets,
        risk_flags=left.risk_flags | right.risk_flags,
        requires_checkpoint=category != "read_only",
        confidence="certain",
        reason=worst.reason,
        grammar_version="appworld-python-ast",
        policy_version="appworld-code-v1",
    )


# ---------------------------------------------------------------------------
# Tool proxy / runtime binding
# ---------------------------------------------------------------------------


#: AppWorld 报错文案的特征词（``environment.py:1009-1083`` **唯一**的失败信号）。
#: ⚠️ **这不是主判据** —— 只在结构化信号覆盖不到时兜底，理由见下。
_FALLBACK_FAILURE_MARKERS = (
    "Execution failed. Traceback:",
    "Execution timed out",
)


def _classify_batch_failure(
    outputs: list[Any],
    events: list[dict[str, Any]],
    code_count: int,
) -> tuple[bool, str]:
    """一批 code 是否失败，以及用了哪条判据（D8）。

    返回 ``(failed, source)``，``source`` ∈ ``{"structured", "fallback"}``。

    **主判据是结构化的**：``world.shell.run_cell`` 返回 IPython 的 ``ExecutionResult``，
    它的 ``.success`` 是 AppWorld 内部真正用来分流成功/失败的依据
    （``environment.py:1049`` 的 ``elif result.success``）。包装它拿到的判定，
    AppWorld 改文案也**不会**失效。超时同样结构化：事件 ``completed=False``。

    **退路**只覆盖"根本没走到 ``run_cell``"的 code —— 那些路径没有任何结构化信号：
    空代码（``"No code available to execute."``）、语法安全检查提前返回、
    以及远端环境（``world.shell is None``，包装不上）。
    这时才参考 AppWorld 的文案；并且**如实报告用了退路**，让文案漂移可被审计发现。
    """

    structured_failed = any(
        event.get("completed") is False or event.get("success") is False
        for event in events
    )
    if len(events) >= code_count:
        return structured_failed, "structured"
    fallback_failed = any(
        isinstance(output, str)
        and any(marker in output for marker in _FALLBACK_FAILURE_MARKERS)
        for output in outputs
    )
    return (structured_failed or fallback_failed), "fallback"


def _checkpoint_digest(directory: Path) -> str:
    """checkpoint 目录内容的确定性指纹（文件名 + 内容，排序后哈希）。

    用于 D2/D7 的**真校验**：``save_state`` 的产物是确定性的，所以"当前状态的指纹
    等于 checkpoint 的指纹"是一个可验证的命题，而不是"``load_state`` 没抛异常"。
    """

    digest = hashlib.sha256()
    for entry in sorted(directory.iterdir()):
        if entry.is_file():
            digest.update(entry.name.encode("utf-8"))
            digest.update(entry.read_bytes())
    return digest.hexdigest()


class AppWorldToolProxy(CallableToolProxy):
    """CallableToolProxy with AppWorld DB checkpoint/restore callbacks.

    ``supports_command_observation = False``（诊断报告 D5）：AppWorld 唯一的工具是
    ``execute_code_batch(codes)``（Python 代码批次），**不是 shell**。executor 的 L2
    越界观察会构造一个 ``bash`` action（``arguments={"command": "git status
    --porcelain"}``），而本 proxy 的 execute 回调读的是 ``arguments["codes"]`` ——
    取不到 key 就退化成空批次，``batch_execute([])`` 返回空、``returncode=0``，
    于是被记成"观测成功、状态无变化"。

    后果是 L2 **静默失效**：``effect_misclassification``（只读却写了东西）在 AppWorld
    上永远发现不了。显式声明不支持 -> executor 返回 ``_L2_NOT_AVAILABLE``（跳过检查），
    这才是诚实的降级。

    ``commits_on_tool_failure = False``：AppWorld 一个 batch 是**一次性生成的 2~14 行
    代码**（实测 6620 条轨迹 / 97335 次调用中，**失败的写 batch 100% 多行**），batch 内
    靠前的写已落盘 —— "FAILED 也 commit" 会把这种部分提交留住。声明后 FAILED 的写
    batch 走既有拒绝路径：rollback -> ``verification_failed`` -> Coordinator 的
    ``_should_fallback``。**只读不受影响**（不建 checkpoint，没有可回滚的东西）。
    """

    #: executor 用 ``getattr(..., True)`` 探测；只有显式 False 才跳过 L2 观察。
    supports_command_observation = False

    #: executor 用 ``getattr(..., True)`` 探测；只有显式 False 才让工具失败阻断 commit。
    commits_on_tool_failure = False

    #: 快照覆盖范围 —— executor 据此决定能不能执行 ``opaque_exec``（分析器看不懂的代码）。
    #:
    #: ``world_state``：本 proxy 的 checkpoint 是 ``world.save_state()``，保存的是
    #: **整个世界状态**。而 agent 的代码跑在 AppWorld 沙箱里（``read_only_open``
    #: 让 ``open()`` 只给读模式，外加模块白名单、``os.read/os.write/os.close``
    #: 黑名单）—— **它写不了 OS 文件系统**；它的全部写入都经过 ``apis.*``，
    #: 那正是 world 的一部分。
    #:
    #: ⇒ 「快照覆盖代码能触达的一切」成立 ⇒ opaque 代码可以执行。
    #:
    #: ⚠️ 必须声明在 proxy 自己身上，不能靠调用方传参：``ControlledExecutor``
    #: 有两个构造点，只改一个的话另一个会静默回落到「不允许」——跑得通、
    #: 日志干净、数字没变，而原因只是改动没生效。
    state_coverage = "world_state"

    def __init__(self, runtime: "AppWorldTraceOffloadRuntime") -> None:
        self._runtime = runtime
        self._state_ids: dict[str, str] = {}
        #: transaction_id -> checkpoint 内容指纹（D2/D7 的校验依据）。
        #: 成功 commit 时**丢弃**（状态合法地前进了，再比对会误判）；
        #: 成功 rollback / 回滚失败时**保留**（前者用于复核，后者是唯一证据）。
        self._checkpoint_digests: dict[str, str] = {}
        super().__init__(
            execute=lambda arguments, cwd: runtime.execute_codes(
                [str(code) for code in arguments.get("codes", [])]
            ),
            effect_scope=self._effect_scope,
            begin_transaction=self._begin_transaction,
            commit=self._commit_transaction,
            rollback=self._rollback_transaction,
            resume_check=self._resume_check,
        )

    def _effect_scope(self, action: ToolAction) -> EffectScope:
        codes = action.arguments.get("codes", []) if isinstance(action.arguments, Mapping) else []
        if not isinstance(codes, list):
            return EffectScope.UNKNOWN
        analyzer = self._runtime.analyzer
        if analyzer is None:
            return EffectScope.UNKNOWN
        return analyzer.analyze("\n".join(str(code) for code in codes)).scope

    def _state_id(self, transaction_id: str) -> str:
        digest = hashlib.sha256(transaction_id.encode("utf-8")).hexdigest()[:24]
        return f"traceoffload-{digest}"

    def _begin_transaction(self, transaction_id: str, action: ToolAction) -> bool:
        del action
        world = self._runtime.world
        if world is None:
            return False
        state_id = self._state_id(transaction_id)
        try:
            world.save_state(state_id=state_id)
        except Exception:
            return False
        # D6：把 checkpoint 复制到我们自己的目录。AppWorld 把 checkpoint 放在
        # ``output_directory`` 之下，而 ``initialize()`` 会 ``rmtree(output_directory)``
        # —— 同 task 再构造一个 world 就会把它删干净。副本放在这个目录之外，
        # 所以能跨 world 实例存活。
        self._mirror_after_save(state_id)
        directory = self._checkpoint_directory(state_id)
        if directory is not None and directory.is_dir():
            try:
                self._checkpoint_digests[transaction_id] = _checkpoint_digest(directory)
            except OSError:
                pass
        self._state_ids[transaction_id] = state_id
        return True

    def _checkpoint_root(self) -> Path:
        """我们自己保存 checkpoint 副本的根目录（在 ``output_directory`` 之外）。"""

        configured = getattr(self._runtime, "checkpoint_root", None)
        if configured:
            return Path(str(configured))
        return Path(tempfile.gettempdir()) / "traceoffload-appworld-checkpoints" / str(
            getattr(self._runtime, "workflow_run_id", "workflow-unknown")
        )

    def _mirror_directory(self, state_id: str) -> Path:
        return self._checkpoint_root() / state_id

    def _checkpoint_directory(self, state_id: str) -> Path | None:
        """AppWorld 自己的 checkpoint 目录（``load_state`` 只认这个路径）。"""

        world = self._runtime.world
        if world is None:
            return None
        root = getattr(world, "output_checkpoints_directory", None)
        if not root:
            return None
        return Path(root) / state_id

    def _mirror_after_save(self, state_id: str) -> bool:
        """``save_state`` 之后把 checkpoint 复制到我们自己的目录。"""

        source = self._checkpoint_directory(state_id)
        if source is None or not source.is_dir():
            return False
        target = self._mirror_directory(state_id)
        try:
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        except OSError:
            return False
        return True

    def _restore_mirror(self, state_id: str) -> bool:
        """``load_state`` 之前把副本拷回 AppWorld 的目录（它只认那个路径）。

        AppWorld 里的目录还在就不用拷（同一 world 内的正常回滚）。
        """

        target = self._checkpoint_directory(state_id)
        if target is None:
            return False
        if target.is_dir():
            return True
        source = self._mirror_directory(state_id)
        if not source.is_dir():
            return False
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        except OSError:
            return False
        return True

    def _state_fingerprint(self, state_id: str) -> str | None:
        """把 world 的**当前状态**存成一个临时 checkpoint，返回其内容指纹。

        只用公开的 `save_state`。探针实测：该产物是**确定性且基于内容**的
        （同一状态 → 同一指纹；状态变了 → 指纹变；`load_state` 之后 → 精确回到原指纹）。
        存完立即删掉临时目录，不留痕。
        """

        world = self._runtime.world
        if world is None:
            return None
        probe_id = f"{state_id}-verify"
        try:
            world.save_state(state_id=probe_id)
        except Exception:
            return None
        directory = self._checkpoint_directory(probe_id)
        try:
            if directory is None or not directory.is_dir():
                return None
            return _checkpoint_digest(directory)
        except OSError:
            return None
        finally:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)

    def _verify_against_checkpoint(self, transaction_id: str, state_id: str) -> bool | None:
        """当前状态是否真的等于该 transaction 的 checkpoint？

        ``True``/``False`` = 校验得出；``None`` = 没有指纹或算不出（无法判定）。
        """

        expected = self._checkpoint_digests.get(transaction_id)
        if expected is None:
            return None
        actual = self._state_fingerprint(state_id)
        if actual is None:
            return None
        return actual == expected

    def _delete_checkpoint(self, state_id: str) -> None:
        """删掉 AppWorld 里的和我们的副本（两处都要删，否则会累积）。"""

        path = self._checkpoint_directory(state_id)
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self._mirror_directory(state_id), ignore_errors=True)

    def _commit_transaction(self, transaction_id: str) -> bool:
        state_id = self._state_ids.pop(transaction_id, None)
        if state_id is not None:
            self._delete_checkpoint(state_id)
        # commit 之后状态合法地前进了，checkpoint 指纹不再有判定意义。
        self._checkpoint_digests.pop(transaction_id, None)
        return True

    def _rollback_transaction(self, transaction_id: str) -> str:
        from traceoffload.contracts import RollbackStatus

        world = self._runtime.world
        state_id = self._state_ids.get(transaction_id)
        if world is None or state_id is None:
            return RollbackStatus.UNKNOWN.value
        # 指纹预检（P9）：``load_state`` 中位 81.44ms，而指纹只要 0.83ms（便宜 98.5×）。
        # 写 batch 工具失败时约 **87.5% 是零副作用**的（实测：报错行在写行之前，
        # traceback 带行号可判），那些 batch 的当前状态与 checkpoint 完全一致，
        # 没必要真 ``load_state``。425 次失败按此可省 **87%** 的回滚耗时。
        #
        # **语义上安全**：指纹看不见的变化，``load_state`` 本来也撤销不了
        # （P8 探针实测 ``venmo.login`` 就属于这类）—— 所以预检不会漏掉任何
        # 回滚真能撤销的东西，不是拿正确性换速度。
        # 判定为 False / None 时不加严：交给下面的真回滚 + D7 复核。
        if self._verify_against_checkpoint(transaction_id, state_id) is True:
            self._state_ids.pop(transaction_id, None)
            self._delete_checkpoint(state_id)
            return RollbackStatus.SUCCEEDED.value
        # D6：AppWorld 的 checkpoint 可能已经被 rmtree 掉了，先从副本拷回去。
        if not self._restore_mirror(state_id):
            self._state_ids.pop(transaction_id, None)
            return RollbackStatus.FAILED.value
        try:
            world.load_state(state_id)
        except Exception:
            return RollbackStatus.FAILED.value
        # D7：`load_state` 没抛 ≠ 状态真的回去了。用 checkpoint 指纹做一次真校验；
        # 校验为 False（状态明确不等于 checkpoint）才算失败，并保留 _state_ids /
        # 指纹，让后续的 resume_check 还能复核。算不出（None）时不加严，保持原语义。
        if self._verify_against_checkpoint(transaction_id, state_id) is False:
            return RollbackStatus.FAILED.value
        self._state_ids.pop(transaction_id, None)
        self._delete_checkpoint(state_id)
        return RollbackStatus.SUCCEEDED.value

    def _resume_check(self, transaction_id: str) -> str:
        """恢复校验（D2）：**有 checkpoint 指纹就做真校验**，否则退回原判据。

        修复前这里是 ``self._runtime.resume_check(...)``，而后者忽略 transaction_id、
        只查 ``world.task is not None`` —— 于是 ``RecoveryCheck.passed`` 恒为 True，
        coordinator 会**直接弹掉 recovery lock**，回滚失败挡不住任何东西。

        现在：建过 checkpoint 的事务用指纹比对（当前状态 vs checkpoint）；比对不上就
        返回 FAILED，锁保留。没有指纹可比的（只读事务、已成功 commit 的写事务）才退回
        原来的"world 还活着"判据 —— 那是无法验证的情形，保持既有语义不回归。
        """

        from traceoffload.contracts import ResumeStatus

        verdict = self._verify_against_checkpoint(
            transaction_id, self._state_id(transaction_id)
        )
        if verdict is True:
            return ResumeStatus.SUCCEEDED.value
        if verdict is False:
            return ResumeStatus.FAILED.value
        return self._runtime.resume_check(transaction_id).value

    def finish_recovery(self, transaction_ids: tuple[str, ...]) -> bool:
        for transaction_id in transaction_ids:
            state_id = self._state_ids.pop(transaction_id, None)
            if state_id is not None:
                self._delete_checkpoint(state_id)
        return super().finish_recovery(transaction_ids)


class AppWorldLanguageModelGateway:
    """TraceOffload ModelGateway over AppWorld's native LanguageModel.generate."""

    def __init__(
        self,
        runtime: "AppWorldTraceOffloadRuntime",
        large_language_model: Any,
        small_language_model: Any,
    ) -> None:
        self._runtime = runtime
        self._models: dict[ModelTargetKind, Any] = {
            ModelTargetKind.LARGE_MODEL: large_language_model,
            ModelTargetKind.SMALL_MODEL: small_language_model,
        }
        self._raw_native_messages: dict[str, dict[str, Any]] = {}

    def raw_message(self, response_id: str) -> dict[str, Any] | None:
        raw = self._raw_native_messages.get(response_id)
        return copy.deepcopy(raw) if raw is not None else None

    def complete(
        self,
        target: ModelTarget,
        request: ModelRequest,
        *,
        model_attempt_id: str,
        deadline_at: AwareDatetime,
    ) -> ModelCallResult:
        del deadline_at
        model = self._models.get(target.kind)
        if model is None:
            raise AppWorldTraceOffloadError(f"no native LanguageModel for target {target.kind}")
        args = self._runtime.native_args_for(request)
        if args is None:
            raise AppWorldTraceOffloadError("missing native AppWorld request payload")
        started = _real_time()
        try:
            native_message = model.generate(
                messages=args["messages"],
                tools=args.get("tools"),
                cache_control_at=args.get("cache_control_at"),
                **args.get("kwargs", {}),
            )
        except Exception as error:
            raise AppWorldTraceOffloadError(
                f"native AppWorld LanguageModel.generate failed: {type(error).__name__}: {error}"
            ) from error
        latency_ms = max(0, round((_real_time() - started) * 1000))
        if not isinstance(native_message, Mapping):
            raise AppWorldTraceOffloadError("native AppWorld model returned a non-mapping message")
        native_message = dict(native_message)
        response_id = uuid4().hex
        input_tokens, output_tokens, cost = _native_usage_counts(native_message)
        error_message = native_message.get("error")
        codes = None if error_message is not None else self._runtime.build_codes(native_message)
        tool_calls: tuple[ModelResponseToolCall, ...] = ()
        if codes:
            tool_calls = (
                ModelResponseToolCall(
                    tool_call_id=f"appworld-step-{response_id[:12]}",
                    tool_name=BATCH_TOOL_NAME,
                    arguments={"codes": list(codes)},
                    raw_arguments=json.dumps({"codes": list(codes)}, ensure_ascii=True),
                ),
            )
        parse_status = ParseStatus.INVALID if error_message is not None else ParseStatus.PARSED
        finish_reason = (
            FinishReason.UNKNOWN
            if error_message is not None
            else FinishReason.TOOL_CALLS
            if tool_calls
            else FinishReason.STOP
        )
        native_json = {
            key: value for key, value in native_message.items() if key != "standardized_usage"
        }
        native_json = _json_safe(native_json)
        response = ModelResponse(
            response_id=response_id,
            model_attempt_id=_identifier(model_attempt_id, "attempt"),
            output_text=_content_text(native_message.get("content")),
            reasoning_content=(
                native_message.get("reasoning_content")
                if isinstance(native_message.get("reasoning_content"), str)
                else None
            ),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            parse_status=parse_status,
            native_message=native_json if isinstance(native_json, dict) else None,
        )
        self._raw_native_messages[response_id] = copy.deepcopy(native_message)
        return ModelCallResult(
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=cost,
        )


@dataclass
class AppWorldTraceOffloadRuntime:
    """Per-task runtime joining native AppWorld hooks to TraceOffload."""

    workflow_run_id: str
    agent: Any
    original_language_model: Any
    model_parameters: ModelParameters
    workflow_goal: WorkflowGoal
    deadline_minutes: float = 10.0
    snapshot_provider: Callable[[], Any] | None = None
    code_builder: Callable[[Any, Mapping[str, Any], Any], list[str] | None] | None = None

    coordinator: Coordinator | None = None
    adapter_session: AdapterSession | None = None
    gateway: AppWorldLanguageModelGateway | None = None
    analyzer: AppWorldCodeEffectAnalyzer | None = None
    tool_proxy: AppWorldToolProxy | None = None
    proxy: "AppWorldLanguageModelProxy | None" = None
    world: Any = None
    original_batch_execute: Callable[[list[str]], list[str]] | None = None
    original_run_cell: Callable[..., Any] | None = None
    #: D8：失败判定的**结构化**来源。``world.shell.run_cell`` 每次调用留一条
    #: ``{"completed": bool, "success": bool | None}``：
    #: ``completed=False`` = 这次调用没返回（超时）；``completed=True`` 时
    #: ``success`` 就是 IPython ``ExecutionResult.success``。
    _cell_events: list[dict[str, Any]] = field(default_factory=list)
    original_close: Callable[[], None] | None = None
    pending_tool_phase: PendingToolPhase | None = None
    lm_step_index: int = 0
    record_lm_step_index: bool = False
    live_request: ModelRequest | None = None
    live_args: dict[str, Any] | None = None
    native_snapshots: dict[int, dict[str, Any]] = field(default_factory=dict)
    _request_keepalive: list[ModelRequest] = field(default_factory=list)
    evaluation: Any = None
    #: ``_world_close`` 吞掉的异常（D3）。不静默丢弃，供 run summary 审计；
    #: 正常情况下应为 None。
    close_error: str | None = None
    #: 我们自己保存 checkpoint 副本的根目录（D6）。必须落在 AppWorld 的
    #: ``output_directory`` **之外** —— 那里会被 ``initialize()`` rmtree。
    #: None → 退回系统临时目录下的稳定位置。
    checkpoint_root: str | None = None

    # -- native hook registration -------------------------------------------------
    def bind_world(self, world: Any) -> None:
        self.world = world
        self.original_batch_execute = world.batch_execute
        self.original_close = world.close
        if self.analyzer is not None:
            self.analyzer.bind_world(world)
        world.batch_execute = MethodType(self._world_batch_execute, world)
        world.close = MethodType(self._world_close, world)
        # D8：失败判定的**结构化**来源。``AppWorld.execute`` 会把 IPython 的
        # ``ExecutionResult``（有 ``.success``）吞成一段人读的文字
        # （``environment.py:1009-1083``）—— 只靠那段文字判失败，AppWorld 一改措辞
        # 就**静默判成成功**，污染 local_success / post-check。
        # ``world.shell`` 与 ``run_cell`` 都是**公开名**，包装它们不违反
        # "不调私有 API"；远端环境 ``shell is None``，包装不上，退回文案匹配。
        shell = getattr(world, "shell", None)
        run_cell = getattr(shell, "run_cell", None)
        if shell is not None and callable(run_cell):
            self.original_run_cell = run_cell
            shell.run_cell = MethodType(self._shell_run_cell, shell)

    def _shell_run_cell(self, shell_self: Any, *args: Any, **kwargs: Any) -> Any:
        """包装 ``world.shell.run_cell``：记录 IPython 的结构化执行结果（D8）。

        **先登记事件、再调用**。这样两条路径区分得开：

        - 超时（``_shell_run_cell`` 里 ``timeout_call`` 抛出去、本调用永不返回）
          → 留下 ``completed=False`` 的事件；
        - ``execute`` 提前返回、**根本没调** ``run_cell``（空代码 / 语法安全检查）
          → 不留下任何事件。

        前者是结构化失败，后者才需要退回文案匹配（见 ``_classify_batch_failure``）。
        """

        del shell_self
        event: dict[str, Any] = {"completed": False, "success": None}
        self._cell_events.append(event)
        if self.original_run_cell is None:
            raise AppWorldTraceOffloadError("run_cell callback missing")
        result = self.original_run_cell(*args, **kwargs)
        event["completed"] = True
        event["success"] = bool(getattr(result, "success", False))
        return result

    def _world_batch_execute(self, world_self: Any, codes: list[str]) -> list[str]:
        del world_self
        return self.batch_execute(list(codes))

    def _world_close(self, world_self: Any, *args: Any, **kwargs: Any) -> None:
        """包装 ``world.close``：抓评估 → 调原方法 → 补偿它漏掉的清理。

        ⚠️ **为什么要吞异常（D3）**：``load_state()`` 内部会调 ``AppWorld.close_all()``，
        它对每个 time freezer 执行 ``stop()`` 且**不重建**；之后 ``close()`` 的
        ``_unset_datetime()`` 会对同一 freezer **再 stop 一次**，freezegun 抛
        ``AttributeError: '_freeze_time' object has no attribute 'fake_names'``。

        这个异常发生在 ``_unset_datetime()``，于是它后面的 ``self.apis.close()``
        **跑不到** —— API 连接泄漏。做法 B（约束：不调私有 API）只能吞掉异常，
        所以这里**额外补偿**一次 ``apis.close()``（公开属性）。

        吞掉的异常不静默丢弃，记进 ``self.close_error`` 供 run summary 审计。
        """

        try:
            self.evaluation = world_self.evaluate()
        except Exception:
            self.evaluation = None
        close_error: Exception | None = None
        if self.original_close is not None:
            try:
                self.original_close(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - D3 已知缺陷，见 docstring
                close_error = error
        if close_error is not None:
            # 原方法在 _unset_datetime() 处提前抛出，apis.close() 没被执行。
            close_apis = getattr(getattr(world_self, "apis", None), "close", None)
            if callable(close_apis):
                try:
                    close_apis()
                except Exception:  # noqa: BLE001 - 补偿失败不应再抛
                    pass
            self.close_error = f"{type(close_error).__name__}: {close_error}"

    # -- native payload / request registration -----------------------------------
    def register_native_call(
        self,
        request: ModelRequest,
        live_args: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        self.live_request = request
        self.live_args = live_args
        self.native_snapshots[id(request)] = snapshot
        self._request_keepalive.append(request)

    def clear_live_call(self) -> None:
        self.live_request = None
        self.live_args = None

    def native_args_for(self, request: ModelRequest) -> dict[str, Any] | None:
        if self.live_request is request and self.live_args is not None:
            return self.live_args
        snapshot = self.native_snapshots.get(id(request))
        if snapshot is not None:
            return snapshot
        payload = request.native_payload
        if isinstance(payload, Mapping) and "messages" in payload:
            return {
                "messages": copy.deepcopy(payload.get("messages")),
                "tools": copy.deepcopy(payload.get("tools")),
                "cache_control_at": payload.get("cache_control_at"),
                "kwargs": copy.deepcopy(payload.get("kwargs") or {}),
            }
        return None

    # -- code/tool helpers --------------------------------------------------------
    def build_codes(self, native_message: Mapping[str, Any]) -> list[str] | None:
        builder = getattr(self, "code_builder", None)
        fn: Callable[[Any, Mapping[str, Any], Any], list[str] | None]
        fn = builder if callable(builder) else _default_code_builder
        return fn(self.agent, native_message, self.world)

    def effect_scope_for_api_name(self, name: str) -> EffectScope:
        analyzer = self.analyzer
        if analyzer is None:
            return EffectScope.UNKNOWN
        normalized = name.replace("__", ".", 1)
        method = analyzer._method_by_api.get(normalized)
        if method in {"GET", "HEAD", "OPTIONS"}:
            return EffectScope.READ_ONLY
        if method is not None:
            return EffectScope.REVERSIBLE_WRITE
        return EffectScope.UNKNOWN

    def tool_definitions_for(
        self, tools: list[dict[str, Any]] | None
    ) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = [
            ToolDefinition(
                tool_name=BATCH_TOOL_NAME,
                schema_ref=VersionedRef(kind="tool_schema", id=BATCH_TOOL_NAME, version="v1"),
                description="Execute generated AppWorld Python code as one transaction.",
                effect_scope=EffectScope.REVERSIBLE_WRITE,
            )
        ]
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, Mapping):
                    continue
                function = tool.get("function")
                if not isinstance(function, Mapping):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                definitions.append(
                    ToolDefinition(
                        tool_name=_identifier(name),
                        schema_ref=VersionedRef(
                            kind="tool_schema",
                            id=_identifier(name),
                            version="v1",
                        ),
                        description=str(function.get("description", "")) or None,
                        effect_scope=self.effect_scope_for_api_name(name),
                    )
                )
        return tuple(definitions)

    # -- tool phase ---------------------------------------------------------------
    def _tool_output_with_effect_category(
        self, proxy_result: Any, codes: Sequence[str]
    ) -> dict[str, Any]:
        """给 ``tool_finished`` 的 result 补上**事前 L1 的效应类别**。

        用途：在线特征 ``prefix_n_A`` 要统计「前缀里**成功且是写**的工具调用数」，
        与离线口径对齐（``classfierTrain/aw3_build.py`` 的 ``tool_event_kind``：
        ``ok and is_write`` → ``"A"``）。

        这个类别是 **pre-check** 的判定，**不在** proxy 的 output dict 里
        （``output`` 只有 outputs/output/returncode/exception_info/...），所以在这里补。
        优先取记录上已有的（执行路径可能已标注），否则用 analyzer 现算；
        **两者都拿不到就不写这个键** —— 让下游按"未知"处理，不猜成只读也不猜成写。
        """

        enriched = dict(getattr(proxy_result, "output", None) or {})
        record = getattr(proxy_result, "record", None)
        category = getattr(record, "effect_category", None)
        if not category and self.analyzer is not None:
            try:
                category = self.analyzer.analyze(
                    "\n".join(str(code) for code in codes)
                ).category
            except Exception:
                category = None
        if category:
            enriched["effect_category"] = str(category)
        return enriched

    def execute_codes(self, codes: list[str]) -> dict[str, Any]:
        if self.original_batch_execute is None:
            return {
                "outputs": [],
                "output": "",
                "returncode": -1,
                "exception_info": "AppWorld world is not initialized",
            }
        # D8：清空上一次执行留下的事件，否则上一批的失败会污染这一批的判定。
        self._cell_events.clear()
        try:
            outputs = self.original_batch_execute(list(codes))
        except Exception as error:
            return {
                "outputs": [],
                "output": "",
                "returncode": -1,
                "exception_info": f"{type(error).__name__}: {error}",
            }
        failed, source = _classify_batch_failure(
            outputs, self._cell_events, len(codes)
        )
        return {
            "outputs": list(outputs),
            "output": "\n".join(str(output) for output in outputs),
            "returncode": 1 if failed else 0,
            "exception_info": "appworld_execution_failed" if failed else "",
            # D8：本次失败判定用了哪条判据（structured / fallback），供事后审计。
            # 出现 fallback 就说明有 code 没走到 run_cell，AppWorld 的文案被参考了。
            "failure_signal_source": source,
        }

    def batch_execute(self, codes: list[str]) -> list[str]:
        pending = self.pending_tool_phase
        if pending is None or not codes:
            if self.original_batch_execute is None:
                raise AppWorldTraceOffloadError("AppWorld world is not initialized")
            return self.original_batch_execute(list(codes))
        actual = tuple(str(code) for code in codes)
        if actual != pending.expected_codes:
            raise AppWorldTraceOffloadError(
                "AppWorld generated-code mismatch between model response and batch_execute"
            )
        if self.coordinator is None or self.adapter_session is None or self.tool_proxy is None:
            raise AppWorldTraceOffloadError("TraceOffload runtime is not fully initialized")
        action = ToolAction(
            tool_call_id=pending.tool_call_id,
            tool_name=BATCH_TOOL_NAME,
            arguments={"codes": list(actual)},
        )
        run_id = self.adapter_session.record_tool_started(
            {
                "tool_call_id": pending.tool_call_id,
                "tool_name": BATCH_TOOL_NAME,
                "codes": list(actual),
            }
        )
        transaction_id = self.coordinator.tool_transaction_id(
            self.workflow_run_id, pending.tool_call_id
        )
        try:
            proxy_result: ToolProxyResult = self.tool_proxy.execute(
                action, transaction_id=transaction_id
            )
        except Exception as error:
            self.adapter_session.record_tool_finished(run_id, error=error)
            raise
        self.adapter_session.record_tool_finished(
            run_id, self._tool_output_with_effect_category(proxy_result, actual)
        )
        self.coordinator.record_tool_proxy_result(self.workflow_run_id, proxy_result)
        message = {
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [
                    {
                        "tool_call_id": pending.tool_call_id,
                        "tool_name": BATCH_TOOL_NAME,
                        "command": "",
                    }
                ]
            },
        }
        fallback = self.coordinator.complete_tool_phase(
            self.workflow_run_id, message, [proxy_result.output]
        )
        if fallback is not None:
            self.adapter_session.queue_recovery_response(
                fallback,
                cost=self.coordinator.accepted_response_cost(self.workflow_run_id, fallback),
            )
            self.pending_tool_phase = _pending_from_response(fallback)
        else:
            self.pending_tool_phase = None
        if self.adapter_session.terminal_error is not None:
            raise AppWorldTraceOffloadError(self.adapter_session.terminal_error)
        outputs = proxy_result.output.get("outputs")
        return [str(output) for output in outputs] if isinstance(outputs, list) else []

    # -- recovery check -----------------------------------------------------------
    def resume_check(self, transaction_id: str) -> Any:
        from traceoffload.contracts import ResumeStatus

        del transaction_id
        if self.world is None:
            return ResumeStatus.UNKNOWN
        try:
            # A live, responsive AppWorld object after load_state is the minimum
            # observable proof for this first adapter.  A stronger sentinel can
            # be added without changing the native Agent contract.
            if getattr(self.world, "task", None) is None:
                return ResumeStatus.UNKNOWN
            return ResumeStatus.SUCCEEDED
        except Exception:
            return ResumeStatus.UNKNOWN

    # -- response conversion ------------------------------------------------------
    def message_from_response(self, response: ModelResponse) -> dict[str, Any]:
        if self.gateway is not None:
            raw = self.gateway.raw_message(response.response_id)
            if raw is not None:
                return raw
        message: dict[str, Any] = {"role": "assistant", "content": response.output_text}
        if response.native_message is not None:
            message.update(dict(response.native_message))
        if response.reasoning_content is not None:
            message["reasoning_content"] = response.reasoning_content
        if response.tool_calls:
            message.setdefault(
                "tool_calls",
                [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": call.raw_arguments
                            or json.dumps(call.arguments, ensure_ascii=True),
                        },
                    }
                    for call in response.tool_calls
                ],
            )
        usage = _empty_standardized_usage()
        if usage is not None:
            message.setdefault("standardized_usage", usage)
        return message


class AppWorldLanguageModelProxy:
    """Drop-in replacement for ``agent.language_model`` at generate() boundary."""

    def __init__(self, runtime: AppWorldTraceOffloadRuntime) -> None:
        self._runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime.original_language_model, name)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cache_control_at: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        runtime = self._runtime
        if runtime.coordinator is None or runtime.adapter_session is None or runtime.gateway is None:
            raise AppWorldTraceOffloadError("TraceOffload runtime is not fully initialized")
        if runtime.adapter_session.terminal_error is not None:
            raise AppWorldTraceOffloadError(runtime.adapter_session.terminal_error)
        recovery = runtime.adapter_session.consume_recovery_response()
        if recovery is not None:
            response, _cost = recovery
            runtime.pending_tool_phase = _pending_from_response(response)
            return runtime.message_from_response(response)
        native_payload = _native_payload_snapshot(messages, tools, cache_control_at, kwargs)
        request = _to_model_request(
            messages,
            runtime.model_parameters,
            tool_definitions=runtime.tool_definitions_for(tools),
            native_payload=native_payload,
        )
        live_args = {
            "messages": messages,
            "tools": tools,
            "cache_control_at": cache_control_at,
            "kwargs": kwargs,
        }
        snapshot = {
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "cache_control_at": cache_control_at,
            "kwargs": copy.deepcopy(kwargs),
        }
        runtime.register_native_call(request, live_args, snapshot)
        # ClassifierFeatureProfiler 的在线特征需要当前 LM span 序号。AppWorld
        # adapter 自身不维护 step_index；仅在 classifier_route 模式下以主 LM
        # 调用次数为 0-based k 写入 framework metadata，避免改变其他既有模式的
        # WorkflowContext / admission 行为。该调用不改变 Agent messages/prompt/action。
        if runtime.record_lm_step_index:
            runtime.adapter_session.update_framework_metadata(
                step_index=runtime.lm_step_index
            )
        runtime.lm_step_index += 1
        recovery_mode: Literal["normal", "rollback_recovery"] = (
            "rollback_recovery"
            if runtime.coordinator.recovery_directive(runtime.workflow_run_id) is not None
            else "normal"
        )
        workflow_context = runtime.adapter_session.begin_lm_invocation(
            request, recovery_mode=recovery_mode
        )
        snapshot_feedback = runtime.snapshot_provider() if runtime.snapshot_provider else None
        try:
            outcome = runtime.coordinator.orchestrate_episode(
                request,
                workflow_context,
                workflow_run_id=runtime.workflow_run_id,
                # ⚠️ **必须和 coordinator 的时钟同一个来源。**
                #
                # coordinator 那边（`Coordinator.__init__`）拿的是 runner 注入的
                # `clock`，AppWorld 那条路注的是 `real_utcnow`（穿透 freezegun）。
                # 这里如果写 `datetime.now(timezone.utc)`，拿到的是**冻结**时间
                # （AppWorld 的 world 把时间冻在剧情日期，如 2022 年），
                # 而 coordinator 的 `produced_at` 是**真实**时间（2026）：
                #
                #     caller_deadline_at(2022+10min) <= produced_at(2026)  ⇒ True
                #     ⇒ CoordinatorError("caller deadline has already passed")
                #
                # 实测踩过（0925）：加了 `clock=real_utcnow` 之后三个任务全崩在这。
                caller_deadline_at=real_utcnow()
                + timedelta(minutes=runtime.deadline_minutes),
                snapshot=snapshot_feedback,
            )
        except CoordinatorError as error:
            raise AppWorldTraceOffloadError(str(error)) from error
        finally:
            runtime.clear_live_call()
        if outcome.response is None:
            raise AppWorldTraceOffloadError(
                f"AppWorld episode ended without an accepted response: {outcome.status}"
            )
        if runtime.adapter_session.terminal_error is not None:
            raise AppWorldTraceOffloadError(runtime.adapter_session.terminal_error)
        runtime.pending_tool_phase = _pending_from_response(outcome.response)
        return runtime.message_from_response(outcome.response)




__all__ = [
    "AppWorldCodeEffectAnalyzer",
    "AppWorldLanguageModelGateway",
    "AppWorldLanguageModelProxy",
    "AppWorldToolProxy",
    "AppWorldTraceOffloadError",
    "AppWorldTraceOffloadRuntime",
    "PendingToolPhase",
]
