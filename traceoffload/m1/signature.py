"""动作签名规范化 —— `act_signature_unique` 与全部 `prog_*` 键的前置依赖。

签名的用途是回答「这两个动作是不是同一个动作」。它决定了：

    act_signature_unique             不同规范动作数
    prog_exact_action_repeat_count   规范动作完全重复数
    prog_same_category_diff_args     同类但参数不同的动作数
    prog_recent5_repeat_rate         最近五步动作重复率
    prog_recent5_new_action_rate     最近五步新动作比例
    prog_no_new_action_run           连续未产生新规范动作步数
    prog_last_action_similarity      最近两动作相似度

AppWorld 的动作是 agent 写在 ```python 代码块里的 `apis.<app>.<api>(...)` 调用。
一次工具执行里可能有多行调用，全部折进**一个**签名。

规范化的五个决定
----------------
1. **解析**：`ast.parse` 优先；解析失败回退到正则（逐字等于 `aw3_build._CALL_RE`）；
   再失败则整条标 `parse_failed`。
2. **键序**：kwargs **按键名排序**；位置参数保持原序（顺序本身是信号）。
3. **变量：保留名字但加标记**（`<name:token>`）。见下方「变量策略」。
4. **截断**：单值 ≤120 字符、最多 12 个 kwargs、字面量容器最多 8 个元素。
5. **折叠**：一次执行内多个调用按原序用 `|` 连接（**不排序** —— 调用顺序是信号）。

变量策略
--------
`FeatureConstants.signature_var_policy` 控制非字面量位置参数怎么渲染：

    "named"   默认。`<name:token>` / `<attr:page.text>` / `<call>` / `<expr>`
              保留变量名 —— 模型写出不同的变量名就是不同的代码，视为不同动作。
    "opaque"  `<name>` —— 所有变量一视同仁。

**为什么默认 named 而不是规格草图里的 opaque**：规格 §3.1.1.1 只规定"去掉运行时 ID"，
没有规定变量名也要抹平。而 AppWorld 是 ReAct agent，**每个 step 的代码是独立写的** ——
同一个变量名意味着模型选择了相同的写法，不同变量名意味着它换了写法。抹平会丢掉这个信号，
并且把"看起来一样其实是两次不同调用"的误判概率推高（`prog_exact_action_repeat_count` 虚高）。

两个方向都有代价，所以做成**预注册常量**而不是硬编码：policy 写进 manifest 的
`constants`，测试断言两个 policy 都可用，报告里跑一次敏感性对比。

已知边界（必须写进报告）
------------------------
**无法区分"同名变量、不同值"的两次调用。** 例如

    t = apis.venmo.login(...)["access_token"]
    apis.venmo.show_transactions(access_token=t)

若两个 step 各自登录成不同用户，签名仍相同 —— 因为值在运行期才知道，
静态分析拿不到。所以 `prog_exact_action_repeat_count` 在"每次重新取 token"
这类模式上会**偏保守**（把不同实体的同类调用也计成重复）。
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .constants import FeatureConstants

#: 逐字等于 `aw3_build._CALL_RE`（:87）。回退路径必须与旧口径同源，
#: 否则新老特征的"动作数"会对不上而没人发现。
CALL_RE = re.compile(r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")

#: 非字面量 AST 节点 → 占位符。
_NODE_KINDS: tuple[tuple[type, str], ...] = (
    (ast.Call, "call"),
    (ast.JoinedStr, "fstr"),
    (ast.Subscript, "sub"),
    (ast.Attribute, "attr"),
    (ast.Name, "name"),
    (ast.BinOp, "expr"),
    (ast.IfExp, "expr"),
    (ast.ListComp, "expr"),
    (ast.DictComp, "expr"),
    (ast.SetComp, "expr"),
    (ast.GeneratorExp, "expr"),
    (ast.Lambda, "expr"),
    (ast.Await, "expr"),
)


@dataclass(frozen=True)
class ParsedCall:
    """一次 `apis.<app>.<api>(...)` 调用。"""

    app: str
    api: str
    positional: tuple[str, ...] = ()
    kwargs: tuple[tuple[str, str], ...] = ()
    parse_failed: bool = False

    def render(self) -> str:
        """`app.api(pos, kw=<v>)` —— 特征里用的规范形式。"""
        parts = list(self.positional)
        parts.extend(f"{k}={v}" for k, v in self.kwargs)
        if self.parse_failed:
            parts.append("<unparsed>")
        return f"{self.app}.{self.api}({','.join(parts)})"

    @property
    def key(self) -> tuple[str, str]:
        return (self.app, self.api)


@dataclass
class SignatureResult:
    """一次工具执行的签名结果。"""

    signature: str
    parsed: bool
    calls: list[ParsedCall] = field(default_factory=list)


# --------------------------------------------------------------------------
# 字面量渲染

def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…<trunc>"


def _render_literal(node: ast.AST, constants: FeatureConstants) -> str | None:
    """字面量 → 规范化字符串。不是字面量返回 None。"""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return _truncate(json.dumps(v, ensure_ascii=False), constants.signature_value_truncate)
        if v is None or isinstance(v, bool):
            return json.dumps(v)
        if isinstance(v, (int, float)):
            # 归一：1 == 1.0 == 1e0。避免同一数值因字面写法不同被判成两个动作。
            f = float(v)
            return str(int(f)) if f.is_integer() else repr(f)
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        rendered: list[str] = []
        for element in node.elts:
            r = _render_literal(element, constants)
            if r is None:
                return None
            rendered.append(r)
        cap = constants.signature_list_cap
        if len(rendered) > cap:
            rendered = rendered[:cap] + ["<...>"]
        return "[" + ",".join(rendered) + "]"
    if isinstance(node, ast.Dict):
        if node.keys and any(k is None for k in node.keys):  # **{...} 展开
            return None
        pairs = []
        for k, val in zip(node.keys, node.values):
            rk = _render_literal(k, constants)  # type: ignore[arg-type]
            rv = _render_literal(val, constants)
            if rk is None or rv is None:
                return None
            pairs.append(f"{rk}:{rv}")
        pairs.sort()  # 字典键序在语义上无关，排序以稳定
        cap = constants.signature_list_cap
        return "{" + ",".join(pairs[:cap] + (["<...>"] if len(pairs) > cap else [])) + "}"
    return None


def _render_arg(node: ast.AST, constants: FeatureConstants) -> str:
    """一个实参的渲染：优先字面量，否则按节点种类给占位符。"""
    lit = _render_literal(node, constants)
    if lit is not None:
        return lit
    for cls, tag in _NODE_KINDS:
        if isinstance(node, cls):
            if tag == "name" and constants.signature_var_policy == "named":
                return f"<name:{node.id}>"  # type: ignore[attr-defined]
            if tag == "attr" and constants.signature_var_policy == "named":
                return f"<attr:{_attr_path(node)}>"
            return f"<{tag}>"
    return "<expr>"


def _attr_path(node: ast.AST) -> str:
    """`a.b.c` → `"a.b.c"`；中间出现非属性/非名字节点时截断。"""
    parts: list[str] = []
    cur: Any = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return "?"


def _api_call_parts(node: ast.AST) -> tuple[str, str] | None:
    """`apis.<app>.<api>(...)` 形状判定。命中则返回 `(app, api)`，否则 None。

    返回元组而不是布尔，是为了让调用方拿到类型收窄后的 `ast.Call`
    —— 否则每个字段访问都要 `# type: ignore`，反而掩盖真的类型错误。
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    inner = func.value
    if not isinstance(inner, ast.Attribute):
        return None
    if not (isinstance(inner.value, ast.Name) and inner.value.id == "apis"):
        return None
    return inner.attr, func.attr


# --------------------------------------------------------------------------
# 两条解析路径

def _calls_via_ast(code: str, constants: FeatureConstants) -> list[ParsedCall] | None:
    """AST 路径。语法错误返回 None（调用方转正则）。"""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, RecursionError):
        return None
    out: list[ParsedCall] = []
    for node in ast.walk(tree):
        parts = _api_call_parts(node)
        if parts is None or not isinstance(node, ast.Call):
            continue
        app, api = parts
        positional = tuple(_render_arg(a, constants) for a in node.args)
        kwargs: list[tuple[str, str]] = []
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs 展开
                kwargs.append(("**", "<expr>"))
            else:
                kwargs.append((kw.arg, _render_arg(kw.value, constants)))
        kwargs.sort(key=lambda kv: kv[0])
        kwargs = kwargs[: constants.signature_max_kwargs]
        out.append(ParsedCall(app=app, api=api, positional=positional, kwargs=tuple(kwargs)))
    return out


def _calls_via_regex(code: str) -> list[ParsedCall]:
    """正则回退 —— 只认 `apis.a.b(` 本身，参数一律标 `<unparsed>`。

    **故意做得粗糙**：回退路径的价值在于"至少知道调了哪个 API"，
    去猜括号配平只会引入新的静默错误。所以回退结果 `parse_failed=True`，
    下游可以据此把该动作排除在"重复检测"之外。
    """
    return [
        ParsedCall(app=m.group(1), api=m.group(2), parse_failed=True)
        for m in CALL_RE.finditer(code or "")
    ]


def api_calls_of_code(code: str, constants: FeatureConstants) -> tuple[list[ParsedCall], bool]:
    """抽取一次工具执行里的全部 API 调用。

    返回 `(calls, parsed_ok)`。`parsed_ok=False` 表示走的是正则回退，
    此时每个 call 的 `parse_failed=True`，**参数信息不可用**。
    """
    if not (code or "").strip():
        return [], True
    calls = _calls_via_ast(code, constants)
    if calls is not None:
        return calls, True
    return _calls_via_regex(code), False


def signature_inputs(code: str, constants: FeatureConstants) -> SignatureResult:
    """从代码块算签名。"""
    calls, ok = api_calls_of_code(code, constants)
    if not calls:
        return SignatureResult(signature="", parsed=ok, calls=[])
    # 一次执行内多个调用按原序连接 —— **不排序**，执行顺序本身是信号。
    sig = "|".join(c.render() for c in calls)
    return SignatureResult(
        signature=_truncate(sig, 800),
        parsed=ok and not any(c.parse_failed for c in calls),
        calls=calls,
    )


def jaccard_tokens(a: str | None, b: str | None) -> float | None:
    """两个签名的 token 级 Jaccard 相似度，供 `prog_last_action_similarity` 用。

    分词：按 `. , ( ) = [ ] { } | < >` 与非字母数字切，去重、去掉纯数字以外的短碎片。
    任一方为 None 返回 None（调用方标 missing）。
    """
    if a is None or b is None:
        return None
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


_SPLIT_RE = re.compile(r"[^0-9A-Za-z_一-鿿]+")


def _tokens(sig: str) -> set[str]:
    return {t for t in _SPLIT_RE.split(sig) if t}


__all__ = [
    "CALL_RE",
    "ParsedCall",
    "SignatureResult",
    "api_calls_of_code",
    "jaccard_tokens",
    "signature_inputs",
]
