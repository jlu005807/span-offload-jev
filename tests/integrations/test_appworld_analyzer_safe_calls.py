"""分析器的「纯调用」判定必须是**从 Python 自己派生**的，不能是手维护的短名单。

为什么单独一个文件
------------------
这不是某一条规则的细节，而是**分析器面对所有 AppWorld 任务时的默认行为**。
手维护的名单每漏一项，后果都一样且都不报错：

    漏项 → `_classify_call` 返回 `unknown_call:` → 准入闸无条件拒 `unknown`
         → **整个 SLM 批次回退大模型**

已经撞过两次，都是同一形状：

    `songs.extend(...)`     `<变量>.<方法>` 这一类没人管        （19/20 次拒绝）
    `next(iterator)`        `_SAFE_CALL_NAMES` 少一个内建函数   （剩下的 1/20）

两次都不是「某个任务特殊」：`extend` 那次散布在 **6 个不同任务**、变量名各不相同。
⇒ 所以修法不是补名字，是**把名单改成从 `builtins` / 内建类型派生**，
`dir()` 出来的就是全集，不会烂。

安全边界（下半部分）：派生出来的集合**必须排除**能做 I/O 或改变执行环境的内建
—— `open` / `eval` / `exec` / `__import__` / `setattr` …。这一半不能靠派生，
所以它被显式列出来，并且每条都有测试钉住。
"""

from __future__ import annotations

import builtins

import pytest

from traceoffload.contracts import EffectScope
from traceoffload.integrations import appworld as module
from traceoffload.integrations.appworld import AppWorldCodeEffectAnalyzer


class _FakeWorld:
    """一个 spotify + supervisor 的 world，和 `test_appworld_simplified` 同形。"""

    def __init__(self) -> None:
        self.task = type(
            "_Task",
            (),
            {
                "api_docs": {
                    "supervisor": {
                        "show_profile": {"method": "GET"},
                        "complete_task": {"method": "POST"},
                    },
                    "spotify": {
                        "login": {"method": "POST"},
                        "show_song": {"method": "GET"},
                    },
                }
            },
        )()


def _analyzer() -> AppWorldCodeEffectAnalyzer:
    analyzer = AppWorldCodeEffectAnalyzer()
    analyzer.bind_world(_FakeWorld())
    return analyzer


# --------------------------------------------------------------- 派生出来的那一半
def test_the_safe_call_set_is_derived_not_hand_picked() -> None:
    """名单必须是**派生**的：`builtins` 里所有公开且不在排除表里的名字都要在。

    这条是防回退的锚 —— 谁把它改回一份手写的短名单，这里立刻红。
    """

    expected = {
        name
        for name in dir(builtins)
        if not name.startswith("_")
        and name not in module._UNSAFE_BUILTINS
        and callable(getattr(builtins, name))
    }

    assert expected <= set(module._SAFE_CALL_NAMES), (
        f"这些内建函数没有进白名单：{sorted(expected - set(module._SAFE_CALL_NAMES))}"
    )


def test_every_public_builtin_is_classified() -> None:
    """行为口径的同一件事：逐个内建函数跑一遍，**一个都不许判 unknown**。

    上一条断言集合，这一条断言分析器**真的用上了**那个集合 ——
    集合对了但 `_classify_call` 没读它，是两种不同的坏法。
    """

    analyzer = _analyzer()
    unknown = []
    for name in sorted(dir(builtins)):
        if (
            name.startswith("_")
            or name in module._UNSAFE_BUILTINS
            or not callable(getattr(builtins, name))
        ):
            continue
        verdict = analyzer.analyze(f"{name}(x)")
        if verdict.category == "unknown":
            unknown.append((name, verdict.reason))

    assert not unknown, f"这些内建函数被判 unknown：{unknown}"


@pytest.mark.parametrize(
    "name",
    ["next", "iter", "any", "all", "map", "filter", "reversed", "frozenset", "bytes", "ord", "chr"],
)
def test_builtins_that_were_missing_are_now_covered(name: str) -> None:
    """逐个点名 —— 这些是**实际撞到过**或明显会撞到的，留作回归证据。"""

    assert _analyzer().analyze(f"{name}(x)").category != "unknown"


@pytest.mark.parametrize(
    "method",
    ["extend", "append", "insert", "pop", "sort", "reverse", "remove", "clear", "copy",
     "update", "get", "keys", "values", "items", "setdefault",
     "add", "discard", "union", "intersection",
     "split", "strip", "lower", "upper", "replace", "join", "startswith", "format"],
)
def test_local_container_methods_are_not_unknown(method: str) -> None:
    """`<局部变量>.<方法>()` —— 这一整类都不该判 unknown。

    ⚠️ 接收者**必须在本 batch 内被赋值过**，这是新判据的一部分：
    「不是 API 就只读」要成立，前提是那个名字确实是本地数据。
    从没被赋值过的名字（函数参数、外部注入）可能是 API 对象 ⇒ 保持 unknown。
    见 `test_never_assigned_receiver_stays_unknown`。
    """

    assert _analyzer().analyze(f"target = []\ntarget.{method}(x)").category != "unknown"


# ------------------------------------------------------------------ 安全的那一半
@pytest.mark.parametrize(
    "dangerous",
    ["open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
     "exit", "quit", "setattr", "delattr", "globals", "locals"],
)
def test_dangerous_builtins_are_never_whitelisted(dangerous: str) -> None:
    """派生**必须**排除这些 —— 它们能读写文件、改执行环境、或凭空取到全局对象。

    `open` 尤其重要：AppWorld 的沙箱虽然在**运行期**把 `open()` 降成只读，
    但分析器判的是**文本**，不能拿运行期的性质去放行。
    """

    assert dangerous in module._UNSAFE_BUILTINS
    assert dangerous not in module._SAFE_CALL_NAMES
    assert _analyzer().analyze(f"{dangerous}(x)").category == "unknown"


def test_aliased_api_object_still_cannot_hide_a_write() -> None:
    """别名（`s = apis.spotify`）**能解析** —— 但解析之后写仍然是写。

    旧实现靠「方法名在不在内建名单里」来间接防别名，于是两种情况都判 unknown：
    别名指向真 API（该解析成写）、别名指向本地方法（该放行）。新实现直接跟踪
    `s = apis.spotify` 这个赋值，所以两者都判对。
    """

    analyzer = _analyzer()

    # 文档里有 ⇒ 按文档判：写就是写
    assert analyzer.analyze("s = apis.spotify\ns.login()\n").category == "declared_write"
    # 文档里没有 ⇒ 认不出，保持 unknown（fail closed）
    assert analyzer.analyze("s = apis.spotify\ns.play()\n").category == "unknown"


def test_api_named_like_a_builtin_method_is_classified_by_the_docs() -> None:
    """world 里真有一个叫 `extend` 的写 API 时，`x.extend()` 就该判成**写**。

    旧实现只能判 unknown（名单撞了就一票否决），新实现查文档 ⇒ 直接判对。"""

    class _CollidingWorld:
        task = type(
            "_Task",
            (),
            {"api_docs": {"spotify": {"extend": {"method": "POST"}}}},
        )()

    analyzer = AppWorldCodeEffectAnalyzer()
    analyzer.bind_world(_CollidingWorld())

    assert analyzer.analyze("s = apis.spotify\ns.extend([])\n").category == "declared_write"


def test_real_api_calls_still_resolve() -> None:
    """放宽局部调用不能把真 API 也放过去 —— 写仍是写、读仍是读。"""

    analyzer = _analyzer()

    assert analyzer.analyze("apis.spotify.show_song()").category == "read_only"
    assert analyzer.analyze("apis.spotify.login()").category == "declared_write"
    assert analyzer.analyze("apis.spotify.unknown_api_xyz()").category == "unknown"


# ==================== 泛化：读写判定靠数据，不靠名单 ====================
# 判读写的**唯一**数据来源是 AppWorld 自带的 `api_docs`（每个 task 有限、可枚举）：
#
#     写 ⟺ 调用了 API，且该 API 的 method 非只读
#     不是 API 调用的 ⇒ 本地 ⇒ 只读（分析器自己的模型：只有 apis/requester 能触达世界）
#
# 唯一要单独拦的是**模块**（`os.system` 不是 API，但能碰文件系统）。
# 于是「unknown」的范围从「所有没列进白名单的写法」收缩到「我们认不出接收者是谁」。


def test_api_alias_resolves_to_write() -> None:
    """`s = apis.spotify` 之后 `s.login()` 就是写 —— 别名必须能解析。

    这是旧实现最主要的漏洞：别名调用一律 unknown ⇒ 整批回退。而它在 AppWorld
    生成的代码里很常见（先取一次对象，再连着调几个 API）。
    """

    assert _analyzer().analyze("s = apis.spotify\ns.login()\n").category == "declared_write"


def test_api_alias_resolves_to_read() -> None:
    assert _analyzer().analyze("s = apis.spotify\ns.show_song()\n").category == "read_only"


def test_api_root_alias_resolves() -> None:
    """别名可以指向 `apis` 本身（`a = apis`），后面再逐层点下去。"""

    assert _analyzer().analyze("a = apis\na.spotify.login()\n").category == "declared_write"


def test_api_alias_to_an_unknown_api_stays_unknown() -> None:
    """别名解析成功但与文档对不上 ⇒ 仍然 unknown（fail closed 不破）。"""

    assert _analyzer().analyze("s = apis.spotify\ns.no_such_api()\n").category == "unknown"


def test_data_returned_by_an_api_is_not_an_alias() -> None:
    """`songs = apis.spotify.show_song()` 拿到的是**数据**，不是 API 对象。

    这条是把别名判定做对的关键：RHS 里出现 `apis` 不足以判定别名 ——
    调用**结果**是数据。所以必须是「RHS 是 apis 的属性链且**不含 Call**」才算别名。
    """

    assert _analyzer().analyze("songs = apis.spotify.show_song()\nsongs.extend([])\n").category == "read_only"


def test_getattr_smuggled_api_object_is_conservatively_tainted() -> None:
    """`getattr(apis, "spotify")` 绕过了属性链 —— 保守起见仍按别名处理。"""

    doc = _analyzer().analyze('s = getattr(apis, "spotify")\ns.login()\n')
    assert doc.category != "read_only", "getattr 取出的对象可能是 API，不能当本地变量"


@pytest.mark.parametrize(
    "code",
    [
        "import datetime\ndatetime.now()",
        "import math\nmath.sqrt(x)",
        "import random\nrandom.choice(items)",
        "import json\njson.dumps(x)",
        "from math import sqrt\nsqrt(x)",
        "import itertools\nitertools.chain(a, b)",
    ],
)
def test_pure_module_calls_are_read_only(code: str) -> None:
    """纯标准库模块的调用 —— 判只读。这是 `datetime.now` 那类拒绝的根治。"""

    assert _analyzer().analyze(code).category == "read_only"


@pytest.mark.parametrize(
    "code",
    [
        'import os\nos.system("rm -rf /")',
        "import pathlib\npathlib.Path('f').write_text('x')",
        "import shutil\nshutil.rmtree('d')",
        "import subprocess\nsubprocess.run(['ls'])",
    ],
)
def test_io_capable_modules_stay_unknown(code: str) -> None:
    """能碰文件系统 / 起进程的模块 —— **保持 unknown**。

    `os.system` 不是 API 调用，但它显然不是「本地操作」。这条是把
    「不是 API 就只读」这条规则收窄到安全范围的那道闸。
    """

    assert _analyzer().analyze(code).category == "unknown"


def test_impure_modules_are_excluded_from_the_pure_set() -> None:
    """纯模块集合必须**派生自 AppWorld 自己的可导入表**，再减掉能碰 I/O 的。

    AppWorld 的 `ALLOWED_MODULE_NAMES` 说的是「能 import」，不是「无副作用」——
    里面就有 `os` / `pathlib` / `io`。所以派生之后必须显式减掉。
    """

    assert {"os", "pathlib", "io", "subprocess", "shutil"} & set(module._PURE_MODULES) == set()
    assert {"math", "datetime", "random", "json", "re"} <= set(module._PURE_MODULES)


def test_pure_module_set_tracks_appworlds_own_list() -> None:
    """纯模块集合不能凭空扩大 —— 每个成员都必须在 AppWorld 的可导入表里。

    这条防的是「为了让某个任务跑通，往纯模块表里加一个它需要的名字」。
    """

    import importlib.util

    spec = importlib.util.find_spec("appworld.common.safety_guard")
    if spec is None:  # pragma: no cover - AppWorld 不在时跳过
        pytest.skip("appworld 不可导入")

    from appworld.common.safety_guard import ALLOWED_MODULE_NAMES

    extra = set(module._PURE_MODULES) - set(ALLOWED_MODULE_NAMES)
    assert not extra, f"这些名字不在 AppWorld 的可导入表里：{sorted(extra)}"


def test_never_assigned_receiver_stays_unknown() -> None:
    """接收者**从没在本 batch 里被赋值** ⇒ 不知道它是什么 ⇒ 保持 unknown。

    这是一条安全闸，不是保守洁癖：`def helper(api): api.login()` 里的 `api`
    完全可能就是 API 对象。放宽它等于允许代码把 API 藏在函数参数后面。
    """

    assert _analyzer().analyze("def helper(api):\n    api.login()\n").category == "unknown"


def test_api_returning_call_result_is_local_data() -> None:
    """接收者是**调用结果** ⇒ 数据 ⇒ 本地（AppWorld 的 API 不返回 API 对象）。"""

    code = (
        "def get_songs():\n"
        "    return apis.spotify.show_song()\n"
        "get_songs().append(x)\n"
    )
    assert _analyzer().analyze(code).category == "read_only"


def test_locally_defined_names_are_local() -> None:
    """`def` / `for` / `with as` 引入的名字也算本地绑定。

    少认一种，那个名字就会被当成「不知道是什么」⇒ unknown ⇒ 整批回退 ——
    和漏一个内建函数是同一种坏法。
    """

    # `def` 出来的函数，按名调用
    assert _analyzer().analyze("def helper():\n    return 1\nhelper()").category == "read_only"
    # `for` 的目标变量
    assert _analyzer().analyze(
        "for item in apis.spotify.show_song():\n    item.extend(x)"
    ).category == "read_only"
    # 函数体里的调用**逐条判定**，不会被 `def` 一起放行
    assert _analyzer().analyze(
        "def helper():\n    return fetch_from_elsewhere()\nhelper()"
    ).category == "unknown"


# ==================== unknown ⇒ 按写处理（建 checkpoint），不是按拒处理 ====================
# 判不出来 ≠ 做不了：AppWorld 的 checkpoint 是 `world.save_state()` —— **整个世界状态**,
# 所以任何触达世界的改动都能被回滚。于是「未知」的正确处置是
# 「建快照 → 执行 → 出事就回滚」，而不是「不执行」。
#
# 结果是判据变干净了：
#
#     只读   ⇐ scope=READ_ONLY          （唯一不建 checkpoint 的一侧）
#     写     ⇐ scope=REVERSIBLE_WRITE   （declared_write / opaque_exec / **unknown**）


@pytest.mark.parametrize(
    "code",
    [
        "os.system('x')",                       # 不纯模块
        "mystery(x)",                           # 从没绑定过的名字
        "s = apis.spotify\ns.no_such_api()",     # 文档里查不到
        "THIS IS NOT PYTHON",                   # 解析不了
    ],
)
def test_unknown_actions_declare_a_write_scope(code: str) -> None:
    """四种 unknown 都必须声明成写 —— category 保持 unknown（审计），scope 变成写。

    只改 scope 不改 category：`category="unknown"` 是**事实**（我们确实不知道
    这是什么），`scope=REVERSIBLE_WRITE` 是**处置**（当作可回滚的写来做）。
    把事实改掉，M4 / 审计 / 误判统计就都看不见这一类了。
    """

    verdict = _analyzer().analyze(code)

    assert verdict.category == "unknown", f"{code} → {verdict.category}"
    assert verdict.scope == EffectScope.REVERSIBLE_WRITE, f"{code} → {verdict.scope}"


def test_only_explicit_reads_skip_the_checkpoint_scope() -> None:
    """反过来钉一遍：**只有明确的只读**是 READ_ONLY，别的都不是。"""

    analyzer = _analyzer()

    assert analyzer.analyze("apis.spotify.show_song()").scope == EffectScope.READ_ONLY
    assert analyzer.analyze("songs = []\nsongs.extend([])").scope == EffectScope.READ_ONLY
    for code in ("apis.spotify.login()", "os.system('x')", "mystery(x)"):
        assert analyzer.analyze(code).scope != EffectScope.READ_ONLY, code
