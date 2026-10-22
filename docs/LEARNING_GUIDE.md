# 读代码导读（TraceOffload）

这份文档不讲功能，讲**这个项目是怎么写出来的**。

功能说明在 [README](../README.md) / [ARCHITECTURE.md](ARCHITECTURE.md) 里已经
有一份了。这里回答的是另一个问题：**读这份代码时，每一处该盯住什么设计、
哪些地方是踩过坑之后改出来的、为什么写成这样而不是另一种写法。**

每处都标了 `file:line`，可以直接点进去看。

---

## 0. 这份代码的"性格"

先说清楚一件事，否则读起来会觉得代码啰嗦。

这个项目的每一个模块，注释里几乎都在回答同一个问题：
**「这里为什么不能写成更简单的样子？」** 而不是「这里在做什么」。

举三个例子：

- `contracts/routing.py:221` 的 `_validate_route_shape` —— 不是"检查参数"，
  是"让错误配置在构造期炸掉，而不是跑到一半才发现守卫没被调用"。
- `m2/predictor.py:81` 的 `_to_float32` —— 不是"转 float32"，
  是"让『在线向量与离线逐位相同』这句话能成立"。
- `mechanism_counters.py:39` 的 `ABSOLUTE_KEYS` —— 不是"排除几个字段"，
  是"避免报告里出现一列恒为 0 的假数据"。

**读法调整**：看到大段注释不要跳过，也不要当成噪音。
它们记录的是**已经付过学费的决策**，是这个项目最有价值的部分。

---

## 1. 骨架：一次调用发生了什么

整件事围绕一个概念：**span**。Agent（ReAct 循环，不改）每调一次大模型，
就是一个 span。TraceOffload 决定**这个 span 交给大模型还是小模型**。

一次 span 的生命周期（`coordinator.py:628` `orchestrate_episode`）：

```
Agent 调 model.generate
   │
   ├─ M1 建模    → RuntimeProfile（画像 + 116 维向量）
   ├─ M2 路由    → RoutePlan（目标模型 + 执行模式 + 守卫方案）
   ├─ M3 执行    → 小模型出候选 → pre-check 守卫 → 事务化 → post-check
   ├─ Agent 拿响应去调工具 → ToolProxy 真执行 → commit / rollback
   └─ M4 记账    → SpanRecord 落盘；有执行失败就写进记忆库
```

**两条接缝**是整个设计的立足点：Agent 本体一行不改，只包住
`model.generate` 和 `env.execute` 两个口子。

---

## 2. 契约层：把不可能犯的错变成构造期错误

**先读 `contracts/`，再读任何实现。** 模块之间不互相 import，
全靠这里的 Pydantic 模型通信。

按顺序读：`base.py`（枚举与公共约定）→ **`routing.py`（最重要）** →
`execution.py` → `records.py`。

### 2.1 整个系统的安全骨架：`_validate_route_shape`

`contracts/routing.py:221-250`。用 `model_validator` 在**构造期**强制：

```python
if target.kind == LARGE_MODEL and execution_mode != DIRECT:      # 大模型只能直达
if target.kind == LARGE_MODEL and fallback_plan is not None:     # 大模型不许有 fallback
if execution_mode == GUARDED_SLM and target.kind != SMALL_MODEL: # guarded 只能配小模型
if execution_mode == DIRECT and verifier_plan.steps:            # 直达不许带守卫
if execution_mode == GUARDED_SLM:
    if not verifier_plan.steps:                                  # 必须有守卫
    if not any(step.stage == PRE_COMMIT for step in ...):        # 必须有 pre-commit
    if len(post_steps) != 1:                                     # 恰好一个 post-check
```

**为什么值得学**：这几行不只是"防御性编程"，它们编码了一个**安全论证**：

> 小模型的输出只有在「先过守卫」+「事后可回滚」两个条件都成立时才被采用。

`DIRECT` 不许带守卫 ⇒ 直达路径上没有"守卫"这个概念，
也就不会出现"以为有守卫其实没有"的情况。

> ⚠️ 这段约束对应一个真实 bug：早期小模型目标写的是 `DIRECT`，
> 于是**守卫一次都没被调用过**。见 `m2/routers.py:16`
> （原文：「这是本系统最早的一个 bug」）和 `routers.py:281-282`。

### 2.2 `VerifierPlan` 的两个细节

`contracts/routing.py:99-111`：

```python
if self.steps and self.empty_plan_reason is not None:
    raise ValueError("empty_plan_reason must be null for a non-empty verifier plan")
if not self.steps and self.empty_plan_reason is None:
    raise ValueError("an empty verifier plan requires empty_plan_reason")
```

**空计划必须说明原因**。因为 DIRECT 路径用的是"显式空计划"，
如果不要求写原因，"空计划"和"忘了填"就分不出来了。
这和 `routers.py:71-77` `_empty_verifier()` 里写死
`empty_plan_reason="large_model_direct_no_guard"` 是配套的。

还有一个细节（`routing.py:71-79`）：`threshold` 只在
`score_at_least` 条件下允许存在，其它条件带 threshold 直接抛。

---

## 3. M2：决策的形状

`m2/routers.py`（446 行）。四种模式 + 一个装饰器。

### 3.1 `_ClassifierRouter`：三个设计决定

`routers.py:276-369`。核心只有一句 `use_small = bool(decision.offload)`，
但三处设计值得抄：

**① 拿不到向量 → 直接走大模型**（`routers.py:311-324`），
reason_code = `l2_add4_vector_unavailable`。注释：

> 不退回旧 artifact —— 那是另一个模型，p 的分布和 τ 的语义都不同，
> 中途换模型等于实验换了变量。

这是**实验方法论**级别的约束，不是工程细节。降级要降到一个**语义等价**的地方，
不能降到一个"看起来也能跑"的地方。

**② 小模型目标一律 `GUARDED_SLM`**（`routers.py:347-361`），
并带 `fallback_plan` 指向大模型。

**③ 两个 p 和两个 τ 都要记**（`routers.py:336-346`）。
`classifier_p` / `classifier_threshold` 记**经济那一段**，
`classifier_p_gate` / `classifier_tau_gate` 记**门控那一段**。
原因见 `contracts/routing.py:172-180`：

> 只看一个 p 分不出「全没卸载」是门控挡的还是经济挡的 ——
> **这两种情况的调参方向相反**。

**这是可观测性设计的范例**：一个布尔决策背后有两个独立的门，
就必须留两个独立的分数，否则事后无法归因。

### 3.2 `_ShadowRouter`：装饰器的正确用法

`routers.py:372-444`。它包住**任何** router：内层照常算、照常记录，
但整个 plan 被强制换成大模型 DIRECT。

**盯住它替换了哪四个字段**（`routers.py:429-434`）：

```python
"target": self._forced_target(),
"execution_mode": ExecutionMode.DIRECT,
"verifier_plan": _empty_verifier(),
"fallback_plan": None,
```

为什么换四个而不是只换 `target`？注释解释了：只换 target 的话
`execution_mode` 还是 `GUARDED_SLM`，这次调用会被拖进守卫路径 ——
而「小模型也走大模型那条路」正是这组实验要保证的东西。

**更盯住它没替换什么**：`decision_meta` 里预测器那几栏
**一个都不许动**。注释：

> 那是「M2 真的算了」的唯一证据；覆盖掉就等于对照组退化成
> 「没装预测器」，说服力全没。

以及：它把**原本的** `reason_code` 记进 `shadow_would_route`（`routers.py:440`）。
这是整个 shadow 臂存在的理由 —— 没有这个字段，「路由器本来会卸多少」
就只能离线估，而不是这条真实轨迹上的数。

> **为什么做成装饰器而不是预测器的一个参数**（`routers.py:375-379`）：
> 「不生效」是对**路由输出**的操作，不是对预测器的操作。
> 放在这一层，换预测器、换预测器的数量、换 mode，这个类**一行都不用改**。

### 3.3 `_RandomRouter` 的可复现性

`routers.py:216-273`。用 `random.Random(seed)`，seed 来自配置
`runner.random_route.seed`。注释：

> so a run is reproducible while the sequence of targets does not depend on
> the task, the span content, or any feedback.

**注意它记了 `self._draws`**（:245, 262）—— 随机对照组必须能事后复盘
每一次掷硬币的结果，否则"随机臂"和"不可解释的臂"没法区分。

---

## 4. add4 预测器：把「会静默失准」的地方全部堵上

`m2/predictor.py`（376 行）。纯 Python 前向，不依赖 sklearn。

### 4.1 116 维怎么来的

文件头注释（`predictor.py:1-17`）就是公式：

```
e_v2   (1024)  ← M1 用 SpanEmbeddingTextBuilderV2 算的 embedding
e_act  (1024)  ← 同一时刻用 SpanActionsTextBuilder 算的「最近 10 个动作签名」

x = [ prefix_n_A, log_est_prompt_tokens ]      # 2，各自标准化
  ⊕ PCA50((e_v2  − mu2) / sd2)                 # 50
  ⊕ PCA64((e_act − mu3) / sd3)                 # 64
                                               # ⇒ 116 维

p_gate = sigmoid(gate_coef · x + gate_b)
p_net  = sigmoid(net_coef  · x + net_b)

卸载 ⟺ p_gate ≥ τ_g 且 p_net ≥ τ_t            # 与，不是或
```

实现在 `features_from`（:282-308）与 `_project`（:262-280）。

### 4.2 三处「不报错但会偏」的防御

这个文件最值得学的，是**它如何防御静默失准**：

**① `joint_normalization` 必须显式为 false**（:145-152）：

```python
joint = spec.get("joint_normalization")
if joint is not False:
    raise ValueError("add4 artifact: joint_normalization 必须显式为 false" ...)
```

注释：*「缺字段就当 false 是危险的：一份抄了 v4 格式、带 norm_mu/norm_sd 的
artifact 会被静默地**不**归一化。」*

> **模式**：布尔开关不要用"缺失即默认值"，尤其当"默认"和"另一个合法取值"
> 指向不同行为时。显式声明。

**② 精度还原**（`_to_float32`，:81-93）：

离线两段嵌入的来源精度不一样 —— `aw3_emb_v2goal.jsonl` 是 float64，
`emb_C_bare_w10.npz` 是 float32。在线返回 float64，直接喂进去就与训练时
差一个舍入。

> 差别小到不影响任何决策，但它让「在线向量与离线逐位相同」这句话不成立 ——
> 而那句话正是 `tests/components/test_l2_add4_golden_features.py` 要钉的东西。
> 所以按 artifact 声明的精度还原，而不是靠容差把问题盖过去。

**③ 输入契约**（`predict_from_span_features`，:316-323）：

> 输入必须已经是 `features_from` 的产物 —— 少一步标准化会得到一个
> 「看起来像向量」的垃圾，而且不报错。

### 4.3 「与」而不是「或」—— 数据结构层面的防护

`decide_from_span_features`（:334-354）：

```python
offload=bool(p_gate >= gate_tau and p_net >= net_tau)
```

为什么这么设计（`predictor.py:29-31`）：

> 对外给的是 `Add4Decision`（含两个 p、两个 τ），而不是一个标量 ——
> 让调用方去重新实现那个与，迟早会出现「只比了一个」的版本。

> **模式**：把"多条件的合取"封装进返回值，而不是把条件散给调用方。
> 调用方只读 `offload` 一个 bool，想用错都用错不了。

---

## 5. M1：特征从哪来，以及「同一语义一份实现」

| 文件 | 行数 | 关键点 |
|---|---|---|
| `m1/appworld_features.py` | 229 | **在线**特征构建器 |
| `m1/span_embedding.py` | 913 | 两段带嵌入的文本配方 |
| `m1/signature.py` | 313 | 动作签名规范化（离线/在线**同一份实现**） |
| `m1/profiler.py` | 264 | RuntimeProfiler + ClassifierFeatureProfiler |

### 5.1 `_message_text`：一个曾经存在的 bug

`appworld_features.py:72-85`。注释记录得很完整：

> 这里曾经多一条 `return str(content)`，对 list/dict 会记下 `repr` 的长度
> （`str(["blob"])` = 8 字符），使 `total_chars` **系统性高估**。
> 当前真实数据里 `content` 只出现 `str`(340103) 与 `None`(207) 两种，
> 所以当时不触发 —— 但只要 AppWorld 改成结构化 content，p 就会静默偏掉。

**这就是"逐位复现"要求的代价**：为了和离线口径一模一样，
在线代码里连"`content` 不是 str 就记 0 字符"这种反直觉的规则都得照抄，
还得写注释解释为什么这么反直觉。

### 5.2 构造期拒绝算不出来的列

`AppWorldOnlineFeatureBuilder.__init__`（:184-200）：

```python
unsupported = [name for name in self.feature_names
               if name not in APPWORLD_SUPPORTED_FEATURES]
if unsupported:
    raise ValueError(f"artifact 要求了在线无法计算的列 {unsupported}；...拒绝静默补 0 ...")
```

注释：

> 补 0 得到的 p 不只是噪声，而是**自信地错**
> （实测：缺 `log_est_prompt_tokens` 时 p 从 0.32 跳到 0.9995）。

**"自信地错"**（confidently wrong）这个词值得记住 —— 它比崩溃危险得多，
因为下游所有断言都会通过。

### 5.3 `prefix_n_A` 的两个条件

`count_prefix_write_ops`（:127-150）：

```python
if getattr(outcome, "success", False)
and getattr(outcome, "effect_category", None) in _WRITE_CATEGORIES
```

**两个条件都要满足，缺一就与离线不一致**。注释还说了第三个决定：

> `effect_category` 为 None（旧事件 / 未分类）时**不计入**：宁可少算，不猜。

### 5.4 `signature.py`：AST 优先，正则兜底

`m1/signature.py:253-265` `api_calls_of_code` 同时用 AST 和正则两条路 ——
AST 优先，正则兜底（代码不是合法 Python 时）。

因为这份文件**离线链路和在线运行时共用**，一旦分两处写，
两边的"动作签名"就会漂移，而漂移的坏法是不报错。

---

## 6. M3：受控执行 —— 安全是怎么实现的

| 文件 | 行数 | 作用 |
|---|---|---|
| `m3/executor.py` | 2245 | 两种执行模式；准入、守卫、超时、fallback |
| `m3/tool_proxy.py` | 491 | 事务边界（execute / begin / commit / rollback） |
| `m3/guards.py` | 531 | 形状守卫 + M4 失败记忆守卫 |
| `m3/checkpoint.py` | 459 | span 级 checkpoint 会话 |
| `m3/integration.py` | 1002 | AdapterSession：把工具历史投影成 prior facts |

### 6.1 M3 不执行动作 —— 只建立边界

`executor.py:274-287` 模式分派；`_execute_guarded_slm` 的 docstring
（:356-363）说明了分工：

> There is deliberately no candidate execution here. A mutating action is
> admitted only after its rollback capability has been established;
> the adapter executes the accepted response in the primary environment
> and reports the result through `execute_tool_phase`.

**M3 只做三件事**：准入（能不能做）、建事务边界（做了怎么撤）、守卫（对不对）。
真正的执行在 adapter 层，结果通过 `execute_tool_phase`（:784）回报。

这是"Agent 本体不改"的第二处体现。

### 6.2 checkpoint 的状态机 —— 11 个状态，转移表是白名单

`m3/checkpoint.py:18-93`。`CheckpointState` 有 11 个状态：

```
NEW → PREPARING → PREPARED → EXECUTING → VERIFYING → COMMITTING
  → COMMITTED_RELEASE_PENDING → CLOSED
```

中间任何一步都可以转到 `ROLLING_BACK` → `ROLLED_BACK_RELEASE_PENDING`，
也可以直接转 `RECOVERY_REQUIRED`。

**转移是白名单**（`_ALLOWED_TRANSITIONS`, :39-93）：
`CheckpointState.CLOSED: frozenset()` —— 终态没有任何出边。

**盯住那两个"半吊子"状态**：

```python
COMMITTED_RELEASE_PENDING = "committed_release_pending"
ROLLED_BACK_RELEASE_PENDING = "rolled_back_release_pending"
```

它们表示"逻辑上已经 commit/rollback，但资源还没释放"。这两个状态被
`_RECOVERY` 集合（:33-37）标记为**需要恢复处理**，和 `RECOVERY_REQUIRED`
放在一起 —— 因为"已提交但没释放"同样意味着有悬挂的句柄。

> **模式**：状态机的合法转移用显式的表表达，而不是散在各个 `if` 里。
> 这样"能不能从 A 到 B"是一个可以一眼看完、可以打印、可以断言的东西。

`checkpoint.py` 的模块 docstring 还说明了**单一所有权**：

> The executor remains responsible for orchestration, but this object owns the
> span's transaction set and its monotonic lifecycle. Backend calls are still
> made by ToolProxy; **they cannot clear this state directly.**

即：ToolProxy 是唯一能把状态清掉的对象，其它人不能。

### 6.3 工具策略是纯数据

`guards.py:46-64` 的 `ToolRule`：

```python
@dataclass(frozen=True)
class ToolRule:
    tool_name: str
    required_argument_paths: tuple[str, ...] = ()
    non_empty_string_paths: tuple[str, ...] = ()
    allowed_argument_keys: tuple[str, ...] | None = None
    max_argument_bytes: int | None = None
    reject_nul: bool = True
    accepted_statuses: tuple[ToolExecutionStatus, ...] = (ToolExecutionStatus.SUCCESS,)
    accepted_returncodes: tuple[int | None, ...] = (None, 0)
```

注释：

> The rule is deliberately data-only. Complex shell or domain validation is
> selected by `verifier_ref` and implemented by a registered verifier, **not by
> evaluating arbitrary configuration code**.

`ToolPolicy.from_mapping`（:94-...）里还有一句：

> Invalid values fail closed with `ValueError` rather than silently widening
> the whitelist.

> **注意方向**：这里 fail-**closed**（拒绝），而 M4 记忆守卫 fail-**open**（放行）。
> 这不是矛盾 —— 见 7.1。

### 6.4 副作用分级：判不出来就往危险一侧倒

副作用分级由接入层注入。AppWorld 用
`integrations/appworld.py:568` 的 `AppWorldCodeEffectAnalyzer`，
它读 OpenAPI 的 HTTP method 来判断读写。

`traceoffload/effects.py:1-14` 的模块 docstring 说清了为什么
`effects.py` 里**只有数据类型、没有分析逻辑**：

> 这个模块只放**共享的数据类型**，不放任何分析逻辑 —— 这样 executor 和
> AppWorld 适配器都能 import 它，而不用把一个 1300 行的 bash 语法分析器
> 带进依赖。

**依赖方向的自觉**：核心模块不该被接入层的重资产拖累。

`effects.py:20-27` 的 `EFFECT_CATEGORIES` 六类里，
`external` / `irreversible` / `unknown` 是**执行前 fail-closed** 的。

---

## 7. M4：跨 span 的记忆

| 文件 | 行数 | 作用 |
|---|---|---|
| `m4/memory.py` | 1618 | 建库、检索、记账 |
| `m4/store.py` | 446 | SQLite 持久化（**维度闸**） |
| `m4/span_features.py` | 214 | 52 维记忆向量 + 动作签名 |
| `m4/action_guard.py` | 80 | 守卫查询入口 |

### 7.1 写库判据 —— 全仓最精确的一个判据

`m4/memory.py:1229-1246` `_span_had_an_executed_failure`：

> 本 Span 里**有没有一个真的执行过、并且失败了的工具调用**。
> 这是 M4 收录的唯一判据。
>
> 为什么不用 `not compute_local_success(...)`：那个是「本地这一段有没有出错」的
> 口径，把**没执行**（准入被拒）和**执行成功了但产物被拒**（post-check REJECT）
> 也算成 False。对 M4 来说那两种都不构成「这个动作会失败」的证据。

三种情况必须分清：

| 情况 | 工具跑了吗 | 进 M4 库吗 | 为什么 |
|---|---|---|---|
| 准入被拒（pre-check 拦截） | 没跑 | **不进** | 没执行就没有"这个动作会失败"的证据 |
| post-check 拒绝，但工具成功 | 跑了，成功 | **不进** | 动作本身没失败，是产物不对 |
| 工具真的执行失败 | 跑了，失败 | **进** | 这才是证据 |

### 7.2 `action_signature`：为什么不能直接用参数字典当钥匙

`m4/span_features.py:93-110`：

```python
def action_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """``(工具名, 参数)`` 的**规范化**签名 —— 键序无关、可哈希、可跨进程比。"""
    return json.dumps([str(tool_name), arguments], ...)
```

注释给了两个理由：

> * 字典不可哈希，装不进 ``frozenset``；
> * **键序不同的两个等值字典必须签出同一个名字** —— 否则同一个动作会因为
>   ``json.dumps`` 的键序差异被当成两个，命中率凭空掉一半。

以及形状的选择：

> 形状取 ``[工具名, 参数]`` 这个二元列表而不是 ``{"tool":…, "args":…}`` ——
> 后者的键会和用户参数里的同名键撞车。

> **模式**：任何要当"键"用的东西，都要先问"它在不同进程/不同字面量下
> 会不会变"。`json.dumps` 的 `sort_keys` 与否，就是"命中率凭空掉一半"
> 和"稳定命中"的分界线。

### 7.3 `failed_action_signatures`：两张表怎么对齐

`span_features.py:40-79`。有个很实在的工程问题：

> 参数在 ``response.tool_calls`` 上，成败在 ``tool_results`` 上 ——
> **两张表都没有对方的字段**（``ToolExecutionResult`` 里根本没有 ``arguments``）。
> 唯一能对齐两者的是 ``tool_call_id``。

三条语义各有理由：

> * **响应里有、结果里没有 ⇒ 不算失败。** 守卫拦下的动作正是这种：它没跑，
>   所以没有成败。当成失败会让「被拦下的」回写进库，下次拦得更狠 —— **自激**。
> * **结果里有、响应里没有 ⇒ 丢弃。** 对不上号，宁可漏记不可记错。
> * **``status`` 成功但 ``returncode`` 非零 ⇒ 仍算失败** —— 与
>   ``m4_memory.compute_local_success`` 同口径，两处判定必须一致。

> **注意「自激」**这个失败模式：守卫拦下的动作被当成"失败"记进库，
> 下次同一个动作会被拦得更狠。反馈环会自己收紧。
> 任何"从失败中学习"的机制都要先问：这个反馈环会不会自激？

### 7.4 52 维 vs 116 维 —— 两个不同的向量空间

`m4/span_features.py:1-16` 的模块 docstring：

> 这个变换有两个消费者，而且它们**必须看到同一个向量**：
> * ``L2V4Predictor`` —— 决定这一步卸不卸（M2 的门1）
> * M4 动作守卫 —— 决定这一步拦不拦（M3 的 verifier）
>
> 两边一旦漂移，M4 嘴里的「相似场景」和预测器眼里的「相似场景」就不是一回事，
> **而且不会报错** —— 这一类错误只能靠钉住同一个 artifact 来防。

它还标出了一个容易误解的地方：

> ⚠️ 52 维里混了 2 个**非嵌入的标量**（``prefix_n_A`` / ``log_est_prompt_tokens``），
> 而且整体过了 ``(x − norm_mu) / norm_sd`` 标准化。所以它**不是「PCA 后的嵌入」**，
> 是「L2_v4 的输入表示」。当嵌入用会不对。

而 `m2/predictor.py:32-36` 说的是另一半：

> **不重算 embedding。** 两段都在 M1 的 ``profile_observer`` 里算过一次并缓存。
> 但**缓存是分开的**：M4 守卫读的是 52 维那份（``guard_vector`` 的产物），
> 本预测器读 116 维那份 —— 两者的向量空间不同，共用会静默失准。

> **模式**："不重算"是好意，但"缓存键一样"会诱导你复用。
> 两个消费者读同一个缓存的**不同视图**时，要用不同的键，或者显式分开存。

### 7.5 `action_guard.py`：三段分开的可测性

`m4/action_guard.py:1-18` 的模块 docstring 画了链路：

```
M3 出候选响应
  → `M4FailureMemoryVerifier.check`      负责**顺序**（形状 → 记忆 → fail-open）
  → **action_guard.py**                   负责**取值**（身份→52 维→逐个 tool call）
  → `M4Memory.query_failed_action`        负责**判定**（相似度 AND 签名）
```

> 三段分开是为了各自可测：顺序那段不碰记忆库、判定那段不碰响应对象、
> 取值这段两边都要碰但只有十几行。

**两条 fail-open**（:13-17）：

> **拿不到向量就放行，不抛异常。** 身份是 None（非 guarded 路径）、或它不在
> 向量缓存里（M1 还没算完、换了进程），都算「没有证据」而不是「有罪」。
> 守卫是增强，不是依赖 —— 它坏了不能把路由卡住。

**这里 fail-open 是对的**（对比 6.3 的工具策略 fail-closed）：
守卫是**增强**（少一个优化），策略是**约束**（少一层保护）。
判据是"这个东西坏了，系统的结果是更安全还是更危险"。

还有一个细节（:68-71）：

```python
# 一次 tool call = 一次查询。**只数真正查了的** ——
# 身份缺失/向量缺失在上面就 return 了，那些不该记成 lookup，
# 否则「守卫没接线」和「守卫查了但没命中」看起来一样。
counters.bump("m4_guard_lookups")
```

> **模式**：计数要区分"没做"和"做了但没结果"。否则无法区分
> "功能没接线" 和 "功能接了但没命中" —— 这两种 bug 的修法完全不同。

### 7.6 维度闸 —— 防止换向量空间后读到老库

`m4/store.py` 有"维度闸"（见 [ARCHITECTURE.md](ARCHITECTURE.md) M4 一节）：
换了向量维度后，旧库里的记录不能被检索到。

这和 7.4 的问题是同一个：**向量空间的版本必须被持久化并校验**，
否则换模型/换维度之后，系统会拿新查询去匹配旧向量，
而且结果看起来是"正常的"（只是全都匹配不上，或者更糟 —— 匹配上了错的）。

---

## 8. Coordinator：串起来 + 三种"强制大模型"

`coordinator.py`（1797 行）。**现在再读它**，因为你已经知道 M1–M4 各自干什么了。

### 8.1 三种"强制大模型"的优先级

`coordinator.py:731-760` 的分支顺序：

1. `_forced_target is not None` —— 调用方显式指定
2. `recovery_directive is not None` —— 回滚失败的恢复锁
3. `loop_directive is not None` —— 重复循环的健康锁

**每一次都绕过 M2**，直接 `_direct_route`（:1485）。注释：

> Do not ask M2 to rediscover a health decision that Coordinator already
> proved from completed tool facts.

> **模式**：M2 是"根据预测做选择"，但有些事**不是预测出来的**，
> 是 Coordinator 从**已经发生的事实**里证明出来的。
> 这类事实不该再过一遍预测器 —— 它可能给出相反的答案。

### 8.2 `RecoveryController`：只有回滚失败才建锁

`coordinator.py:176-322`。docstring：

> A lock is created only after an SLM post-check failure whose rollback failed.
> While it exists, Coordinator bypasses normal routing and forces a large-model
> DIRECT recovery span. ``resume_check`` is intentionally a separate operation
> from the empty verifier plan of that direct route.

**注意 `resume_check` 是独立操作**（:181-183）—— 恢复要检查状态能不能续上，
和"这次调用要不要守卫"是两件事。混在一起会导致"恢复路径上没有守卫"
被误解成 bug。

### 8.3 循环健康：两个不同的循环

`coordinator.py:137-159` `_WorkflowLoopState`。这是**从实测数据里长出来的**机制。

两个循环（`coordinator.py:141-148`）：

- **命令循环**：两次相同的只读动作批次 → **一次性**强制大模型
- **推理循环**：跨 episode 重复同一段推理内容 → **持久**强制大模型，
  直到有一个改变了状态的 episode 来解除

理由写在 `coordinator.py:146-148`：

> The same reasoning content repeated across episodes (while commands keep
> shifting inside the same read-only region) arms a persistent direct-large
> lock that is only released by a state-changing episode, **because one
> large-model turn may not itself perform the missing write**.

> **为什么状态是"基于已完成的事实"而不是"基于命令名或解析器启发式"**
> （`coordinator.py:139-140`）：
> The state is deliberately based on completed tool facts and accepted model
> outputs, rather than command names or parser-specific heuristics.

### 8.4 `_WorkflowLoopState` 的字段设计

```python
last_action_fingerprint: str | None = None
repeat_count: int = 0
action_summary: str | None = None
last_reasoning_fingerprint: str | None = None
reasoning_repeat_count: int = 0
reasoning_summary: str | None = None
loop_kind: Literal["command", "reasoning"] = "command"
force_direct: bool = False
```

**用 fingerprint 而不是存原始内容**：只存"指纹"（比较用），
不存可能很大的原文。`action_summary` / `reasoning_summary` 是给人看的
（写进 reason_code / 日志）。

### 8.5 fallback 的防递归设计

`_start_fallback_episode`（`coordinator.py:1195-1250`）的 docstring：

> IMPORTANT: Fallback episodes are ALWAYS forced to use:
> - Large model (LLM)
> - DIRECT execution mode (no guards, no further fallback)
> - AttemptRole.LARGE_MODEL_FALLBACK
>
> This prevents infinite fallback recursion: a fallback episode cannot itself
> trigger another fallback, as DIRECT mode is not subject to verification failure.

并且有断言兜底（:1223-1225）：

```python
assert fallback.target.kind == ModelTargetKind.LARGE_MODEL, (
    "Fallback target must be LARGE_MODEL to prevent infinite recursion"
)
```

另外 `FallbackPlan` 的契约层也拦了一次（`contracts/routing.py:127-131`）：
`fallback_must_use_large_model`。

> **三道防线**：契约（构造期）→ 断言（运行时）→ 显式强制（`_forced_target`）。
> 同一个不变量钉三层，是"这个不变量很重要"的明确信号。

还有一处细节（:1226-1228）：

> The fallback receives the unchanged Agent history. Route and execution facts
> remain in the trace instead of becoming a synthetic model-visible handoff message.

**fallback 时不改 Agent 的历史**，路由事实留在 trace 里，不变成一条
"模型能看见的交接消息"。这避免了污染上下文。

---

## 9. `mechanism_counters.py`：可观测性的设计

这个模块（231 行）的文件头注释记录了**为什么需要它**：

> 不加这个，**摘掉守卫的臂和满血臂在报告里长得一模一样** —— 同样的成功率、
> 同样的 token。分不清「没效果」和「没触发」。

实测（0925，A5-full 60 次跑）三条观测路径全断：

> * ``summary.json`` 里没有 guard/memory 字段
> * AppWorld 的 ``logger.jsonl`` 是空的
> * ``M4FailureMemoryVerifier`` 拒绝时**复用内层 verifier 的 `guard_ref`**
>   ⇒ 记忆那一层在 ``verifier_refs`` 里也不可见

> **这是"为什么需要可观测性"的最好范例**：不是"有数据比没数据好"，
> 而是"**没有这一层，你连问题出在哪个环节都定位不了**"。

### 9.1 四条设计约束

**① 键必须先注册**（`mechanism_counters.py:40-72` `T2_KEYS` / `T2_MAPS`）：

> 写错键名报 ``KeyError``。静默建新键 = 报告里多一列恒为 0 的假数据，
> 而**假数据比缺数据坏 —— 缺数据会被发现，假数据不会**。

**② 未 bump 的键出现在快照里（值为 0）**：

> 缺键会让汇总脚本以为「字段没实现」而跳过断言；0 才是「实现了但没触发」。

**③ `null()` 空接收器**：

> 组件默认不接线（``counters=None``），免得每个调用点都写
> ``if counters is not None`` —— 漏一处就是 AttributeError 打断主链路。

**④ 线程安全**：

> 写入在 M4 worker 线程、读取在主线程。**丢计数会让断言假通过**。

### 9.2 `ABSOLUTE_KEYS` —— 有些字段做差是没意义的

`mechanism_counters.py:81-105`。分类：库存快照（`m4_records_at_start/_end`）、
派生值（`*_avg_ms`）、瞬时值（`queue_size`）、常量（`threshold`、`input_dim`）。

注释里记了一个特别阴的坑：

> `tau_gate` / `tau_net` 是**选出来的常量**，before/after 里一样 ⇒ 做差必然得 0。
> 这一条尤其阴：``input_dim`` 会从 52 变成 0，读起来像「维度是 0」。
>
> 实测踩过一次：summary 里 ``tau_gate: 0.0 / tau_net: 0.0``，
> 而跑的时候是 0.5381 / 0.5523 —— 读起来像「阈值是 0，所以什么都卸载」。

> **模式**：`delta_of(before, after)` 这种"前后快照做差"的模式，
> 必须配一个**显式的例外清单**。否则每个常量字段都会变成一个
> "看起来是 0 的假数据"。

### 9.3 放顶层而不是 `components/`

`mechanism_counters.py:24-30`：

> ``components/__init__`` 会 import coordinator → router → 回头 import ``guards``。
> 放在 ``components/`` 下就构成**循环导入**
> （实测 ``ImportError: cannot import name 'ToolPolicy' from partially
> initialized module``）。本模块是叶子（零依赖），放顶层最安全。

> **模式**：被低层模块（`guards.py`）依赖的东西，不能放在会被高层模块
> 导入的包里。这是一个**模块图**问题，不是命名问题。

---

## 10. 接入层：AppWorld

`integrations/appworld.py`（1807 行）。

### 10.1 一个字符串常量，两个地方必须一致

`appworld.py:89-92`：

```python
#: 这几个位置原先各自硬编码同一个字符串，**全靠手工同步**。收敛成常量之后，
#: ``tests/integrations/test_appworld_batch_tool_name.py`` 再钉住「集成侧 == 策略侧」。
BATCH_TOOL_NAME = "execute_code_batch"
```

> **模式**：同一个字符串出现在 N 个地方时，抽成常量只是第一步，
> 第二步是**让测试钉住"这些地方确实相等"**。

### 10.2 `_UNSAFE_BUILTINS` —— 为什么不能派生

`appworld.py:93-101`：

```python
#: **不能**放行的内建函数 —— 派生安全名单时从这里排除。
#: 这一半不能靠派生：它们要么能做 I/O，要么能改执行环境，要么能凭空取到
#: 全局对象。**必须显式列出**。

_UNSAFE_BUILTINS = frozenset({
    "open", "input", "breakpoint",
    ...
})
```

还有一条特别到位的说明（:98-100）：

> ``open`` 尤其重要：AppWorld 的沙箱在**运行期**把 ``open()`` 降成只读模式，
> 但分析器判的是**文本** —— **不能拿运行期的性质去放行静态分析**。

### 10.3 诚实的降级

`appworld.py:923-934`，`AppWorldToolProxy` 的两个能力声明：

```python
supports_command_observation = False
```

注释：

> executor 的 L2 越界观察会构造一个 ``bash`` action，而本 proxy 的 execute 回调
> 读的是 ``arguments["codes"]`` —— 取不到 key 就退化成空批次，
> ``batch_execute([])`` 返回空、``returncode=0``，于是被记成"观测成功、状态无变化"。
>
> 后果是 L2 **静默失效**：``effect_misclassification``（只读却写了东西）在
> AppWorld 上永远发现不了。显式声明不支持 → executor 返回
> ``_L2_NOT_AVAILABLE``（跳过检查），这才是**诚实的降级**。

以及 `commits_on_tool_failure = False`（:936-940）：

> AppWorld 一个 batch 是**一次性生成的 2~14 行代码**（实测 6620 条轨迹 /
> 97335 次调用中，**失败的写 batch 100% 多行**），batch 内靠前的写已落盘 ——
> "FAILED 也 commit" 会把这种部分提交留住。

> **模式**：当某个能力在你的环境里**没法正确实现**时，
> 显式声明"不支持"好过实现一个"看起来能跑"的版本。
> 前者会跳过检查，后者会给出错误的结论。

---

## 11. 离线链路（`router/`）

11 个文件、约 1200 行。五个 CLI 入口：

| 命令 | 作用 |
|---|---|
| `python -m router.build` | 从 `runs/` 采集 → `features.jsonl` + `texts.jsonl` |
| `python -m router.embed texts` | 算 V2_goal_recent 嵌入 |
| `python -m router.embed actions` | 算动作签名嵌入（最近 10 个） |
| `python -m router.train` | 训练 → `model.json` + `val_scores.jsonl` |
| `python -m router.select_threshold` | 挑阈值 → `tau.json` |
| `python -m router.export` | 导出 → `router/artifact/l2_add4_twostage_logit.json` |

### 11.1 特征口径与在线对齐

`router/features.py` 的模块 docstring 就把两个标量定义清楚了：

> ```
> prefix_n_A           前缀里分类为「写」的工具执行个数（AppWorld OpenAPI 的 HTTP method 判定）
> log_est_prompt_tokens = log1p(前缀消息总字符数 / 4)
> ```

并且说明了 V2 文本配方的关键设计（`features.py:11-16`）：

> ```
> [GOAL] <任务指令>
> [TOOLS] <上一步调用的 api>
> [REASONING] <思考尾部 800 字>
> [ASSISTANT] <输出尾部 1500 字>
> ...（最近 10 个 span）
> ```
> 去掉了工具输出（v1 里占 99% 的体量），这是 v2 配方的核心。

**「去掉工具输出」是一个很有价值的设计**：工具输出占了文本体积的 99%，
去掉之后，嵌入主要反映"意图"而不是"环境回显"。

### 11.2 OpenAPI 查不到就报错，不静默降级

`router/features.py:56` 定义 `MIN_API_METHODS = 470`，
`features.py:163-168` 在采集时强制：

```python
MIN_API_METHODS = 470

if len(api_methods) < min_api_methods:
    raise RuntimeError(
        f"OpenAPI 只加载到 {len(api_methods)} 个 operationId（< {min_api_methods}）："
        f"检查 --openapi-dir={openapi_dir}。缺它会让写动作**静默退化成启发式判定**。"
    )
```

> **模式**：当"降级"会改变语义（启发式 vs 权威表）时，
> 就**不要降级**，直接失败。这里 473 个 operationId 是权威的 HTTP method 表，
> 拿不到就等于把"写"判定变成了猜。

### 11.3 阈值选择的真实目标函数

`router/select_threshold.py:35-49` `sweep_a`：

```python
grid = np.quantile(p_gate, np.linspace(0, 1, 60))
for a in grid:
    m = p_gate >= a
    sub_net, sub_score = net[m], p_net[m]
    order = np.argsort(-sub_score)
    cum = np.cumsum(sub_net[order])
    i = int(np.argmax(cum))
```

目标是（文件头 :5-7）：

> ```
> U(tau_g, tau_t) = Σ_{p_gate >= tau_g 且 p_net >= tau_t} net_saving(点)
> tau* = argmax U
> ```

**不是 F1，不是准确率** —— 是"净省下来的钱"。因为这个项目要优化的是
成本，不是分类指标。

**算法上有个漂亮的优化**：外层 60 个 `tau_gate` 分位点，内层对 `p_net`
做**一次一维前缀扫描**（`np.cumsum`），所以是 60 次 O(n)，不是 3600 次 O(n)。

**稳健化是家族级 bootstrap**（:52-73），文件头解释为什么按家族而不是按点：

> 重采样家族（不是点 —— 同家族的点高度相关，按点会把区间算窄）

而且它自己承认了偏差（:18）：

> ⚠️ 挑出来的成绩是乐观的：同一批数据既选阈值又报成绩。
> **测试集实跑一次才是无偏的。**

> **模式**：用同一批数据既调参又报成绩时，**必须在文档里写明这一点**。
> 这里的诚实值得学。

---

## 12. 文档与代码不一致的地方（读代码时以代码为准）

仓库的文档写得很好，但有几处已经和代码脱节。

### 12.1 `configs/pilot.yaml` 的 `runner.modes` 是死字段

`configs/pilot.yaml:48-54` 写着 6 个模式：

```yaml
modes:
- llm_only
- slm_only
- static_route
- guarded_fallback
- oracle
- traceoffload
```

但代码里 `PILOT_MODES`（`runner.py:122-127`）只有 4 个：

```python
PILOT_MODES = ("llm_only", "slm_only", "random_route", "classifier_route")
```

而这个 yaml 字段**根本不是 runner 读的** —— runner 读的是
`runner.history_store.modes`（`runner.py:459`）。`pilot.yaml` 里的
`runner.modes` 是**遗留死字段**。

`runner.py:466-468` 明确说了这个设计：

> 未知模式**忽略**而不是报错：老配置里还写着已移除的 mode 名，
> 直接拒绝会让一份能跑 llm_only 的配置因为无关字段而失败。

> **注意这个取舍**：配置校验有时候应该"忽略未知键"而不是"报错"，
> 尤其当老配置里有过期字段时。判据是"这个字段会不会影响行为"。

### 12.2 `docs/ARCHITECTURE.md` 提到的 `runner.classifier.feature_builder`

[ARCHITECTURE.md:92](ARCHITECTURE.md#L92) 把它列为配置键，
但 `configs/` 下**没有任何 yaml 有这个键**。它在代码里只作为一个默认值
出现在 `experiments/run_appworld.py:151`。

### 12.3 `docs/DATA_PACKAGE.md` 不存在

README 的「文档地图」和「已知边界」都指向 `docs/DATA_PACKAGE.md`，
但 `docs/` 下**没有这个文件**。`experiments/check_data.py` 是存在的，
但校验流程的文档缺失。

### 12.4 手臂只有 3 个，不是 5 个

README:31 说「`configs/arms/*.yaml`（五个臂）」。实际 `configs/arms/` 下只有：

- `offload-active.yaml` —— 路由器生效
- `offload-shadow.yaml` —— 路由器算但不生效（对照组）
- `m4-on.yaml` —— 路由器 + M4 记忆

另两个臂（`native-large` / `native-small`）不在 `configs/arms/` 里，
它们是**原生 baseline**，走 `experiments/run_native.py`，不经过 TraceOffload。
`run_round.py:121` 的帮助文本说得很清楚：
*「臂名（= configs/arms/<arm>.yaml 或 native-large/native-small）」*。

### 12.5 工具白名单取决于用哪个 config

`configs/pilot.yaml` 的 `tool_policy.allowed_tools` 写的是 `bash`；
但 `configs/arms/*.yaml` **三个臂都没有 `tool_policy` 这一段**。

合并逻辑在 `experiments/run_appworld.py:174-198` `_prepare_runner_config`：
**浅合并**（顶层键缺失才用默认值，嵌套 dict 整体覆盖）。实测：

| 用的 config | 最终 `allowed_tools` |
|---|---|
| `configs/pilot.yaml` | `['bash']` ← **不适用于 AppWorld** |
| `configs/arms/*.yaml` | `['execute_code_batch']` ← AppWorld 正确值 |

`execute_code_batch` 来自 `_runner_defaults()`（`run_appworld.py:132-146`），
常量定义在 `integrations/appworld.py:92`。

> **学习要点**：`pilot.yaml` 是**通用 base**，不是 AppWorld 可直接跑的配置 ——
> 直接用它跑 AppWorld 会让白名单停在 `bash`，与 AppWorld 的
> `execute_code_batch(codes)` 工具对不上。跑实验一律用 `configs/arms/` 下的。

---

## 13. 在 Windows 上跑测试

本仓的代码**是为 Linux 写的**。在 Windows 上 `python -m pytest tests` 会集体失败：

```
AttributeError: module 'time' has no attribute 'clock_gettime'
```

**根因**：`traceoffload/clock.py:56` 和 `:76` 用了
`time.clock_gettime(CLOCK_MONOTONIC)` / `time.clock_gettime(CLOCK_REALTIME)`。
这是 **Unix 专有 API**，Windows 的 `time` 模块没有。

**为什么这个模块这么写**（`clock.py:1-43` 的长注释值得读）：AppWorld 在
world 存续期间装着 freezegun，而 freezegun 会**遍历 `sys.modules`**，
把任何「值等于 `time.perf_counter`」的模块级名字就地替换 —— 连
`from time import perf_counter` 这种写法也中招。只有直接进 libc 的
`clock_gettime` 不受影响。

> 实测（受控实验，world 内真睡 0.2s）：
> ```
> world 外: perf_counter Δ=200.116 ms   __module__='time'
> world 内: perf_counter Δ=  0.000 ms   __module__='freezegun.api'
>           clock_gettime Δ=200.437 ms  ← 只它不受影响
> ```

而且这个坑**已经咬过三次**（`clock.py:26-31`）：`embedding_latency_ms`
（4/4 恒 0）、`ToolExecutionResult.latency_ms`（线上 4/4 恒 0，
而且会被 executor 累加进 phase latency，**让整条 span 的时延系统性低估**）。

**跑法**：本仓带了 pytest 插件 `tests/winclock.py`，它只在
`time.clock_gettime` 缺失时补一个等价实现（在 Linux 上加载也不改变行为）。

```bash
python -m pytest tests -q                          # Unix
PYTHONPATH=tests python -m pytest tests -q -p winclock   # Windows
```

实测：**373** 个测试里 **371 通过、2 失败**，2 个失败都是环境问题
（AppWorld 没 `pip install -e`；freezegun 冻结了 shim 用的 `time.time`）。

> **学习要点**：看到一批测试集体失败时，**先找共同根因**。
> 这个仓的 `clock.py` 就是典型例子 —— 22 个失败、1 个根因、2 行代码。

---

## 14. 建议的学习顺序

```
第 1 轮（半天）  contracts/routing.py → interfaces.py → ARCHITECTURE.md
                 目标：搞懂 RoutePlan 的形状，以及它为什么被强制成那样

第 2 轮（半天）  m2/routers.py → m2/predictor.py
                 目标：搞懂「什么时候用小模型」这个决策是怎么算出来的

第 3 轮（1 天）   m1/ → m4/
                 目标：搞懂特征从哪来、失败怎么被记住、为什么不自激

第 4 轮（1 天）   m3/checkpoint.py → m3/executor.py → integrations/appworld.py
                 目标：搞懂「出错了怎么回滚」和「状态机怎么保证不乱」

第 5 轮（半天）   coordinator.py → mechanism_counters.py
                 目标：把上面几块串起来，并理解可观测性为什么是必需品

第 6 轮（半天）   router/ + experiments/
                 目标：搞懂模型怎么训出来的、实验怎么跑、指标怎么定义
```

### 一个自测题

读完前 5 轮之后，试着回答：

> 一个 span 走了小模型，pre-check 守卫放行了，工具也执行了，
> 但 post-check 发现产物不对。此时：
> 1. 发生了 rollback 吗？
> 2. fallback 到大模型了吗？
> 3. 这条记录进 M4 库了吗？

<details>
<summary>答案</summary>

**1. 视有无副作用而定。**

`executor.py:1195-1201`：

```python
needs_recovery = bool(planned_write_ids or observed_risky_ids)
rollback_status = (
    self._rollback_transactions(rollback_ids)
    if needs_recovery
    else RollbackStatus.NOT_NEEDED
)
```

动作是只读 ⇒ `NOT_NEEDED`（没东西可回滚）；动作是写 ⇒ 真回滚。

回滚失败（`FAILED`/`PARTIAL`/`UNKNOWN`）会 `retain_for_recovery`
保留 checkpoint（:1207-1213），并给相关记录打 `fault_class="env_fault"` ——
环境的锅，不算模型业务失败。

**2. 会。**

`coordinator.py:1180-1193` `_should_fallback` 要求四个条件同时成立：

```python
route.execution_mode == ExecutionMode.GUARDED_SLM
and route.fallback_plan is not None
and result.status in (MODEL_FAILED, TIMEOUT, VERIFICATION_FAILED)
and not result.state_outcome.recovery_required
```

post-check 拒绝 ⇒ `VERIFICATION_FAILED` ⇒ 满足。
fallback 走 `_start_fallback_episode`（:1195-1250），**强制**
大模型 + DIRECT + `AttemptRole.LARGE_MODEL_FALLBACK`。

**3. 不进。**

`memory.py:1229-1246` `_span_had_an_executed_failure` 只认
`ToolExecutionStatus.FAILED`。这里的工具**执行成功了**，
只是产物被 post-check 拒了 —— 这不是「这个动作会失败」的证据。

> 对照 `docs/RESULTS.md` 第二轮的第三条自洽：
> *「post-check 拒 ≠ rollback：m4-on 有 31 次 post-check 拒但只有 19 次回滚
> —— 差额是只读/未知效应，没有可回滚的东西。」*

</details>

---

## 附：关键文件行数速查

| 目录 | 文件数 | 总行数 | 什么时候读 |
|---|---:|---:|---|
| `traceoffload/` | 47 | ~20000 | 主运行时（在线） |
| `router/` | 11 | ~1200 | 离线训练链路 |
| `experiments/` | 7 | ~2200 | 实验驱动 |
| `tests/` | 31 | ~8000 | 规格说明 |

`traceoffload/` 里最该读的 6 个文件（按学习顺序）：
`contracts/routing.py` → `m2/routers.py` → `m2/predictor.py` →
`m4/span_features.py` → `m3/checkpoint.py` → `coordinator.py`
