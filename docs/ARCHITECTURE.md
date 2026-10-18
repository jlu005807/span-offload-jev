# 架构

四层，两个接缝。Agent 本体不改，只包住 `model.generate` 与 `env.execute`。

```
Agent（ReAct 循环，不动）
   │ ① model 接缝            │ ② env 接缝
   ▼                          ▼
TraceOffloadModelWrapper   AppWorldToolProxy
   └──────────┬───────────────┘
              ▼
          Coordinator ── M1.build_profile → M2.decide → M3.execute
              │                                        │
              └────────── M4.observe ←── 记账 ──────────┘
```

## 一次调用的生命周期

1. Agent 调 `model.generate` → wrapper 拦下，开一个 episode
2. **M1** 建画像：估 token、建两段带嵌入的文本（V2_goal_recent / C_bare_w10）
3. **M2** 出 RoutePlan：
   - 取不到 116 维向量 → 大模型 DIRECT
   - `p_gate ≥ τ_gate ∧ p_net ≥ τ_net` → 小模型 **GUARDED_SLM**
   - 否则 → 大模型 DIRECT
4. **M3** 执行：
   - DIRECT：一次调用，成功即接受
   - GUARDED_SLM：小模型出候选 → pre-check 守卫（形状 + M4 失败记忆）→
     过不了就 REJECT_AND_FALLBACK；过了就建 checkpoint，返回 PREPARED
5. Agent 拿到响应去调工具 → ToolProxy 真执行 → post-check → commit / rollback
6. 记账：SpanRecord 进 EventStore；有执行失败就进 M4 记忆

## M1 建模（`m1/`）

| 文件 | 作用 |
|---|---|
| `profiler.py` | RuntimeProfiler（单遍）+ ClassifierFeatureProfiler（补在线特征） |
| `span_embedding.py` | 两段文本配方：V2_goal_recent、C_bare_w10（最近 10 个动作签名） |
| `signature.py` | 动作签名规范化（离线/在线**同一份实现**） |
| `embedding_client.py` | Ollama /api/embed 客户端 |
| `appworld_features.py` | 两个前缀标量：prefix_n_A、log_est_prompt_tokens |

## M2 路由（`m2/`）

`predictor.py` 读 artifact，纯 Python 前向：116 维 = [2 标量] + PCA50(v2) + PCA64(actions)，
两个 logit 出 `(p_gate, p_net)`。

`routers.py` 四种模式：`llm_only` / `slm_only` / `random_route` / `classifier_route`，
外加 `_ShadowRouter` 装饰器：照常算、照常记录，但把整个 plan 换成大模型 DIRECT。

**契约**：小模型目标必须 `GUARDED_SLM` 且带 pre-commit + 恰好一个 post-check；
大模型目标必须 `DIRECT` 且无 fallback。构造 RoutePlan 时校验。

## M3 受控执行（`m3/`）

| 文件 | 作用 |
|---|---|
| `executor.py` | 两种执行模式；准入、守卫、超时、fallback |
| `checkpoint.py` | span 级 checkpoint 会话 |
| `tool_proxy.py` | 事务边界（execute / begin / commit / rollback） |
| `guards.py` | 形状守卫 + M4 失败记忆守卫（继承而非替换） |
| `integration.py` | AdapterSession：把工具历史投影成 prior facts |

副作用分级由接入层注入：AppWorld 用 `integrations/appworld.py` 的
`AppWorldCodeEffectAnalyzer`（读 OpenAPI 的 HTTP method）。认不出的动作一律
fail-closed（先建快照再执行，或直接拒绝）。

## M4 记忆（`m4/`）

- `memory.py`：建库、检索（动作签名精确匹配 ∧ 场景余弦 ≥ 阈值）、记账
- `store.py`：SQLite 持久化（维度闸防止换向量空间后读到老库）
- `span_features.py`：52 维记忆向量（PCA50 + 2 个标量）
- `action_guard.py`：守卫查询入口

**写库判据**：本 span 里存在一个执行失败的工具。不是「本地没成功」——
准入被拒、post-check 拒但工具成功，这两类都不收。

## 铁律

1. 组件之间不互相 import，用 `interfaces.py` 的 Protocol 组合
2. 所有「增强」（M4 记忆、预测器、动作守卫）都是 fail-open，且必须可观测
3. 换预测器、换守卫、换执行模式，都不许改 Agent 代码

---

## 配置键与装配

装配发生在 `experiments/run_appworld.py` 的 `AppWorldPilotRunner.__init__` / `_build_coordinator`：
它先 `_prepare_runner_config(raw)` 把默认值补全，再构造 M1–M4 并注入 Coordinator。

| 配置键 | 作用 | 谁读 |
|---|---|---|
| `runner.classifier.feature_builder` | 在线特征构建器（`appworld`） | `runner._make_profiler` |
| `runner.l2_add4.enabled / model_path` | 116 维两阶段预测器 | `runner.__init__` → `_make_router` |
| `runner.l2_add4.tau_gate / tau_net` | 覆盖 artifact 里的阈值（敏感性实验用） | `runner._l2_add4_lookup` |
| `runner.shadow_route.enabled / force_target` | shadow 臂：决定不生效 | `runner._make_router` 包一层 `_ShadowRouter` |
| `runner.m4_memory.enabled` | 是否建 M4 记忆库 | `runner._start_m4_memory` |
| `runner.m4_memory.embedding_mode` | 必须是 `m1_online`（向量来自 M1） | `runner._start_m1_embedding` |
| `runner.m4_memory.template_version` | 必须是 `span-embed-v2`（add4 的第 ③ 段） | 同上 |
| `runner.m4_memory.memory_guard_threshold` | 记忆守卫的余弦阈值（默认 0.8，M4 臂用 0.7） | `m4/memory.py` |
| `runner.tool_policy.allowed_tools` | 工具白名单（AppWorld 是 `execute_code_batch`） | `m3/guards.py` |
| `runner.history_store.*` | span 记录落盘（sqlite / memory） | `runner._make_history_stores` |
| `runner.policy.*` | 路由策略（预算、utility 权重） | `runner._route_policy` |

**四条构造期硬约束**（配错直接抛，不留到跑起来才炸）：

1. `classifier_route` 必须配 `l2_add4.enabled: true`（本仓库只保留这一个预测器）
2. `l2_v4` 已移除 —— 配成 true 会明确报错，而不是静默降级
3. `l2_add4` 要求 `template_version: span-embed-v2`
4. `l2_add4` 要求 `embedding_mode: m1_online`

