# fallback 与 rollback 的实测

数据源：跑批落盘的 span 记录 `<run>/classifier_route/history.sqlite3`。
用法：`python3 experiments/verify_mechanisms.py runs/one_round/*/r1`

⚠️ **不要**看 summary 里的 `fallback_by_reason` / `checkpoints_created` / `transactions_rolled_back`：
这几个计数位在代码里**从未被 bump**（基线版本也没有），永远是 0。

## 两个现象的定义

| 现象 | 记录里的样子 | 触发条件 |
|---|---|---|
| **fallback** | `span_kind="fallback"`、`span_relation="fallback_of"` 的 child span | 父 span 是 `verification_failed` 或 `MODEL_FAILED/TIMEOUT`，且 `recovery_required=False` |
| **rollback** | 工具执行的 `rollback_status ∈ {succeeded, failed}` | 动作是 `REVERSIBLE_WRITE` 且建过 checkpoint，且 post-check 拒绝 |

两者是一条链：**守卫拒 → 回滚（若已执行）→ fallback 大模型**。
另有一条短路：**pre-check 拒（M4 记忆）→ 没执行 → 直接 fallback，无需回滚**。

## 一轮实测（每臂 90 个实例）

| 臂 | span 总数 | fallback | pre_commit 拒（M4 记忆） | post_check 拒 | rollback succeeded |
|---|---:|---:|---:|---:|---:|
| offload-active | 1936 | 26 | 0 | 26 | 21 |
| offload-shadow | 1813 | **0** | 0 | 0 | **0** |
| m4-on | 1835 | 35 | **2** | 33 | 25 |

三条结论：

1. **fallback 是真发生的**，26/35 次，父 span 全部是 `verification_failed` 且 `post_check=reject`。
2. **rollback 也是真发生的**，21/25 次成功；剩下的 post_check 拒没回滚，是因为动作是只读或
   未知效应（`rollback_status=not_needed`）—— 没有可回滚的东西，不是失败。
3. **shadow 臂是干净的对照**：预测器照算，但执行全走大模型，于是守卫一次都没跑，
   fallback 与 rollback 都是 0。这反过来证明前两行的数不是「机制空转」。

## 一条完整链路（offload-active，task 2c544f9_2）

```
episode-15  guarded_slm  →  pre_commit: pass
                          →  工具执行 execute_code_batch（declared_write）失败
                          →  post_check: reject
                          →  rollback_status = succeeded（回滚到 span 级 checkpoint）
                          →  span status = verification_failed
     └─ child: fallback span → target=large_model, reason=slm_fallback（span_relation=fallback_of）
```

## 一条"执行前拦截"（m4-on，task 7238049_2）

```
episode-40  guarded_slm  →  pre_commit: reject（M4 失败记忆命中同一动作签名）
                         →  model_attempts=1, tool_executions=0   ← 什么都没执行
                         →  span status = verification_failed
     └─ child: fallback span → large_model, reason=slm_fallback
```

这条是 M4 真正起作用的样子：**在执行之前**拦下，所以没有回滚、也没有副作用。
