# 实验协议

## 五个臂

| 臂 | 预测器 | M4 | 说明 |
|---|---|---|---|
| `native-large` | — | — | 全 27B，不接框架 |
| `native-small` | — | — | 全 9B，不接框架 |
| `offload-shadow` | add4 | 关 | 预测器照算，**决定不生效** |
| `offload-active` | add4 | 关 | 接入并卸载 |
| `m4-on` | add4 | **开** | 同上 + 失败记忆 |

干净的两对对比：`shadow` vs `active`（只差「决定生不生效」）、
`active` vs `m4-on`（只差 M4）。四个臂两两之间不能随便比。

## 跑法

- **90 个实例，顺序固定**（`experiments/instances-90.txt`）。M4 记忆跨实例累积，
  顺序变了 M4 那一行就不可比。
- **一轮 = 一次冷启动 + 90 个实例**。M4 每轮从空库开始（`r<k>/m4.sqlite3`）。
- 并发 4（别超 5）：单任务峰值 RSS ~780 MB，且共享同一对端点。
- 同一实例重跑是**原地替换** summary 里的行，不重复追加。

## 指标口径（三列最容易读错）

| 列 | 定义 |
|---|---|
| 成功率 | `task_success` 的均值 |
| **卸载率** | 交给小模型的**调用数** / 总调用数（读 `framework_model_calls_by_target`） |
| 总 token | Σ `total_tokens` |
| **每次调用** | 模型调用数 / 任务数（一个任务平均调多少次模型） |
| 模型延迟 | 每个任务的模型调用总耗时 |

⚠️ 卸载率**不能**从预测器的 p 值反推：shadow 臂上预测器照算、分数全记着，
但执行一律走大模型。按 p 值算会把 shadow 报成 21%，而它实际是 0%。

## 参考结果（本仓库代码 + 旧数据，report.py 逐位复现）

| 臂 | n | 成功率 | 卸载率 | 总 token | 每次调用 | 模型延迟 |
|---|---:|---:|---:|---:|---:|---:|
| native-large | 450 | 89.8% | 0% | 167.4M | 20.7 | 127.0s |
| native-small | 450 | 38.9% | 100% | 114.2M | 25.1 | 189.6s |
| offload-shadow | 450 | 88.4% | 0% | 174.3M | 20.6 | 139.0s |
| offload-active | 450 | 71.6% | 18.7% | 152.2M | 19.8 | 131.8s |
| m4-on | 450 | 69.6% | 19.6% | 150.6M | 20.1 | 127.9s |

（EvoRoute 与 GlimpRouter 是外部基线，实现在 `baseline_Eurosys`，不在本仓库。）

## 验收

1. 每轮 summary 行数 = 90
2. `failure != null` 的行 = 0
3. 每行的位置与实例清单逐位相同
4. `decision_meta.classifier_predictor` = `l2_add4_two_stage`，不是 `*_unavailable`
5. 该臂不要的机制计数全 0（如 M4 关时 `m4_guard_lookups == 0`）
