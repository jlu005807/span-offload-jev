# 固定 Span 对照实验：已有产物检查

检查日期：2026-10-26。只读检查，无新增 GPU 作业。

## 已确认的产物

服务器项目：`/share/home/shenwsh3/ozy/span-offload-jev`。

| 项目 | 检查结果 | 用途与限制 |
|---|---|---|
| 调用前完整请求 | `runs/jev-context-90-518659/batch/r1/classifier_route/history.sqlite3` 有 1825 条 `span_records`；全部有 `model_request.messages` 和 `native_payload` | 可作为固定前缀来源；尚未逐条校验消息完整性与跨产物关联 |
| Jev 实际输入 | 三组 `routing-decisions-*.jsonl` 保存 `request.state` 与概率 | context 保存 context；four_features 保存 features；combined 同时保存两者。不同运行的步骤不能直接视作相同 Span |
| 四特征 | combined / four_features 审计样本均有 `prefix_n_A`、`log_est_prompt_tokens`、`V2_goal_recent`、`C_bare_w10` | 建议优先从同一次 combined 运行选样本，核对对应完整请求，保留原有特征与构建元数据 |
| context 的四特征 | 抽查第一条 `runtime_profile.online_features` 为空 | 不能声称 context 历史库已经包含完整四特征；若用该轨迹需按原实现重建并核验 |
| 执行结果 | 历史记录含 `execution_summary`，AppWorld 输出含 `environment_io.md`、`api_calls.jsonl`、任务评测及最终数据库产物 | 可检查已执行分支，不能当作未执行 SLM 分支的标准答案 |
| 每步环境快照 | runs 下未发现 checkpoints 目录；原代码事务提交会删除环境快照及副本 | 暂不能直接恢复任意调用前状态；最终数据库文件不等于每步完整状态 |
| 原标签包 | 当前服务器项目无 `data-package` | 未获得同前缀的 `l2_new` 和 `net_saving` 标签 |

纯上下文批跑的原生 AppWorld 输出目录：

`appworld/experiments/outputs/jev-batch-0cefb232efbf4896a415553705f0148e`。

检查任务 `099174d_1` 看到最终 `dbs/*.jsonl`、日志、使用量和任务评分，没有逐步快照。SQLite 的 `checkpoint_sessions`、`outcome_events`、`feedback_snapshots` 表均为空；这些表名本身不代表已经保存了可重放状态。

## 能立即做什么

从同一次 combined 运行抽取固定 Span，关联完整请求、任务目标、四特征、调用标识和输入构建元数据，重新构建三种输入并核验。这样可以公平比较同前缀上的概率、候选卸载和输入可用性，但不能计算局部正确率或真实净节省。

## 判断哪种提示词最准确还缺什么

优先获取学长已有同前缀的分支执行数据、标签及生成规则。若无法取得，需要恢复采集流程或重放前缀后保存环境状态，执行 LLM / SLM 对照分支。重放是否能准确还原数据库、执行命名空间及其他状态必须单独验证，不能仅凭日志存在就宣称可恢复。

任务成功标签不能代替每步 `l2_new`；未执行 SLM 的步骤不能从 LLM 结果推导 SLM 标签。原路由预测也不能代替真实标签。
