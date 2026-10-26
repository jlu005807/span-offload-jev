# four_features 批跑未启动的排查（2026-10-23）

已确认 four_features 路由可以工作；两次尝试均未进入完整 90 任务批跑，原因不同。

## 第一次：269315 / 269390

- 单任务 `099174d_1` 成功，AppWorld 7 项通过、0 项未通过。
- Jev 37 次请求全部成功，无输入不可用，全部选择 LLM。
- 随后 `run_batch.py` 在开始执行任何批跑任务之前，调用 `check_services` 时发生 `httpx.ConnectError: Connection refused`。
- 当时健康检查日志未记录具体失败 URL，不能确定是 LLM、SLM 还是 Jev 端点；启动阶段针对 Jev 的 curl 拒绝连接记录本身不能证明批跑失败端点就是 Jev。
- 全量任务未启动。原始结果保留于服务器 `runs/jev-four-features-90-269390/`。

## 第二次：295083 / 295084

- Jev 正常：18 次请求全部成功，全部选择 LLM，无输入超限、超时或审计错误。
- Agent 在第 16 步生成分页搜索代码，传入不支持的 access_token 参数而执行失败；第 17 步删除该参数后继续同一搜索方案。
- 搜索方案使用 while 循环逐页收集全部结果，并执行 `print(results)`。
- 随后的环境输出含 1840 个商品（1840 个唯一 product_id），文本长 1,090,992 字符。
- 下一次 LLM 请求被服务拒绝：输入至少 253,953 tokens，加上请求的 8,192 输出 tokens，至少为 262,145，超过服务上限 262,144。
- Agent 在购买前终止。评测结果为 1 项通过、6 项未通过；没有新增订单。
- 单任务验收要求 task_success=true，因此 ready_for_batch=false，批跑脚本退出码为 2，90 任务未启动。

关键区分：此处超限的是执行 Agent 的 LLM 完整历史，和 Jev 的 4096 状态 token 预算是两条独立链路。改短 Jev 输入不能解决 Agent 的超长工具返回。

AppWorld 日志中的 termination 保存了真实 400 错误，但外层 summary 的 failure=null：Agent 内部记录终止后返回，外层未捕获到抛出异常。因此不能把 failure=null 解释为整个执行过程没有模型错误。

服务器证据：

```text
runs/jev-four-features-90-295084/single/acceptance.json
runs/jev-four-features-90-295084/single/appworld-classifier_route-summary.json
appworld/experiments/outputs/jev-099174d_1-7346beec1be64bf4bfa932d0c73bfbf8/tasks/099174d_1/logs/logger.jsonl
appworld/experiments/outputs/jev-099174d_1-7346beec1be64bf4bfa932d0c73bfbf8/tasks/099174d_1/evaluation/report.md
```

本地已保存提取的排查证据于 `runs/jev-four-features-diagnosis/`；完整轨迹仍在服务器。

## 修复建议

1. 对执行 Agent 增加结构化的长工具输出保护：完整结果落盘，提供有界的预览和访问完整结果的方式，避免全部内容进入后续提示。该修改应对各对照臂一致生效，并记录版本。
2. 发送 LLM 请求前检查完整上下文预算，保留必需指令、任务目标和最新相关信息；仅把 max_tokens 减少一点只能暂时绕过本次错误，无法解决后续上下文继续增长。
3. 汇总 Agent termination 原因，避免只看外层 failure 字段而遗漏模型错误。
4. 健康检查已有有限重试；仍应记录失败的服务角色、端点和错误类别，便于区分服务故障，不能仅凭连接错误推定某个模型服务崩溃。
5. 保留两次失败尝试，不用反复挑成功轨迹覆盖原结果；任务未成功应作为实验结果保留。技术接入验收与任务质量统计需要明确区分。

本次仅完成原因排查及文档整理，没有为长输出保护修改执行逻辑，也没有重新申请资源。295083 已取消；用户指定保留的 284851 未操作，排查末次查询仍正常运行。

## 后续验证：原生 native-large 单任务

按用户要求，用原有 `experiments/run_native.py --model large` 对同一任务 `099174d_1` 跑一次，无 Jev/TraceOffload 路由，无新增长输出保护。调用同一 Qwen3.8-27B 服务，使用之前保存的模型与解码配置，40 步上限；dataset_name 标为该任务真实所属的 test_challenge，仅用于元数据记录。

| 指标 | 原生基线本次结果 |
|---|---:|
| task_success / task_completed | true / true |
| 评测通过 / 未通过 | 7 / 0 |
| LLM 调用 | 34 |
| 实际任务耗时 | 127.696 秒 |
| 模型调用累计耗时 | 84.742 秒 |
| 输入 / 输出 tokens | 596154 / 4574 |
| 总 tokens | 600728 |
| failure | null |
| 最终 environment_io.md 大小 | 60724 bytes |

Slurm 303094 正常完成，ExitCode=0:0，总作业时间 00:09:21。申请 2 个 CPU 核、15 分钟上限，没有新申请 GPU；主要额外耗时来自共享文件系统上的依赖冷加载。期间用于查看导入位置的 5 秒 strace 诊断 step 因自身 timeout 记为 124，不是基线任务失败。主作业结束后 CPU 资源自动释放，284851 保持运行。

本地证据：`runs/native-large-check-303094/`，包括 result.json、config.yaml、evaluation-report.md、environment_io.md、slurm-status.txt。服务器输出实验名 `native-large-check-303094`。

本次没有复现四特征失败尝试中的超长输出，证明该任务能由原生基线完成；不能证明原链路已经具备长输出保护，也不能说明四特征必然导致失败。第一次四特征尝试同样成功，第二次才发生超限。要归因于接入差异或共享模型的执行波动，还需要比较实际请求和轨迹，单次成功不足以作因果结论。

### 轨迹对照后的进一步结论

成功 native-large 轨迹先查看第一页结果，再定义 find_exact_product；匹配到商品名称便 return，只打印所选商品的 ID、名称、价格等摘要，随后完成购物车及下单操作。失败 four_features 轨迹则把所有分页结果 extend 到 results，最后 print(results)，一次产生 1840 个商品的完整文本。直接导致上下文膨胀的是这个输出策略差异。

两个 four_features 尝试均只派发 LLM，没有 SLM 调用：第一次 37 次有效 Jev 判断后任务成功，第二次 18 次有效判断后任务因 LLM 上下文超限终止。当前证据不支持“Jev 选错成 SLM 导致失败”。Jev 选 LLM 也不保证 LLM 后续代码正确，长输出保护仍需执行链路处理。

代码检查显示 AppWorldLanguageModelGateway 使用 runtime 保存的原生 messages/tools/kwargs 调用原生 LanguageModel.generate；Jev 的判断提示词单独发给 Jev，没有直接拼到执行 LLM 的任务提示词中。随后进行了下述固定前缀离线 HTTP 正文对账；原两次任务的历史在线 HTTP 原始请求没有完整捕获，因此不声称已完成历史在线抓包对账。

共享服务的并发批次、运行时数值非确定性、服务重启/配置差异和接入层行为差异都仍是轨迹波动的待排查因素；现有单轮轨迹不能量化或选定其中一个。若要验证接入透明性，应对相同固定前缀记录并比较 native 与 wrapped 实际请求，再在受控服务条件下重复，而非把不同轨迹的最终成功/失败直接归因于 Jev。

## 固定前缀离线 HTTP 正文对账

从失败运行保存的 `classifier_route/history.sqlite3` 提取第 1、17、18 次调用的完整 model_request/native_payload，并从审计日志提取对应的成功 Jev 响应。将同一前缀分别送入：

1. 原生 AppWorld LanguageModel.generate。
2. 生产 AppWorldLanguageModelProxy → Coordinator → JevRouter → ControlledExecutor → AppWorldLanguageModelGateway → 同类原生 LanguageModel.generate。

Jev 使用历史响应回放，三次均选择 LLM；LLM 使用真实 OpenAI SDK 序列化，但由 httpx.MockTransport 截获请求并返回固定成功响应，没有远程模型推理和 Agent 工具执行。HTTP 正文在 SDK 完成序列化后比较。没有新增服务器作业或 GPU 申请，284851 未操作。

| 真实调用前缀 | 原始消息数 | HTTP URL/方法/正文一致 | 正文字节 SHA256 一致 | 固定响应返回 Agent 后一致 |
|---|---:|---|---|---|
| 第 1 次 | 23 | 是 | 是 | 是 |
| 第 17 次（生成大量打印代码前） | 55 | 是 | 是 | 是 |
| 第 18 次（已包含超长工具输出） | 57 | 是 | 是 | 是 |

两条路径的输入对象在调用后也均保持不变。正文包含相同 messages、model、max_tokens、parallel_tool_calls、seed、temperature、top_p；Jev 路由提示词未进入执行 LLM 的消息。四个关键源文件与服务器对应文件在归一化换行后 SHA256 一致：run_native.py、run_appworld.py、integrations/appworld.py、AppWorld language_model.py。

本地诊断脚本：`runs/setup/check_payload_transparency.py`。完整比较报告：`runs/jev-payload-comparison/report.json`；该文件记录独立 attempt 目录，其中保存各前缀的 direct-request.json 和 wrapped-request.json。原始前缀也保留在同目录的 recorded-prefixes.json。`runs/` 为忽略的本地实验产物。

结论边界：在这三个真实前缀、已记录的 LLM 路由选择和本地实际客户端代码下，没有发现接入层改写请求或固定响应的问题。该验证不模拟服务端上下文长度校验，也不证明实际模型输出可逐位复现；不能把本地固定成功响应解释成 256K 超长请求在线能够成功。仍未通过受控的在线重复推理确认轨迹波动的具体来源。

## 用户确认后的失败处理修订

按用户要求，保留原有 Agent 输出策略，不增加代码重试或上下文压缩。通过观察原生 logger.show_message 的 termination 边界捕获 Agent 内部吞掉的模型错误，在 native 和 TraceOffload summary 中添加 task_failure_reason；上下文超限记为 context_length_exceeded，task_success=false，已有 token/调用记录保留。

技术验收改为 integration：检查明确任务结果、真实 Jev 成功请求、实际模型派发和完整审计，不再要求所选购物任务必须成功。任务质量失败和已识别的上下文超限允许进入批跑；初始化等未知运行器异常仍阻止启动。完整批跑保留每个失败任务并继续下一项，最终报告增加 task_failures_by_reason。不同服务整体不可用时停止，不对失败任务自动覆盖补跑。

修订同时增加 SLM 端点真正就绪后的启动顺序和实验专用 SLM 的显式退出清理，避免作业启动即假定模型已就绪，或 CPU 批跑失败后 GPU 仍独立占用。用户要求保留的 284851 不参与该清理。
