# four_features 重跑：失败留存并继续

按用户确认，任务上下文超限直接记为失败，保留已发生的用量，不重新生成代码、不压缩输出、不覆盖补跑；继续固定清单的后续任务。技术接入验收与任务成功率分开。

修改内容：

- `experiments/task_outcomes.py` 观察原生 Agent 的 termination 日志调用；日志关闭时也能捕获。native 和 TraceOffload 两个入口统一写入 `task_failure_reason`。
- 失败类别：`context_length_exceeded`、`agent_terminated`、`evaluation_failed`、`runtime_error`；成功为 null。
- 单任务 gate 改为 `acceptance_kind=integration`。质量失败或已识别的上下文超限不阻止批跑，仍要求成功 Jev 判断、模型派发和完整审计；未知运行器异常阻止启动。
- 批跑保留失败行继续下一任务，最终 report 写 `task_failures_by_reason`。模型服务整体不可用仍停止。
- 启动时等待 SLM 端点就绪，再启动 Jev；配置快照、源文件 SHA256 和独立输出目录保留。
- 实验专用 SLM 的元数据写独立文件，避免覆盖保留服务的信息文件；退出显式清理本次新提交的 SLM 作业。

验证证据：本地 WSL 全部 543 项测试通过；最后一次针对性本地测试 27 项通过；服务器同组 27 项通过，Python 编译及 Bash 语法检查通过。另用真实 AppWorld Agent 连续运行两个任务，将 LLM HTTP 用本地 MockTransport 模拟为上下文超限，两个任务均被捕获为 context_length_exceeded、分别保留记录，第二任务正常进入；两任务的技术接入验收均通过。该冒烟没有真实模型推理，不作为任务质量结果。本地产物 `runs/context-failure-smoke-a5a56ed9e148/`。

本次提交：

| 作业 | 用途 | 资源与退出 |
|---|---|---|
| 305899 | 实验专用 Qwen3.5-9B | node04 单张 A800，最长 4 小时；305900 退出时自动取消 |
| 305900 | Jev、技术验收、固定 90 任务 | CPU 2 核；Jev 通过 step 共用 305899 GPU |
| 284851 | 用户要求保留的独立服务 | 不参与本次清理，不操作 |

LLM 沿用共享 Qwen3.8-27B 端点，仍受共享服务条件限制；SLM/Jev 地址沿用 node04。输入模式为 four_features，阈值 0.8，字符限制 12000，M4 关闭，清单为 `experiments/instances-90.txt`；没有调提示词或阈值。

服务器输出根目录：

```text
/share/home/shenwsh3/ozy/span-offload-jev/runs/jev-four-features-90-305900/
  config.yaml
  source-hashes.json
  single/acceptance.json
  single/appworld-classifier_route-summary.json
  batch/r1/appworld-classifier_route-summary.json
  batch/r1/progress.jsonl
  batch/r1/report.json
  service/
```

作业日志：`runs/jev-four-features-305900.log`、`runs/slm-fourfeat-305899.log`。单任务技术验收轨迹单独保存，不加入 90 任务统计。

文档记录的是提交与验证状态，完整任务结果以 batch/r1 中的最终文件为准，不能把 Slurm RUNNING 当作已开始模型调用或全部完成。
