# OpenJev 实机单任务验收与 90 任务批跑

日期：2026-10-23。本文区分已完成的单任务验收与仍在运行的批跑，不将作业提交成功当成全部任务完成。

## 运行位置与环境

- 项目：`/share/home/shenwsh3/ozy/span-offload-jev`，基础提交 `87b451d`。
- 本次补充 `deploy/jev/` 部署入口与测试，未修改核心 Router、Coordinator、M3 或 AppWorld 任务/评测源码；补充代码尚未提交。
- Agent：项目 `.venv`，Python 3.11.13；依赖约束见 [appworld-constraints.txt](../deploy/jev/appworld-constraints.txt)。
- Jev：`~/ozy/jev-service-env`，在 `~/singularity/vllm_qwen35_sandbox` 内运行，Python 3.12.3，`wev-ai==0.2.2`。
- Jev 模型：`~/ozy/models/OpenJev-1.7B`，四个关键文件的下载元数据均指向 revision `7418ae8b6e3b6a6507a8a58729ae491cd1c732d9`。
- `/v1/models` 返回的部署模型名是 **`wev-1.7b`**，不是 Hugging Face 仓库名。

实际服务：

| 用途 | 型号 | 地址 |
|---|---|---|
| LLM | Qwen3.8-27B | `http://192.168.10.25:8101/v1` |
| SLM | Qwen3.5-9B | `http://192.168.10.26:8800/v1` |
| Jev | OpenJev-1.7B / `wev-1.7b` | `http://192.168.10.26:8900` |

Jev 与自己的 SLM allocation `251550` 共用一张 A800。未停止原 SLM 作业；重复申请的 LLM 排队作业 `263435` 已取消，改为复用可用端点。

## 原生编码与接口验收

`openjev_preflight.py` 与实际安装版本的 `wev.api.to_record` 和 `wev.model.encode(strict=True)` 逐 token 比较：3 个正常案例相同，2 个超长案例均拒绝；覆盖中文、嵌套数据和特殊 token 转义。

实际限制：state 4096 token、单问题 branch 8192 token、模型位置上限 32768。
服务未覆盖这些限制。预检直接拒绝超限，不调用上游缩短逻辑。

成功任务的 25 条原生响应中，`usage.input_tokens` 与预检计数全部相同。输入 token 合计 67,773；答案 JSON 序列化 token 合计 1,369，后者不是自回归生成量。原生服务耗时合计 2,411ms。

部署 revision 由本地下载元数据固定；原生响应不能证明实际加载的权重版本，因此审计中的 `revision_verified=false` 保持不变，不伪造服务器证明。

## 真实单任务结果

任务：`099174d_1`，按 SimpleNote 购物清单在模拟 Amazon 下单。

配置：active、combined、阈值 0.8、M4 关闭、最大 Agent 步数 40。

| 指标 | 实测 |
|---|---:|
| task_success | true |
| AppWorld 评测通过 / 失败 | 7 / 0 |
| 运行时 failure | null |
| 端到端耗时 | 144,522ms |
| 实际 LLM / SLM 派发 | 28 / 0 |
| LLM/SLM total_tokens | 399,533 |
| Jev 决策 | 28 |
| Jev HTTP 尝试 / 成功 / 错误 | 25 / 25 / 0 |
| 输入投影超限，未发送 | 3 |
| Jev 意向 LLM / SLM | 25 / 0 |
| Jev 客户端累计耗时 | 3,328ms |
| 审计完整 | true |

Jev 客户端耗时已包含在任务端到端耗时中，不再相加。本任务没有发生正常阈值下的卸载，不能据此宣称节省或路由质量提升。费用未知，保持 null。

成功结果目录：

```text
runs/stage2-20261023/active-099174d_1-third/
  acceptance.json
  appworld-classifier_route-summary.json
  config.yaml
  manifest.json
  native-usage-validation.json
  routing-decisions-*.jsonl
```

完整 AppWorld 环境执行、API 记录和任务输出：

```text
appworld/experiments/outputs/jev-099174d_1-e856cceb9a624241b685a5d205d46c60/
```

前两次失败保留在 `active-099174d_1-first/second`，未覆盖。第一轮因新版 FastAPI 不接受 `Path(embed=...)` 而初始化失败；第二轮因新版 SQLModel/SQLAlchemy 日期校验导致下单 API 失败。固定兼容依赖后，第三轮成功；没有修改基准 API、任务数据库种子或评分规则。

## 独立机制测试

正常阈值任务没有选 SLM，因此另做两项明确标记的单步测试，不纳入任务质量结果：

1. `mechanism-tau0-one-step`：仅此测试将阈值设为 0、步数设为 1；真实派发 SLM 1 次，守卫检查 1 次。
2. `mechanism-guard-reject`：同一单步条件下，人为拒绝第一次 pre-check；真实派发 SLM 1 次、LLM fallback 1 次，`fallback_by_reason.slm_fallback=1`，仅询问 Jev 1 次。

两个单步测试都不要求完成整个购物任务，不能用它们的任务失败计算路由准确率。正式批跑阈值仍为 0.8。

服务器回归：**534 passed，1 项依赖弃用 warning**。日志为 `runs/stage2-20261023/server-regression.log`。
Shadow 依赖旧路由及嵌入服务，本轮未运行；不宣称已完成 Shadow 对照或全部输入模式比较。

## 90 任务批跑

范围采用 [DATA.md](DATA.md) 的项目固定自定义测试清单：`experiments/instances-90.txt`。
逐 ID 核对后，它跨官方 train 11、dev 4、test_normal 23、test_challenge 52 个任务；不是官方 train 全量，也不是官方测试集全量。与文档旧路由训练/验证集在 task_id 层面无交集。
任务数、去重及每个任务的 specs、dbs、ground_truth 路径已检查。无需下载旧路由训练/验证标签包。

不将官方 dev 57、test_normal 168、test_challenge 417 混入同一个结果。服务器已有官方各划分的 732 个任务定义；正式运行哪一组取决于实验清单。

本轮作业：**264111**，CPU 分区串行运行 Agent，在原 SLM GPU allocation 内启动 Jev step；最长 24 小时。

```text
runs/jev-combined-90-20261023-122426/
  batch-264111.log
  r1/
    instances.txt
    manifest.json
    config.yaml
    progress.jsonl
    appworld-classifier_route-summary.json
    routing-decisions-*.jsonl
    report.json                         # 全部完成后生成
  service/
    jev-server.log
    warmup.json
    gpu-memory.csv
    gpu-process-memory.log
```

AppWorld 完整 Agent 日志、环境交互、API 调用、数据库差量和评分输出位置见 `r1/manifest.json` 的 `appworld_output`。
每个任务结束立即落盘 summary 与 progress，失败保留；健康检查失败则记录剩余任务并停止，不覆盖补跑。

启动后采样的 Jev 进程显存最高为 4,704MiB（约 4.59GiB），共享卡当时仍有约 5GiB 余量；这是周期采样，不是绝对峰值，也没有验证 256K 极限输入。
模型启动与预热另存，不计入任务内路由耗时；GPU价格和未返回响应的服务端计算成本仍未知。

更新：90 任务已全部完成，85 成功、5 未成功，Slurm 作业 264111 以 ExitCode=0:0 正常结束，耗时 02:47:15。SLM/Jev 资源随后按用户要求释放。最终指标、历史实现对比和限制见 [JEV_RESULTS_20261023.md](JEV_RESULTS_20261023.md)。以下命令用于查询归档记录（作业已结束，squeue 不再显示该作业）：

```bash
cd ~/ozy/span-offload-jev
squeue -j 264111
cat runs/jev-combined-90-20261023-122426/r1/progress.jsonl
```
