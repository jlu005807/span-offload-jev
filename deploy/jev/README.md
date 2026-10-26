# OpenJev 单任务部署入口

`openjev_preflight.py` 为 OpenJev 0.2.2 的单个 `choice` 路由问题计算完整输入 token。
它保留 native render、特殊 token 转义、候选顺序和边界 token；state 或 branch 超限直接拒绝，不进行服务端的缩短重试。
本模块只加载 tokenizer，不加载模型权重，不建立额外 HTTP 服务。

实验环境需要 `tokenizers==0.22.2`。首次接入必须将 `encode_ids` 与实际安装的
`wev.api.to_record`、`wev.model.encode(..., strict=True)` 逐 token 比较，并保存核对结果。
更换服务版本、模板、tokenizer 或输入限额后需要重新验证。服务启动时不要使用与模型
`wev.json` 不一致的 `--max_state` / `--max_branch` 覆盖参数。

实验环境的已验证依赖约束见 `appworld-constraints.txt`，与 Jev 的 Python 3.12
服务环境分开。新版 FastAPI 不再接受 AppWorld 使用的 `Path(embed=...)`；新版
SQLModel/SQLAlchemy 的日期类型校验也会阻止基准中的下单 API。固定兼容版本，
不要修改任务、评测器或基准 API 来绕过错误。

运行使用已有 `AppWorldPilotRunner` 的注入接口，不更改通用 CLI：

```bash
cd ~/ozy/span-offload-jev
export APPWORLD_ROOT="$PWD/appworld"
export JEV_MODEL_DIR="$HOME/ozy/models/OpenJev-1.7B"
export JEV_RUN_CONFIG="$PWD/runs/stage2-20261023/active-config.yaml"
export JEV_RUN_OUTPUT="$PWD/runs/stage2-20261023/active-099174d_1-$(date +%Y%m%d-%H%M%S)"
export JEV_TASK_ID=099174d_1
.venv/bin/python deploy/jev/run_single_task.py
```

配置文件必须填写真实 LLM、SLM、Jev 的 IP 地址和服务模型名。每次使用新输出目录；
入口拒绝覆盖现有目录，并保存配置快照。AppWorld 输出名称包含随机 nonce，真实路径
记录在 `manifest.json`；Agent 日志开启。Jev 原生服务独立运行，端口示例为 8900。

单任务现在验收技术链路：必须得到明确的布尔任务结果、有真实 Jev HTTP 成功、审计完整且
存在实际模型派发计数。任务评分失败、Agent 提前终止或已识别的上下文超限不阻止批跑；
未知运行器异常仍会阻止批跑。`acceptance_kind=integration` 明确区分链路验收和任务质量，
`acceptance.json` 同时保存任务成功与失败原因。正常进程退出不能替代这些检查。

原生 Agent 的 termination 消息在日志方法边界被观察，不依赖 verbose 日志落盘。
summary 的 `task_failure_reason` 区分 `context_length_exceeded`、`agent_terminated`、
`evaluation_failed` 与 `runtime_error`，成功为 null；已发生的调用/token 记录保留。
不增加模型重试、输出压缩或失败任务补跑，同样的分类也应用于 native 基线。

## 固定 90 任务批跑

在线比较采用 `experiments/instances-90.txt`，对应项目自定义测试集（来自 AppWorld
官方 train/dev/test 的混合任务，具体组成见 `docs/DATA.md`）。不需要旧路由的 249/33 任务前缀重放与标签包；它们用于离线训练
与逐 Span 评价，不是在线 Agent 的输入。不要将官方全部 732 任务混为同一个测试集。

`run_batch.py` 要求 `JEV_ACCEPTANCE_FILE` 指向通过的单任务验收，且配置 SHA256
与单任务一致。额外设置 `JEV_TASK_MANIFEST` 和全新的 `JEV_RUN_OUTPUT`。逐任务
原子保存 summary，追加 `progress.jsonl`，最终生成 `report.json`，失败行保留。
服务不可用时停止并保存剩余任务清单，不在原目录自动覆盖补跑。

`four_features_accept_and_batch.sbatch` 会先等待 SLM 和 Jev 服务就绪，再验收技术链路，
随后顺序跑 90 个任务。任务失败留在原行并继续下一项，report 汇总失败类别。
脚本给配置保存独立快照；只有提交者显式设置 `CANCEL_OWN_SLM_ON_EXIT=1` 时，
退出才取消传入的实验专用 SLM allocation。共享服务不得设置该开关，不能传入 284851。

`batch_shared.sbatch` 是当前集群的操作脚本：CPU 作业运行 Agent，在自己的现有
SLM GPU allocation 内创建 Jev step。必须填写 `SLM_JOB_ID`、`PROJECT`、
`JEV_SESSION_DIR` 与上述变量；当前示例 IP 与端口固定，换节点时同步配置和脚本。
不要停止用户原有 SLM 作业；脚本退出只清理自己启动的 Jev step。

`shared_gpu_service.sh` 保存共享卡总显存和进程显存采样。Jev 启动与预热另行记录，
不属于任务内 `client_total_ms`；GPU 价格未知时仍不推断费用。上游 `output_tokens`
是答案 JSON 的分词计数，不代表自回归生成了这些 token。

本地测试：

```bash
python -m pytest tests/test_openjev_preflight.py -q -o addopts=
```

这些测试证明输入计数边界行为；真实 encoder 等价验证、GPU 推理和 AppWorld 任务
验收另行记录，不能用此测试结果代替。
