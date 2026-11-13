# 后续 test 指标的原始数据充分性审计

审计日期：2026-11-13。指标审计以本地代码和已有报告为依据；第 5、6 节另记录服务器端 test 数据预检和用户授权启动的标签作业。不改变正在运行的 dev 阈值实验。

本次目标是确认哪些原始字段需要保留，能离线派生的指标不增加运行时记录。没有保留新建的派生汇总模块，也没有修改现有运行器。后续新增的 CPU test 标签作业入口及提交情况见第 6 节；它不执行端到端 test。

## 1. 审计结论

正常完成的端到端任务已经记录了大部分所需原始数据。主表指标应从逐任务 `results.json`、配置和 manifest 离线计算，不能仅凭组级 `report.json` 还原 SD、pass^5 或时延分布。

| 指标 | 所需原始数据 | 当前记录与结论 |
|---|---|---|
| success | 每个 task-run 的 `task_success`、正式任务清单 | 已记录。成功数/预定执行数；失败仍在分母中 |
| 逐轮均值 ± sd | task_id、每轮身份、每次 task_success、每轮完整清单 | 单次运行的 run_id 与 manifest 已记录，可显式建立 round_id→run_id 映射后离线算；不能把 run_id 当成轮次顺序 |
| pass^5 | 同一 task 的五轮 task_success | 记录结构足够，实际需要五轮数据；三轮不能报告 pass^5 |
| 总卸载率 | 每任务大小模型调用数 | `framework_model_calls_by_target` 已记录；跨任务跨轮合并计数 |
| 有效卸载率 | 同上，另加 SLM fallback 次数 | 正常完成任务的 `mechanism.fallback_by_reason.slm_fallback` 已记录；可离线算 |
| cost/run | 大小模型输入/输出 token、冻结的模型单价、执行数 | `framework_tokens_by_target_io` 已记录；提示词/LoRA 比较行另有 `estimated_execution_cost_usd` 和 `price_table_version` |
| cost/success | 同上，加成功执行数 | 原始字段已够；失败执行成本也计入分子 |
| 成本下降 vs LLM | 路由组和实测纯 LLM 组的同口径费用 | 需要同任务、同轮数的纯 LLM 基线；仅有路由结果不能计算实测下降 |
| 节省比例 | 同上 | 不额外记录，离线计算成本比值 |
| p95 | 每个 task-run 的端到端耗时 | 正常行有 `e2e_wall_clock_ms`；比较器所有正常/超时行还有 `worker_wall_clock_ms`，可统一用后者覆盖 worker 初始化与执行时间 |
| AUC、AP | 每条有标签的路由状态的真实类别、连续 P(SLM) | 离线评估脚本保存 `gold_route`、`p_slm`、`p_llm`；已有 dev 产物具备计算条件。后续 test 标签和产物需单独准备/核验 |

这里的“已记录”指源码路径确实写出这些字段，不代表本次已经逐个检查服务器上的每份数据文件。实际汇总仍须检查缺失、重复、遥测完整性与配置一致性。

## 2. 已确认的指标口径

`run` 表示一个 task 的一次完整执行，`round` 表示同一配置遍历全部任务的一轮。设每轮有 N 个任务，共 R 轮。

- **success**：成功 task-run 数 / (N×R)。对完整、等任务数的轮次，它等于各轮 success 的均值。
- **均值 ± sd**：先算每轮 success，再取均值和样本标准差，SD 分母是 R−1。SD 不是 95% CI；只有一轮时不能估计轮间 SD。
- **pass^5**：用户已确认，五轮全部成功的 task 数 / N；不是五次至少成功一次，也不是总体 success 的五次方。
- **总卸载率**：ΣSLM / [ΣSLM+ΣLLM]。
- **有效卸载率**：(ΣSLM−Σslm_fallback) / [ΣSLM+ΣLLM]。
- **cost/run**：全部 task-run 成本 / (N×R)。
- **cost/success**：全部 task-run 成本 / 成功 task-run 数；不能仅对成功任务的费用取平均。成功数为零时留空。
- **成本下降 vs LLM**：LLM cost/run − 路由 cost/run，单位 USD/run。
- **节省比例**：1 − 路由组总成本 / 同口径 LLM 组总成本。负值表示更贵；无基线或基线为零时留空。
- **p95**：把全部 task-run 耗时合并后取第 95 百分位数，包含失败和超时；采用固定的线性插值方法，单位为秒。不要对每轮 p95 再求平均。

总卸载率是实际派给 SLM 的调用占比，不是 Jev 的 SLM 意图概率。两种卸载率的分母都是已经发生的大小模型调用总数，包含 fallback 后的 LLM 调用，不因扣 fallback 而减少分母。

例如 SLM 40 次、LLM 60 次、SLM fallback 10 次，则总卸载率 40%，有效卸载率 30%。这些比例由调用计数加总计算，不是各任务比例的简单平均。

有效卸载率严格沿用用户给定公式。它表示扣除了已记录 fallback 的 SLM 调用，不能单凭这个数保证最终 task 成功。当前 fallback 计数在 `Coordinator._start_fallback_episode` 启动 LLM fallback span 时增加；若某次 SLM 未被接受，但因为恢复状态等原因没有启动 fallback，SLM−fallback 不一定精确等于 accepted SLM 次数。更严格的接受记录已在 span 的 `execution_summary.accepted_model_attempt_id` 与 `model_attempts` 中，可保留作审计，不替换用户公式。

## 3. 必须保留的产物与当前边界

建议完整收集已有产物，而不是增加主表派生字段：

1. `manifest.json`：任务 ID、split、group、run_id、config/source/prompt 身份；额外明确 round_id→run_id 对照。
2. `results.json`：正式 `stage="batch"` 的逐任务结果。smoke 单独保存，不进入正式分母。
3. 每组实际 YAML、source snapshot、checkpoint 身份及价格表版本。
4. 每任务路由 JSONL、history SQLite 和 evaluator 报告，供调用、接受和失败原因复核。
5. 离线分类器评估的 `predictions.jsonl`、`metrics.json` 以及标签版本和原始样本身份。

当前存在以下边界，应在后续 test 入口准备时处理，不通过新增派生指标掩盖：

- **轮次映射未内嵌到每个任务行**：现有 run_id 可区分运行，但 round 1/2/3/4/5 必须在实验清单中明确对应；这是可复现信息，不需要在推理热路径中重复计算指标。
- **超时消费无法保证完整**：比较器会从 `partial-execution.json` 恢复已完成调用统计，并标记 `consumption_complete=false`。在途请求的 token、尚未写出的机制计数可能缺失，不能事后用零补齐。成功率可按超时失败计算；完整成本和两种完整卸载率应标为未知/下界。
- **异常路径的 fallback 可能缺失**：正常完成分支写出机制增量；异常/中断恢复不能保证该字段存在。整个 map 缺失表示未知；已有空 `fallback_by_reason={}` 才表示已知零。
- **调用计数边界**：`_GatewayMeter.complete` 在 gateway 返回后追加统计，不能据此宣称覆盖所有抛异常 HTTP 请求或服务端在途计费。无此类异常的正常任务可按现有统计口径计算。
- **价格口径**：当前成本是执行模型 token 估算，包含 SLM 和 fallback LLM，不含 Jev/GPU/基础设施。纯 LLM 原始行未必有同名估算字段，但其输入/输出 token 可以使用相同冻结价格表离线补算；不能把未知 cost 当零。
- **超时时延**：worker 耗时是被中断时已消耗的时间，不是任务自然完成时间。若采用 worker 口径，要对全部任务统一采用它，同时报告 timeout 数。
- **test 入口**：LoRA 阈值 profile 固定检查 56 个官方 dev task；转换器目前只写 train/dev；分类器评估函数固定读取 dev。后续 test 需要独立 split/manifest/标签路径支持，不能重命名 dev 文件冒充 test。

因此，目前不需要为这些主表指标新增运行时派生字段；应先明确 test 清单、五轮映射、LLM 基线和带标签的离线 test 状态集合，再按上述缺失/异常规则验收原始数据。

## 4. 选定阈值后如何计算 AUC 与 AP

离线分类器的评估单位是一条**模型调用之前的状态**，不是整个 AppWorld task。正类约定为 SLM：

```text
y = 1：gold_route == "SLM"
y = 0：gold_route == "LLM"
score = p_slm
```

转换器当前映射为原始标签 0→LLM、1/2→SLM；unknown/mask=0 不进入有监督评估。计算结果必须说明这是相对于哪个标签规则、哪个执行模型对的分类表现。

**ROC-AUC** 衡量排序能力：随机取一条真实 SLM 样本和一条真实 LLM 样本，SLM 样本分数更高的比例，平分计 0.5。等价于对全部概率阈值形成 ROC 曲线并计算面积。它使用连续分数，不能只用选定阈值后的 0/1 决策代替。

**AP（Average Precision）** 先按 score 从高到低排序；扫过所有不同分数时计算 precision、recall，按 `Σ (recall_j−recall_{j−1}) × precision_j` 求和。同分数一起处理。这里采用非插值 AP，不把 PR 曲线梯形积分与 AP 混为一谈。

对于固定 checkpoint、同一组带标签状态和同一套概率，选部署阈值 0.55、0.65 或 0.80 都不改变 AUC/AP。阈值控制另外的 TP、FP、FN、TN，以及 precision=TP/(TP+FP)、recall=TP/(TP+FN)。阈值变化后在线 Agent 会进入不同轨迹，若拿不同轨迹状态计算指标，样本分布本身已改变，不能当作相同离线 test 的比较。

最小示例：

```text
真实标签 y： [0,   0,   1,    1]
P(SLM)：     [0.1, 0.4, 0.35, 0.8]

ROC-AUC = 0.75
AP = 0.833333...
阈值 0.5：TP=1，FP=0，FN=1，TN=2；precision=1，recall=0.5。
```

现有 `scripts/jev_lora/evaluate_route_metrics.py` 已实现 `roc_auc`、`average_precision` 和 `threshold_metrics`，并把逐条 `gold_route`、`p_slm`、`p_llm` 写入 `predictions.jsonl`，可连同保留的 `_meta` 信息定位 task_id/k。已有数值测试覆盖上述示例、同分数和缺少正/负类的边界。ROC-AUC 在只有一类时未定义；没有正类时 AP 未定义；脚本用 null 表示。

线上 `routing-decisions-*.jsonl` 保存了每次 Jev 响应中的连续概率，但没有自动拥有每个实际新状态的独立 gold label。仅凭在线成功/失败、是否触发 fallback 或接受模型身份，不能生成等价的真实路由标签：未执行 SLM 的步骤缺少 SLM 反事实结果，guard 通过也不等于标签规则的成功。因此，在线成功率/卸载率与离线 AUC/AP 应分开报告。

后续在 dev 选好 checkpoint/阈值后，可在独立带标签 test 状态集合上评一次连续概率，离线得到 AUC/AP 和固定阈值 P/R；再用端到端 test 五轮执行得到 success、pass^5、卸载、成本和时延。这两类评估各自回答分类排序质量与完整任务行为，不能互相替代。

## 5. Test 标签生成预检与可执行方式

2026-11-13 对服务器数据及脚本做了只读预检和两点内存复算，没有提交全量标注作业。

原始数据目录为 `/share/home/shenwsh3/qwen36_vllm/AppWorldData_1102/测试集/`，包含 243 个任务，其中 test_normal 102 个、test_challenge 141 个；manifest 标为 complete 的 231 个、partial 的 12 个。243 份 original trace 均在，但 README 提到的 tasks.txt 实际不存在，应使用 MANIFEST.json/TASKS.csv。

原标注脚本 `discover_points()` 实际发现 4,757 个候选 `(task_id,k)`，来自 233 个有分支目录的任务；另外 10 个任务没有可发现的分支点。这个数不等于通过守卫的有效标签数：不完整、前缀不一致等点应由原脚本标为 unknown/mask=0，不能提前当成 LLM 负类。任务覆盖范围和未标注任务仍须单独报告。

与既有 train/dev 标签的 task ID 比较，test 任务交集均为零。本检查是 task 层面的重叠检查，不等于做过全部 trace 指纹或历史调参使用审计。

三个脚本在 LoRA 副本与原项目中哈希相同：

| 文件 | SHA-256 |
|---|---|
| label_spans_appworld_0920.py | `bb03f3c6222930df51ec435bd8284ceec5a2eba17ad77e98208d7d2455369686` |
| label_spans_appworld_l123.py | `8a68f43ecc81c2e52fb89f90be12489e68037d538dc27c244f95b837dcd17222` |
| aw3_build.py | `b55691a19ddfd8e7769e6e89605897c7c4e90b31d5fabb681eb9c0490ca8785c` |

应复用原项目 `classfierTrain/02_数据标注/label_spans_appworld_0920.py`，使用 conda 环境 `/share/home/shenwsh3/.conda/envs/span-offload/bin/python`。LoRA 的 Python 环境不必承担该旧项目的依赖。

启动时还需显式提供 AppWorld 源码与 API 文档：原项目没有 `appworld/data/api_docs/openapi`，默认 `API_METHODS` 会为空，静默退回动词启发式。当前可用文档位于 `span-offload-jev-prompts/appworld/data/api_docs/openapi`，预检加载出 473 条 API 方法定义。

在此启动方式下，复算了 train 的 `07b42fd_1,k=7` 与 dev 的 `0d8a4ee_1,k=6`；两者的 gate、L1、L2、L3、label、mask、reasons 都与现有标签逐项一致，单点读数据与标注分别耗时约 4.92 秒、4.14 秒。这是抽样复现检查，尚未重新核验所有旧标签。

正式标注参数保持原口径：`counterfactual-l123-v1`、bootstrap=2000、expected_reps=10、seed=20260920；使用逐点独立随机流，不启用 legacy-global-rng。计划输出目录为：

```text
/share/home/shenwsh3/ozy/jev-appworld-lora/labels/test/appworld_labels_0920/
    labels_appworld_0920.jsonl
    summary_appworld_0920.json
```

下列命令应在 CPU allocation 中运行，不能当作登录节点上的全量预检。它只读取已采集的 trace/分支结果，不请求 Jev、LLM、SLM，也不执行 AppWorld 工具；未实际执行此全量命令：

```bash
/share/home/shenwsh3/.conda/envs/span-offload/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

base = Path('/share/home/shenwsh3/ozy')
appworld = base / 'span-offload-jev-prompts' / 'appworld'
sys.path.insert(0, str(appworld / 'src'))
script = (base / 'trace-offload-orginal' / 'classfierTrain'
          / '02_数据标注' / 'label_spans_appworld_0920.py')
spec = importlib.util.spec_from_file_location('test_labeler', script)
labeler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labeler)
labeler.aw3.API_DOCS = appworld / 'data' / 'api_docs' / 'openapi'
labeler.API_METHODS = labeler.aw3.load_api_methods()
if not labeler.API_METHODS:
    raise RuntimeError('AppWorld API method definitions are unavailable')
out = base / 'jev-appworld-lora' / 'labels' / 'test' / 'appworld_labels_0920'
if out.exists():
    raise FileExistsError(f'Refusing to overwrite label output: {out}')
raise SystemExit(labeler.main([
    '--data', '/share/home/shenwsh3/qwen36_vllm/AppWorldData_1102/测试集',
    '--out', str(out),
    '--bootstrap', '2000', '--expected-reps', '10', '--seed', '20260920',
]))
PY
```

旧 train/dev 标注分别处理 1,387/879 点，执行步骤耗时约 90/48 分钟。按候选数量粗估 test 约需 4–6 小时，I/O 与轨迹长度可使估计偏移，建议初始申请 8 小时 CPU 资源（如 2 核、8 GB、无 GPU）。现有脚本逐点串行，申请更多 CPU 不会自动加速。

标注后先验收唯一 task/k、label/mask、unknown 原因和完整性，再转换独立 test.jsonl；记录实际有效点数、覆盖任务数及超长输入排除数。没有分支的任务不会凭空产生标签，也不能用 original 成败补作标签。

## 6. 已提交的 test 标签作业

用户于 2026-11-13 授权启动。作业入口为本地 `deploy/jev/label_appworld_test.sbatch`，服务器副本为 `/share/home/shenwsh3/ozy/jev-appworld-lora/scripts/jev_lora/label_appworld_test-20261113.sbatch`。入口已通过 Bash/Python 语法检查及服务器预检。

- Slurm 作业号：`667297`，名称 `jev-label-test`。
- 分区：`x86_64`，account `zhangyc29`、QoS `32gpu`；2 CPU、8 小时、无 GPU。
- 已核验启动：2026-11-13 16:09:11，节点 `2488-node01`；时限至 2026-11-14 00:09:11（集群时间）。
- 原先 `--mem=8G` 的提交被 Slurm 拒绝，未创建作业；CPU 节点的 Slurm `RealMemory` 配置均为 1 MB，与实际物理内存不符。正式入口沿用旧 train/dev 作业的无显式内存申请方式，未预留 8 GB。
- 参数：bootstrap=2000、expected_reps=10、seed=20260920、逐点独立随机流。
- 输入 manifest SHA-256：`438c4a8a7da70f3ab491c43280d015645db13da962b0a5cdaecf4ee3e30294ed`。
- 标注输出：`/share/home/shenwsh3/ozy/jev-appworld-lora/labels/test/appworld_labels_0920/`。
- 日志：`/share/home/shenwsh3/ozy/jev-appworld-lora/runs/label-test-667297.log`。
- 收据及实际提交脚本：`/share/home/shenwsh3/ozy/jev-appworld-lora/runs/label-test-667297/`，其中 `label-generation-receipt.json` 保存脚本、辅助库、API 文档、输入清单和 train/dev 标签哈希，以及参数和覆盖范围。

作业内预检返回 `LABEL_PREFLIGHT_OK`，candidate_points=4757、tasks_with_points=233、api_method_count=473、train_dev_task_overlap=0。最后一次启动检查确认处于 RUNNING，尚未宣称标签完成。

入口拒绝覆盖已有标签输出。标注结束后会自动校验输出与 4,757 个候选 `(task_id,k)` 一一对应、summary 行数和任务数一致、label/mask 合法，并写输出哈希。标签脚本将所有点处理完后才写最终 JSONL；运行期间以日志中的 `[已处理/4757] labeled` 查看进度。
