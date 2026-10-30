# 固定 Span 三种输入对照

本实验比较同一批调用前信息下的 `context`、`four_features`、`combined`。不执行 Agent，不调用 LLM/SLM；没有局部正确性和净节省标签，不能据此宣称某种提示词最准确。

## 实现

- `experiments/jev_fixed_span_data.py`：只读关联历史 SQLite 和 Jev 审计，确定性采样、重建输入、导出哈希清单。
- `experiments/jev_fixed_span.py`：原生客户端单次请求、失败记录、配对概率及阈值统计。使用环境变量入口。
- `deploy/jev/fixed_span_compare.sbatch`：单独启动 Jev，预热后运行并自动结束；需要离线验收通过。
- `tests/test_jev_fixed_span.py`：覆盖完整请求重建、关联/内容错误、抽样、篡改、失败继续、token 预检、响应校验、配对口径和验收门控。

关键修正：combined 在线构建会删除已被四特征覆盖的目标和 assistant 文本。因此不能从它直接删除 `features` 得到纯上下文。代码从完整 `model_request`、原标量和经验证的模板边界重建三种模式，使用原 `JevInputBuilder`；原 combined 必须复现一致，否则拒绝导出，不自动换样本。

每个 workflow 第一调用的空历史、空特征消息索引和完整 `[GOAL]` 文本用于确认模板消息数及目标。多行目标保留，不用空行切割猜测。模型输入只取调用前请求及特征；不混入历史记录中同一行附带的执行结果或任务评分。

## 本次样本与离线验收

源运行：`runs/jev-combined-90-20261023-122426/r1`。

数据集：`runs/jev-fixed-span-dataset-20261026`。

- 原审计 1837 条，可关联重建候选 1569 条，268 条因缺少完整 combined 输入被排除。
- seed=42，20 个任务，每个任务 5 个位置，共 100 个 Span。
- 100 个 combined 输入已逐一复现原审计；这证明输入重建一致，不是正确性标签验证。
- 只读访问源 SQLite；清单保存源配置、审计、数据库哈希，样本保存原始请求及输入指纹。
- 数据集中包含任务上下文，保留在被 git 忽略的 runs 下，不提交原始轨迹。

本地测试（Windows）：相关 46 项通过；Linux 在加入最终验收门控前的客户端/输入/路由回归 138 项通过。服务器首次离线重建验收作业 `542947` 通过；最终验收作业 `543238` 正常结束，46 项通过，100 个样本加载和完整性检查通过。`acceptance.json` 已生成并由真实运行入口再次校验。

真实实验作业 `543337` 已提交，最近查询为等待优先级调度。时限 1 小时；输出预期位于 `runs/jev-fixed-span-543337/inference`。提交不等于推理完成，最终以作业状态和 `report.json` 为准。

## 复现入口

在项目根目录执行（准备独立输出目录，拒绝覆盖）：

```bash
export JEV_FIXED_ACTION=extract
export JEV_FIXED_SOURCE="$PWD/runs/jev-combined-90-20261023-122426/r1"
export JEV_FIXED_DATASET="$PWD/runs/jev-fixed-span-dataset-new"
export JEV_FIXED_COUNT=100 JEV_FIXED_TASKS=20 JEV_FIXED_SEED=42
.venv/bin/python -m experiments.jev_fixed_span
```

真实请求入口要求 `JEV_FIXED_ACTION=run`、`JEV_FIXED_DATASET`、`JEV_FIXED_ACCEPTANCE`、`JEV_RUN_CONFIG`、`JEV_MODEL_DIR`、`JEV_RUN_OUTPUT`。验收文件必须由测试和真实样本完整性检查通过后生成，绑定数据集与实现哈希。修改实现或样本后必须重新验收。

Slurm 提交脚本读取 `PROJECT`、数据集和验收路径，自动生成 localhost 端点配置。默认 1 张 A800、4 CPU、16GB 主存，最长 1 小时，跑完即结束；不使用、不取消用户的既有 SLM 作业。启动预热单独保存在 `warmup.json`，不混入 300 次测量请求；测量沿用原配置 2 秒超时，不重试、不压缩、不替换失败样本。

## 结果口径

每个 Span 的三模式顺序按固定种子打乱，问题和候选顺序由原输入构建器统一产生。`inference/decisions.jsonl` 每次请求落盘，`inference/report.json` 同时统计：

1. 全部固定样本：各模式可用率、平均概率、候选卸载数/全部样本数、已知输入 token 与耗时。
2. 三模式均成功的交集：相同样本上的概率差及阈值比例，避免直接比较不同成功子集。

失败请求的输入消耗未知，不当成零；报告列出有用量的成功响应数。阈值结果是已有分数的探索，不推算真实任务成功率或节省。源数据来自项目既有 90 任务，这批样本不能再作为调参后的独立测试集；本次结果也不覆盖被排除的超长输入总体。
