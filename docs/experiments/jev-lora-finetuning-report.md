# OpenJev AppWorld LoRA 微调报告

**报告日期：** 2026-11-09  
**微调作业：** Slurm `592205`，`COMPLETED`，exit code `0`，用时 `00:36:05`  
**微调产物：** `/share/home/shenwsh3/ozy/jev-appworld-lora/runs/finetune/appworld-v1-context-r16-lr5e-5-seed0-20261107/`  
**范围：** 总结 OpenJev-1.7B 在 AppWorld `counterfactual-l123-v1` 数据上的 LoRA 微调与路由分类 dev 评估。本文不汇报 AppWorld 阈值路由实验。

## 摘要

本次微调以 OpenJev-1.7B 导出模型为基础，从已初始化的 seed 训练检查点继续训练。训练数据使用 V1 context-choice 格式：对每个模型调用前状态输入一条 System One `choice` 问题，目标为 `LLM` 或 `SLM`；原始标签 `L1`、`L2` 均映射至 `SLM`，标签 `L0` 映射至 `LLM`。

最终训练运行了 **2 个 epoch、318 个 optimizer step**。LoRA rank 为 **16**，batch size 为 **1**、梯度累积为 **8**，BF16；LoRA adapter 与 Pointer Head 均参与训练。与 seed checkpoint 相比，微调 checkpoint 在同一个 **810 条 dev 样本**上的 ROC-AUC 从 **0.5909 提升到 0.7540**，AP 从 **0.6555 提升到 0.8121**。固定 `P(SLM) >= 0.5` 的分类准确率由 **61.7% 提升到 68.3%**。

这些结果支持“微调改善了当前 dev 集上的路由标签区分能力”，但不能直接推出 AppWorld 整任务成功率提高，也不能视为独立 test 上的泛化结论。Dev 在两轮训练中均用于评估监控；独立 test 尚未评估。

## 数据与标签

### 标签来源及含义

标签来自 AppWorld counterfactual 标注项目，版本为 `counterfactual-l123-v1`。训练和验证集使用相同标注口径，原始标签保留在转换数据的 `_meta` 中；OpenJev 学习的是二分类路由目标：

| 原始标签 | 转换后的路由目标 | 含义 |
|---|---|---|
| `0`（L0） | `LLM` | 不卸载，选择大模型 |
| `1`（L1） | `SLM` | 卸载，选择小模型 |
| `2`（L2） | `SLM` | 卸载，选择小模型 |
| `unknown` / mask=0 | 排除 | 标签证据不足，不参与训练 |

标签统计摘要中的原始点数与转换后样本数不同：转换会排除不可用于线上路由的超长状态；之后 OpenJev loader 还可能再按模型输入限制丢弃样本。两步计数分别报告如下，不应混为一个过滤过程。

### 转换数据规模

转换器重建与线上 V1 `context` 请求一致的输入：调用前 `state.context`、实际大/小模型 ID、V1 instructions 与 `LLM`/`SLM` criteria。训练与 dev task 不重叠，`train_dev_task_overlap=0`。训练样本原始 task 中模型 ID 是 `Qwen3.8-27B` 与 `Qwen3.5-9B`。

| 划分 | 标注点 | 标签 0 / LLM | 标签 1 / SLM | 标签 2 / SLM | unknown/masked | 因线上状态长度排除 | 转换后 JSONL |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,387 | 619 | 504 | 263 | 1 | 107 | 1,279 |
| dev | 879 | 382 | 361 | 136 | 0 | 69 | 810 |

Dev 转换后的 810 条二分类目标分布为 **333 LLM / 477 SLM**。训练 JSONL 转换后 1,279 条；实际训练日志报告 OpenJev loader 使用 **1,269** 条、另有 **10** 条被 loader 丢弃。现有日志没有把这 10 条逐项归因为某一个具体原因，因此本报告不进一步推断。Dev loader 接受全部 **810** 条，未丢弃。

超长排除的 converter 规则是尽可能移除最旧历史 span，直到 state 满足线上字符约束；如果只剩必要状态仍超过 `max_state_chars=12000`，该标注点不进入转换数据。训练阶段另有 OpenJev 自身的数据编码限制（`max_state=4096`、`max_branch=2048`、`max_tokens=6144`）。

### 数据隔离与防泄漏

- 训练和 dev 按 task 划分，转换摘要确认 task 交集为 0。
- 每个点只使用第 `k` 次模型调用前可见的状态/上下文，任务结果、调用后的工具输出及标签解释不拼入模型输入。
- Prompt 版本固定为 V1；训练 JSONL 的 V1 instructions/criteria SHA-256 为 `a520f9c99e990a6ce282dbcf29d5347ff085685953c86e394dfc1f3f21362b2a`。
- 训练和 dev 的 task 级数据来自两套标注数据；dev 标签用于训练过程中的每轮监控和最终 dev 评估，因此不是未触碰的 test 集。

## 模型与训练设置

### 初始化与可训练部分

- 基础模型：`/share/home/shenwsh3/ozy/models/OpenJev-1.7B`，BF16，28 层。
- 初始化检查点：`runs/checkpoints/openjev-1.7b-export-seed-run`，即微调前 seed checkpoint。
- LoRA adapter 目标模块：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。这覆盖 attention 投影和 FFN 投影；不是只加在 q/k/v 上。
- LoRA `rank r=16`：每个目标线性层的低秩更新由 rank-16 的两个较小矩阵表达。`alpha=32` 控制 adapter 更新缩放；`dropout=0.05` 是 LoRA 分支的 dropout。
- Pointer Head：类型 `pointer`、`head_dim=256`。训练配置未冻结 head，训练日志报告总 trainable 参数约 **18.5M**，故本次训练同时更新 LoRA adapter 与 Pointer Head；基础 Transformer 权重保持冻结，仅通过 LoRA 增量适配。

### 超参数

| 参数 | 设定 | 说明 |
|---|---:|---|
| epoch | 2 | 完整遍历 1,269 条 loader 接受的训练记录两次 |
| LoRA rank | 16 | Adapter 低秩维度 |
| LoRA alpha | 32 | Adapter 更新缩放参数 |
| LoRA dropout | 0.05 | Adapter 分支 dropout |
| LoRA 学习率 | `5e-5` | LoRA 参数学习率 |
| Pointer Head 学习率 | `5e-5` | Pointer Head 参数学习率 |
| weight decay | `0.01` | AdamW 类优化器的权重衰减设置 |
| micro-batch | 1 | 单个前向/反向小批次大小 |
| gradient accumulation | 8 | 累积 8 个 micro-batch 后更新；通常相当于约 8 条记录的有效 batch，epoch 尾部除外 |
| optimizer steps | 318 | 每 epoch 159 step，共 2 epoch |
| dtype | BF16 | 微调前向与训练精度 |
| seed | 0 | 随机种子 |
| max_state / max_branch | 4096 / 2048 tokens | OpenJev 编码器允许的 state / question branch 上限 |
| max_tokens | 6144 | 整条训练样本 token 上限 |
| eval_n | 810 | 每轮评估全部 dev 条目 |

每轮记录的 loss 是训练进程按 batch 输出的在线 loss；以下 dev accuracy/ECE 是独立的全量 dev 评估统计，二者不要互相替代。

## 训练过程与每轮评估

训练作业 `592205` 于 **2026-11-07 17:47:41 CST** 在 `5500-node02` 启动，Slurm 用时 **36 分 05 秒**，正常完成。训练日志报告完整训练循环约 23.2 分钟，峰值显存约 **4.9 GB**；作业总时长还包含训练循环之外的步骤，现有产物未对这段时间进一步拆分。每轮遍历 1,269 条记录，dev 评估 810 条。

| 训练轮次（日志从 0 编号） | 累计 step | Dev accuracy | Dev ECE | Dev mean confidence | Dev latency median / p90 |
|---:|---:|---:|---:|---:|---:|
| 1（epoch 0） | 159 | 68.15% | 0.0686 | 0.7501 | 107.5 / 145.9 ms |
| 2（epoch 1） | 318 | 68.27% | 0.0791 | 0.7618 | 107.4 / 146.0 ms |

第二轮 accuracy 比第一轮高 **0.12 个百分点**；ECE 从 0.0686 升至 0.0791，校准误差略变大。两轮 dev accuracy 接近，不能据此声称持续大幅提升。最终保存的是完成第二轮后的 checkpoint。

训练前另做了 32 条样本、1 epoch 的 smoke training，Slurm 作业 `591327` 正常完成，检查 adapter/head 保存和载入链路。Smoke 的 dev n=32、accuracy 37.5%、ECE 0.4823，样本很小且只是运行链路测试；它不是最终模型性能估计，也不与全量 dev 指标混算。

## 微调前后分类指标

使用同一个 `dev.jsonl`，同一组 810 条样本、同一标签分布（333 LLM / 477 SLM）、同一正类 `SLM`，分别对 seed checkpoint 与最终 LoRA checkpoint 运行概率评估。两次 loader 均接受 810 条、丢弃 0 条。

| 指标 | Seed checkpoint | LoRA 最终 checkpoint | 变化 |
|---|---:|---:|---:|
| ROC-AUC | 0.5909 | **0.7540** | +0.1631（+16.31 个百分点） |
| Average Precision（AP） | 0.6555 | **0.8121** | +0.1565（+15.65 个百分点） |
| Accuracy，固定 `P(SLM) >= 0.5` | 61.73%（500/810） | **68.27%（553/810）** | +6.54 个百分点 |

固定 0.5 仅在这里作为前后分类器的同口径诊断点，并非阈值搜索或部署阈值推荐。

### 固定 0.5 的混淆矩阵

正类为 SLM，规则为 `P(SLM) >= 0.5` 预测 SLM，否则预测 LLM。

| 模型 | TP：正确判 SLM | FP：LLM 误判 SLM | FN：SLM 误判 LLM | TN：正确判 LLM |
|---|---:|---:|---:|---:|
| Seed checkpoint | 441 | 274 | 36 | 59 |
| LoRA 最终 checkpoint | 338 | 118 | 139 | 215 |

微调后错误卸载（FP）减少 156 条，正确识别为 LLM（TN）增加 156 条；同时漏掉应卸载的 SLM 样本（FN）增加 103 条，TP 减少 103 条。这与 AUC/AP 改善并不矛盾：AUC/AP 衡量整体排序/区分能力，而固定 0.5 的混淆矩阵还受到概率校准与分类边界位置影响。对实际路由而言，错误卸载和漏卸载成本也不对称，不能只凭 accuracy 判定最终策略。

## 结论与限制

1. **路由分类微调在该 dev 集上有明确改善。** AUC 从 0.5909 到 0.7540，AP 从 0.6555 到 0.8121，固定 0.5 accuracy 从 61.73% 到 68.27%。
2. **收益主要表现为更会排序/区分候选，而非固定 0.5 下召回更高。** 微调后 precision 为 74.12%（338/456），recall 为 70.86%（338/477）；seed 在 0.5 下 precision 为 61.68%（441/715）、recall 为 92.45%（441/477）。因此把误卸载压低的同时，也增加了漏卸载。
3. **校准尚非全面改善。** 第二轮内置评估 ECE 从 0.0686 增至 0.0791；需要单独评估概率校准，不能把 AUC/AP 上升理解成概率值本身已完全可靠。
4. **dev 不是独立 test。** Dev 在训练两个 epoch 的过程中每轮被评估，并用于最终前后比较；这些数字支持开发决策，不是未见数据上的最终泛化指标。
5. **尚未由这份报告判断端到端 AppWorld 收益。** 分类器标签预测质量与 Agent 完整任务是否通过评测不是同一指标。本文不纳入任何阈值路由任务成功率、卸载率或不同 task manifest 的比较。

## 产物与复现定位

- 数据转换目录：`/share/home/shenwsh3/ozy/jev-appworld-lora/data/openjev-v1-context/`
- 转换摘要：`data/openjev-v1-context/conversion_summary.json`
- 训练配置：`runs/finetune/appworld-v1-context-r16-lr5e-5-seed0-20261107/training_config.json`
- 每轮训练评估：`runs/finetune/appworld-v1-context-r16-lr5e-5-seed0-20261107/metrics.json`
- Adapter 配置与权重：同一训练目录下 `adapter_config.json`、`adapter_model.safetensors`
- Pointer Head 权重：同一训练目录下 `head.pt`
- Seed dev 概率评估：`runs/evaluations/seed-dev-full-20261107/`
- LoRA dev 概率评估：`runs/evaluations/finetuned-dev-full-20261107/`
- LoRA 训练运行：OpenJev 源码 commit `da6a68fb9195958cc0a6cdb7b35587acec436a65`；Span-Offload 比较器仓库当时 HEAD `8e4fe5b`。

分类指标脚本为 `scripts/jev_lora/evaluate_route_metrics.py`。报告中的 ROC-AUC/AP 按 `P(SLM)` 对正类 SLM 计算；预测文件保留逐样本 `p_slm`、`p_llm`、task ID、卸载点 k 和原始 L0/L1/L2 标签，便于后续复核。
