# Jev LoRA AppWorld 输入数据抽样审计

**审计日期：** 2026-11-04  
**范围：** 只读核验 AppWorld Span-Offload 轨迹与 OpenJev 训练输入的对应关系；未转换数据、未启动训练。

## 结论摘要

- 原始轨迹记录了每个模型步骤调用前的 `input_context.messages`，可作为卸载决策样本的主要输入候选。
- 抽查任务中，trace 的 `steps[k-1].step_id` 与卸载点 `k` 一致；k=1、2 的 `prefix_trace.next_input_context` 与对应 trace 步骤的输入消息哈希及消息数完全一致。
- 抽查到的 `native_payload.tools` 为空。AppWorld 交互工具调用记录在步骤输出的 `tool_logs` 中，不应误作调用前的工具定义或训练输入。
- 快照包保留 `prefix_trace.json`、`state_ref.json` 和 `first_token_probe.json`，但没有快照 `state/` 目录。因此目前可验证的是模型可见对话输入，不能据此恢复完整 AppWorld 内部状态。
- README 所列标注脚本未在已检查的位置找到；训练标签尚未生成，批量转换前必须找回并核对该脚本及其版本。

## 抽样方法与证据

抽查了训练集中的两个完整任务：`229360a_1`（15 步）和 `07b42fd_1`（13 步）。对 k=1、2、6、13 检查步骤编号、调用前消息数、`native_payload.tools` 数量、`tool_logs` 数量及环境快照引用。

以 `229360a_1` 为例：

| 卸载点 k | `step_id` | 调用前消息数 | `native_payload.tools` 数 | `tool_logs` 数 | 快照引用 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 23 | 0 | 1 | 存在，指向 k0001 |
| 2 | 2 | 25 | 0 | 1 | 存在，指向 k0002 |
| 6 | 6 | 33 | 0 | 1 | 存在，指向 k0006 |
| 13 | 13 | 47 | 0 | 1 | 存在，指向 k0013 |

`07b42fd_1` 的 trace 同样按 k 顺序递增；抽查的 k=1、2、6、13 均有对应 `step_id`，消息数分别为 23、25、33、47，`native_payload.tools` 数量均为 0。

进一步比较了 `229360a_1` 的 k=1、2：

- k=1 的 `trace.steps[0].input_context` 与 `snapshots/k0001/prefix_trace.json` 中的 `next_input_context`：消息数均为 23，`input_messages_sha256` 均为 `b36a9820202286a6a5198f7e0ab80716fb40b788d75d6426128e1de406d9c29f`。
- k=2 的对应输入：消息数均为 25，`input_messages_sha256` 均为 `e4df00f9f22c21460aa3d3ab7f1897c848d4bee1a7d27a576fd17789f7b97bcf`。

这支持“第 k 步卸载点对应第 k 个模型调用前上下文”的映射，但目前只对两个任务做了结构抽查、对一个任务的 k=1、2 做了哈希逐项确认；批处理前仍应对全部任务自动化验证。

## 输入字段判断

### 可用的主要输入

`original/trace.json` 中第 k 步的 `steps[k-1].input_context.messages` 是调用前上下文候选。样本中消息由 `role` 和 `content` 组成，内容包含模型当时可见的对话历史。训练时应保留原始顺序，并确保消息截断/序列化方式与部署路由器实际收到的输入一致。

### 工具信息

抽查的 `input_context.native_payload.tools` 数量为 0。步骤含有 `tool_logs`，但那是该模型响应对应的 AppWorld 执行记录，属于决策之后的信息，不能输入卸载分类器。需在完整数据审计中确认所有步骤是否都遵循这一结构；若线上路由器还会看到轨迹中未记录的工具配置，也必须明确其训练/部署对齐方式。

### 快照与环境状态

抽查的快照目录含 `prefix_trace.json`、`state_ref.json`、`first_token_probe.json`，没有 `state/` 目录。`state_ref.json` 保存快照 ID、状态哈希和原状态目录引用；引用不等于当前数据包内可读取的状态内容。`prefix_trace.json` 可验证前缀与下一次输入上下文，但不能替代完整环境状态。因此第一版输入不应虚构或推断未保存的环境状态。

## 标注脚本来源核查

训练集 README 声明：

- 标注脚本：`classfierTrain/02_数据标注/label_spans_appworld_0920.py`
- 标签口径：`counterfactual-l123-v1`
- README 示例路径：`/home/sws/Desktop/TraceOffload`

在服务器检查的 `~/ozy`、`~/qwen36_vllm` 相关目录及当前本地仓库中，未找到该脚本或对应 `02_数据标注` 目录。README 中的绝对路径指向另一环境，不能证明服务器上存在该源码。当前 LoRA 项目的 `labels/` 目录没有标注文件。

## 建议的训练目标映射

为与“卸载给小模型 / 保留给大模型”的实验决策一致，主目标建议使用一个 System One `noul` 问题：

- L0 -> `false`（不卸载）
- L1、L2 -> `true`（卸载）

原始 L0/L1/L2 标签应作为行级元数据保留，供分析安全性与净收益使用，不应混入输入文本。样本输入只截止在第 k 步模型生成输出之前；不得包含该步输出、之后的消息、T/B 分支结果、成功率统计或标签计算结果。

## 后续门槛

1. 找回标注脚本及其依赖/版本，核对其准入规则与 README 的 `counterfactual-l123-v1` 说明一致。
2. 对完整 train/dev 数据验证 `step_id`、k、快照前缀输入哈希、`messages` 结构和缺失值；记录排除原因及标签分布。
3. 定义与线上路由器一致的 messages 序列化方式，并抽样人工检查决策边界处没有未来信息泄漏。
4. 只有以上检查通过后，才生成独立的 OpenJev `{request, labels}` 数据文件；原始 trace 保持只读，test 集不用于训练或阈值选择。
