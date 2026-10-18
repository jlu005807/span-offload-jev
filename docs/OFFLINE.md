# 离线路由器：116 维特征、训练、选阈值

## 特征

```
116 维 = [prefix_n_A, log_est_prompt_tokens]        2 列，折内标准化
       + PCA50(V2_goal_recent 嵌入 1024 维)          50 列
       + PCA64(C_bare_w10 嵌入 1024 维)              64 列
```

| 段 | 是什么 | 从哪来 |
|---|---|---|
| ① `prefix_n_A` | 前缀里分类为「写」的工具执行数 | `router/build.py`，用 OpenAPI 的 HTTP method 判 |
| ② `log_est_prompt_tokens` | `log1p(前缀消息总字符数/4)` | 同上 |
| ③ `V2_goal_recent` | `[GOAL] 任务指令` + 最近 10 个 span 的 `[TOOLS]/[REASONING]/[ASSISTANT]` | `router/features.py` |
| ④ `C_bare_w10` | 最近 10 个动作的规范化签名（函数名 + 参数），不带 category | `router/actions.py` |

③④ 各用 Ollama `qwen3-embedding:0.6b` 编成 1024 维（不归一化、num_ctx=8192）。
**空的动作文本（k=1）给零向量，不调用模型** —— 编码空串会得到一个非零向量，
静默污染第 4 段。

## 训练

两个 logit，同一个 116 维输入：

| 模型 | 标签 | 回答 |
|---|---|---|
| gate | `l2_new` | 这一步本地做对不对 |
| net | `net_saving > 0` | 这一步省不省钱 |

为什么是两个：单模型 AUC 最高 0.8236 但 oracle 捕获只有 11.3%；两阶段 AUC 0.7634
却有 16.0%。AUC 只衡量单分数排序，看不见两个阈值的自由度。

纪律：标准化与两个 PCA **只在训练集上拟合**，验证集只 transform。

## 阈值

```
卸载 ⟺ p_gate ≥ τ_gate ∧ p_net ≥ τ_net
(τ_gate, τ_net) = argmax Σ_{被卸载的点} net_saving
```

- **不是 F1**：F1 最优阈值下卸载率 70%+，oracle 捕获 −101.9%（倒亏）
- 二维搜索：τ_gate 取 60 个分位点，每个点上对 p_net 做一维前缀扫描（60 次 O(n)）
- 家族级 bootstrap 400 轮取中位数作部署值（按家族重采样，不按点）

## 产物

`router/artifact/l2_add4_twostage_logit.json`（1.8 MB）：两个 PCA 的均值/系数、
两组标准化统计、两个 logit 的系数与截距、两个阈值、验证集指标。

在线侧 `traceoffload/m2/predictor.py` 读同一份文件，用同一套公式重建 116 维
（连 float32 还原都对齐），因此**离线与在线逐位一致**。

## 已验证的复现

在 4950 点训练 / 588 点验证上重跑：

| 项 | 参考 artifact | 重跑 |
|---|---|---|
| gate 家族均衡 AUC | 0.7694 | 0.7694 |
| net 家族均衡 AUC | 0.7087 | 0.7087 |
| τ_gate | 0.538072475100225 | 0.538072475100225 |
| τ_net | 0.5523107182337628 | 0.5523107182337628 |
| PCA / 标准化 / 系数 | — | max\|Δ\| = 0.0 |
| 验证集卸载 | 119/588（捕获 16.05%） | 同 |

---

## 从零跑到 artifact（命令）

输入就是 `docs/DATA.md` §2 列的那几份文件。以训练集 249 task / 4950 cell、验证集 33 / 588 为例：

```bash
# 1) 特征：两个标量 + 两段文本（只读 runs/，不碰 branches/）
python3 -m router.build --runs runs/训练集 --out data/训练集 \
    --openapi-dir appworld/data/api_docs/openapi
python3 -m router.build --runs runs/验证集 --out data/验证集 \
    --openapi-dir appworld/data/api_docs/openapi

# 2) 嵌入（Ollama qwen3-embedding:0.6b，1024 维，不归一化、num_ctx=8192）
python3 -m router.embed texts --texts data/训练集/texts.jsonl \
    --out data/训练集/aw3_emb_v2goal.jsonl
python3 -m router.embed actions --texts data/训练集/actions_texts.jsonl \
    --vectors data/训练集/emb_C_bare_w10.npz --ids data/训练集/cell_ids.npz

# 3) 训练两个 logit（折内标准化 + PCA50/PCA64）
python3 -m router.train \
    --train-dir data/训练集 --val-dir data/验证集 \
    --train-labels <labels_appworld_0920.jsonl> <labels_待采集训练实例.jsonl> \
    --val-labels <labels_验证集.jsonl> \
    --net <net_saving_labels.jsonl> \
    --actions data/训练集/emb_C_bare_w10.npz --action-ids data/训练集/cell_ids.npz \
    --out out/model.json --val-scores out/val_scores.jsonl

# 4) 验证集挑阈值（最大化 Σ净节省，家族级 bootstrap）
python3 -m router.select_threshold --scores out/val_scores.jsonl --out out/tau.json

# 5) 导出 artifact（τ 在落盘参数自己的分数上重选）
python3 -m router.export --model out/model.json --val-dir data/验证集 \
    --val-labels <labels_验证集.jsonl> --net <net_saving_labels.jsonl> \
    --actions data/训练集/emb_C_bare_w10.npz --action-ids data/训练集/cell_ids.npz \
    --out router/artifact/l2_add4_twostage_logit.json
```

期望结果（与仓库自带 artifact 逐位一致）：

| 项 | 值 |
|---|---|
| gate 家族均衡 AUC | 0.7694 |
| net 家族均衡 AUC | 0.7087 |
| τ_gate | 0.538072475100225 |
| τ_net | 0.5523107182337628 |
| 验证集卸载 | 119/588（oracle 捕获 16.05%） |

⚠️ `router.build` 只读 `original/snapshots/k####/prefix_trace.json`；一旦发现自己在读
`trace.json` 或 `branches/`，就是泄漏了第 k 步及未来的信息。

