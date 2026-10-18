# TraceOffload

按 span 决定「这一步交给大模型还是小模型」的运行时，跑在 AppWorld 上。

- **M1 建模**：把一次模型调用（一个 span）编码成画像 + 向量
- **M2 路由**：116 维两阶段预测器，`p_gate ≥ τ_gate ∧ p_net ≥ τ_net` 才卸载
- **M3 受控执行**：小模型的动作先过守卫、再事务化执行；被拒/失败就回滚并回退大模型
- **M4 记忆**：跨 span 记录「哪些动作在相似场景下失败过」，供守卫在执行前拦截

Agent 本体一行不改：只包住 `model.generate` 与 `env.execute` 两个接缝。

---

## 为什么需要它

Agent 的每一步都要调一次大模型。大模型贵、慢，但很多步骤是"照猫画虎"的机械动作，
小模型够用。问题是 agent **有状态**：交给小模型省了钱，但一步做错可能污染整个任务状态。

所以这里同时做四件事：**判断**（这一步卸不卸）、**兜底**（卸了但输出不行，在执行前拦下来）、
**可逆**（真落下副作用要能回滚）、**学习**（每一步的结果记账，让下一步更准）。

---

## 这个仓库里有什么

| 路径 | 内容 | 说明文档 |
|---|---|---|
| `traceoffload/` | 运行时：contracts + M1/M2/M3/M4 + coordinator/runner + AppWorld 适配器 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| `router/` | 离线链路：特征 → 嵌入 → 训练 → 选阈值 → 导出 artifact | [docs/OFFLINE.md](docs/OFFLINE.md) |
| `experiments/` | 单任务入口、跑批、汇总、补跑、机制核验 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| `configs/` | `pilot.yaml`（基础）+ `arms/*.yaml`（五个臂） | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| `tests/` | 只覆盖保留下来的模块（`python3 -m pytest tests -q`） | — |
| `docs/` | 背景、数据、训练、架构、实验、结果 | 见下方「文档地图」 |

---

## 快速开始（五分钟）

```bash
# 1) AppWorld：必须克隆到 <repo>/appworld
git clone https://github.com/StonyBrookNLP/appworld.git appworld
git -C appworld checkout 42b5bcf && pip install -e appworld

# 2) 依赖
pip install pydantic PyYAML numpy scikit-learn pytest

# 3) 三个本地服务：8300 大模型 / 8800 小模型 / 11434 嵌入
#    探活命令见 docs/SETUP.md

# 4) 自检
python3 -m pytest tests -q

# 5) 跑一个任务（用仓库自带的训练好的预测器，不需要重训）
mkdir -p runs/demo/_root/experiments/outputs
ln -sfn "$PWD/appworld/data" runs/demo/_root/data
APPWORLD_ROOT="$PWD/runs/demo/_root" python3 experiments/run_appworld.py \
    --config configs/arms/offload-active.yaml --mode classifier_route \
    --task-id 099174d_1 --output-dir runs/demo

# 6) 跑一轮并出表
python3 experiments/run_round.py --arm offload-active --rounds 1 --workers 4
python3 experiments/report.py --root runs/one_round
```

完整的环境说明见 [docs/SETUP.md](docs/SETUP.md)。

---

## 数据从哪来

```
一个 AppWorld 任务
  → original/            原生 agent 全大模型跑一遍（含每一步的快照）
  → branches/k####/     从第 k 步起，分别用大模型重放 / 小模型接替，各 10 次
  → 对比两条分支        → 这一步"卸载值不值"的标签
```

| 集合 | 任务数 | cell 数 | 用途 |
|---|---:|---:|---|
| 训练集 | 249 | 4950 | 训练路由器 |
| 验证集 | 33 | 588 | 挑阈值 |
| 测试集 | 90 | — | 在线跑批 |

三份清单**互不重叠**，完整 task_id 列表和采集步骤见 [docs/DATA.md](docs/DATA.md)
（重采数据前先查那张表，避免重复）。

---

## 路由器怎么训

```
116 维 = [prefix_n_A, log_est_prompt_tokens]          2 列
       + PCA50(V2_goal_recent 嵌入 1024 维)            50 列
       + PCA64(C_bare_w10  嵌入 1024 维)               64 列

两个 logit：gate ← l2_new（本地做对不对） ；net ← net_saving>0（省不省钱）
两个阈值：在验证集上最大化 Σ净节省 —— 不是 F1、不是准确率
```

```bash
python3 -m router.build   --runs runs/训练集  --out data/训练集  --openapi-dir appworld/data/api_docs/openapi
python3 -m router.embed   texts   --texts data/训练集/texts.jsonl --out data/训练集/aw3_emb_v2goal.jsonl
python3 -m router.embed   actions --texts data/训练集/actions_texts.jsonl \
    --vectors data/训练集/emb_C_bare_w10.npz --ids data/训练集/cell_ids.npz
python3 -m router.train   --train-dir data/训练集 --val-dir data/验证集 ... \
    --out out/model.json --val-scores out/val_scores.jsonl
python3 -m router.select_threshold --scores out/val_scores.jsonl --out out/tau.json
python3 -m router.export  --model out/model.json ... \
    --out router/artifact/l2_add4_twostage_logit.json
```

**这个仓库已经带了训好的 artifact**（`router/artifact/`），跑实验不需要重训。
重训需要 128 MB 派生数据，清单与来源见 [docs/DATA.md](docs/DATA.md) §2。
管线细节与逐位复现记录见 [docs/OFFLINE.md](docs/OFFLINE.md)。

---

## 架构怎么跑

```
Agent（ReAct，不动）
   │ ① model 接缝                  │ ② env 接缝
   ▼                                ▼
TraceOffloadModelWrapper        AppWorldToolProxy
   └──────────────┬─────────────────┘
                  ▼
              Coordinator ── M1 → M2 → M3
                  │               │
                  └── M4 ←─ 记账 ──┘
```

一次调用的顺序、每层职责、配置键与装配点见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
**fallback 与 rollback 到底有没有发生、长什么样**见 [docs/MECHANISMS.md](docs/MECHANISMS.md)。

---

## 文档地图

| 文档 | 讲什么 |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | 环境、AppWorld 放哪、三个服务、自检 |
| [docs/DATA.md](docs/DATA.md) | 数据怎么采、派生数据清单、三份实例清单 |
| [docs/DATA_PACKAGE.md](docs/DATA_PACKAGE.md) | 作者提供的数据包：放哪、怎么校验 |
| [docs/OFFLINE.md](docs/OFFLINE.md) | 116 维特征、两个 logit、阈值挑选、复现记录 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | M1–M4 各模块、装配顺序、配置键 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 五个臂、跑法、指标口径、验收 |
| [docs/MECHANISMS.md](docs/MECHANISMS.md) | fallback / rollback 的判据与实测 |
| [docs/RESULTS.md](docs/RESULTS.md) | 两轮实测结果与参考表对照 |

---

## 已知边界

- **`appworld/` 不在 git 里**（上游 374 MB），克隆位置见 [docs/SETUP.md](docs/SETUP.md)。
- **`runs/` 不在 git 里**（训练集 70 GB + 验证集 8.8 GB）：都是模型轨迹，太大且含模型输出。
- **派生训练数据（126.5 MB）不在 git 里**：作者单独提供数据包，解压到 `data-package/` 后用
  `python3 experiments/check_data.py` 校验。不重训就不需要，详见 [docs/DATA_PACKAGE.md](docs/DATA_PACKAGE.md)。
- **分类器只保留 116 维 add4**：52 维 v4、bash 语法效应分析器、SWE-bench 接入、反馈发布器都已删除，
  它们不在最终实验的路径上。
- **summary 里的机制计数需要看第二轮起的数据**：`runs/one_round` 那轮跑在接线之前，
  `fallback_by_reason` 等字段恒为 0；`runs/round_counters` 才是可用的。
