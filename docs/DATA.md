# 数据：怎么采、放在哪、有哪些实例

## 1. 数据是怎么来的（span-offload 采集器）

训练数据不是跑一遍任务就能拿到的，而是**对每个任务做一次"前缀 + 多分支重放"**：

```
一个 AppWorld 任务（原生 agent，全大模型跑完）
  └─ original/            完整轨迹（含第 k 步及之后，特征侧禁止读）
      ├─ trace.json
      ├─ calls.json / tool_calls.json / stats.json
      └─ snapshots/k0001 … k00NN/
          ├─ prefix_trace.json     ← 第 k 步之前的前缀（特征只许读它）
          └─ state_ref.json        ← 该时刻的世界状态哈希
  └─ branches/k0001 … k00NN/
      ├─ llm_replay/rep01 … rep10/    大模型从第 k 步重跑（对照）
      └─ slm_offload/rep01 … rep10/   小模型从第 k 步接替（实验组）
  └─ status.json
```

对比 "第 k 步交给谁" 的两个分支，就得到这一步的标签（本地是否出错、任务是否仍成功、
下游是否被搞坏）。**一个 (task, k) 就是一个 cell**，训练/验证集分别 4950 / 588 个。

### 采集器在哪

采集代码（`spanOffload/` 与 `bench-resources/run_appworld_spanoffload.py`）在仓库精简时被删除，
因为它不在"路由器 → 接入 → 跑测试"这条链上。需要重采数据时从 tag 取回：

```bash
git worktree add ../collector pre-trim      # 或 git archive pre-trim | tar -x -C ../collector
cd ../collector
python bench-resources/run_appworld_spanoffload.py \
    --config configs/spanoffload-qwen-local-full.yaml \
    --task-id 024c982_1 \
    --output-dir runs/训练集 \
    --repeats 10 \
    --modes llm_replay,slm_offload,original_span
```

关键参数：

| 参数 | 含义 |
|---|---|
| `--repeats 10` | 每个分支重放 10 次（标签里的成功率/方差都靠它） |
| `--modes` | `original_span` 只跑原始轨迹；`llm_replay,slm_offload` 才是分支 |
| `--k` | 只跑某一步（调试用） |
| `--skip-branches` | 只采 original（新任务摸底用，省时间） |

⚠️ 采集要 70 GB / 8.8 GB 量级的磁盘，且每任务 10×2 次重放，**不建议全量重采**。

## 2. 派生数据（训练直接吃的东西）

采集产物之上还要算三样东西，脚本在 tag `pre-trim` 的 `classfierTrain/` 与 `router/` 里：

| 产物 | 谁产的 | 训练是否需要 |
|---|---|---|
| `aw3_features.jsonl` + `aw3_emb_v2goal.jsonl` | `router/build.py` + `router/embed.py` | ✅ |
| `actions_texts.jsonl` + `emb_C_bare_w10.npz` + `cell_ids.npz` | `router/actions.py` + `router/embed.py` | ✅ |
| `labels_appworld_0920.jsonl`（`l2_new`） | tag 里的标注器 | ✅ |
| `net_saving_labels.jsonl` | tag 里的 `net_saving_label.py` | ✅ |
| `token_labels.jsonl` | tag 里的 `token_saving_label.py` | 只在算 net_saving 时需要 |

合计约 **126.5 MB**（最大两份是两段 1024 维嵌入：69 MB + 23 MB）。**不在 git 里**。

作者会单独提供压缩包：解压到 `<repo>/data-package/`，用
`python3 experiments/check_data.py` 校验（存在性 + 字节数 + sha256 + 行数）。
文件清单与每个文件的含义见 [DATA_PACKAGE.md](DATA_PACKAGE.md)。
也可以按上面的流程从 `runs/` 重新生成。

## 3. 三份实例清单（防止重复采集）

划分规则：**task_id 互不重叠**（实测交集为 0）。同一 task 的不同 `k` 是不同 cell，不算重复。

| 集合 | 任务数 | cell 数 | 用途 | 数据根 |
|---|---:|---:|---|---|
| 训练集 | 249 | 4950 | 训练路由器 | `runs/训练集` |
| 验证集 | 33 | 588 | 挑阈值、报离线指标 | `runs/验证集` |
| 测试集 | 90 | — | 在线跑批（只跑 1 轮） | AppWorld `train` split |

**新增数据前先查下面三张表**：task_id 已存在就别再采；要扩训练集只能加**没出现过的 task**。

## 4. 实例清单

### 训练集（249 个）

```
024c982_1 024c982_2 024c982_3 07b42fd_2 07bb666_1 099174d_2 09ac073_2
0d01c76_1 0d01c76_2 0d01c76_3 0d22252_1 0d8a4ee_3 0de03ea_3 1150ed6_1
1150ed6_2 1150ed6_3 13547f5_1 13547f5_2 13547f5_3 143527e_3 16be9ce_1
18670a5_1 1a79e37_1 1b2bae9_3 1b988ac_2 1c4bd27_3 20c1328_3 21abae1_1
229360a_3 22cc237_1 23cf851_1 23d431c_2 245cb43_1 258796c_1 270f1ff_1
270f1ff_2 270f1ff_3 277d81d_1 27e1026_2 287e338_3 29a7b7e_2 29caf6f_3
2a163ab_1 2c544f9_1 2d9f728_3 2e9b91e_1 2ff4dfb_3 302c169_3 31dc501_1
31dc501_2 31dc501_3 325d6ec_2 34d9492_2 3650990_1 365e0a3_2 36705ae_3
37a8675_3 383a053_2 383cbac_2 396c5a2_2 3aa1a22_1 3aa1a22_2 3aa1a22_3
3b8fb7a_3 3c13f5a_3 3d9a636_3 3f3c139_3 3fcc458_2 4242c97_3 425a494_1
432dc7a_1 4441ee9_3 476b213_3 4815c06_1 4ac4a8d_3 4d12842_1 4ec8de5_1
4fab96f_1 50e1ac9_1 522e5e5_3 5238afc_1 530b157_2 552869a_3 57c3486_1
5800354_2 59bcfc8_3 59fae45_1 59fae45_2 59fae45_3 5a83b05_1 5a83b05_2
5a83b05_3 5e27cd7_3 60d0b5b_3 6104387_3 6171bbc_1 634f342_1 6474048_2
6588a51_1 66b7899_3 68ee2c9_2 690d51b_3 692c77d_2 69ba40f_1 6a5e690_3
6b6ca61_1 6b6ca61_2 6b6ca61_3 6bdbc26_3 6c2c621_1 6d59d90_3 6ea6792_1
6f4b9a5_1 6f4b9a5_2 6f4b9a5_3 7238049_1 7264edc_1 7434096_3 7574325_2
771d8fc_2 77bcb81_2 7847649_2 7bdaf23_1 7d26579_3 7d7fbf6_1 7e1be84_3
80acbaf_3 81be677_3 82d2e01_1 82e2fac_2 83a7951_1 83a7951_2 83a7951_3
8749218_3 8a13317_2 8ce6779_3 8d42650_1 8f79e35_2 9016950_1 906f2c7_1
90adc3f_2 9126bf0_2 92fe421_1 953b296_1 96bf160_1 986aa4e_1 986aa4e_2
9871968_2 98d2608_2 998908e_3 9a16324_3 9aae7da_3 9b2dc64_3 9bf2c8a_1
9dabbc9_1 9ef034e_3 9ef798c_1 9ef798c_2 9ef798c_3 a132254_2 a1d3dfd_3
a30375d_3 a3ba388_2 a53a8fd_3 a5b0084_2 a676f2a_2 a7179fa_3 a8f302f_1
a97e35d_1 aa8502b_2 aaa25aa_1 ac62177_1 adb1060_3 af84964_1 afc0fce_1
afc4005_2 b05f69a_3 b08d5ef_1 b0934aa_3 b0a8eae_1 b119b1f_3 b3bdcc1_1
b4707e8_1 b68eb83_2 b6d1f70_3 b71318e_3 b7a9ee9_1 b9c5c9a_3 ba46d91_2
baeb104_2 bcb9696_2 bde252e_3 c1091c7_2 c77c005_1 c901732_1 c9458ed_3
cb4ee7a_2 ccb4494_3 cdaaea5_2 cdf61bd_3 ce359b5_2 cef9191_3 cf6abd2_3
d037dc8_1 d0b1f43_1 d18139b_1 d194965_1 d37c235_3 d4e9306_1 d6d8cb6_2
d8e490b_3 d9987f6_2 dac78d9_2 dbc0276_3 dc5c5c6_3 dcb2559_1 dcbd34b_2
df61dc5_1 e0fe09c_3 e201314_2 e3d6c94_2 e52623a_3 e70b117_3 e775c78_1
e7a10f8_1 e7f15ba_1 e85d92a_3 eb5ad85_2 ec437da_2 ec655df_3 efc3cea_2
f099b4c_1 f323bae_2 f37e804_3 f3a6713_1 f691597_1 f6936d4_3 f6f2e67_1
f861c32_2 f99d726_1 fa327a6_1 fac291d_1 fb05fed_2 fd1f8fa_3 fdc4b74_3
fddb6b6_3 ff58e36_3 ffe6d5e_2 ffea2b5_2
```

### 验证集（33 个）

```
042a9fc_2 042a9fc_3 09ac073_1 09b0ee6_1 09b0ee6_2 09b0ee6_3 0a9d82a_1
0a9d82a_2 0a9d82a_3 0d8a4ee_1 0de03ea_1 0de03ea_2 143527e_2 166f4ff_1
166f4ff_2 166f4ff_3 18670a5_3 229360a_1 396c5a2_1 3b8fb7a_1 3d9a636_2
50e1ac9_3 522e5e5_2 634f342_2 ac62177_2 b68eb83_1 cdf61bd_1 d37c235_2
d9987f6_1 f6f2e67_2 f6f2e67_3 fb05fed_3 fdc4b74_1
```

### 测试集（90 个）

```
099174d_1 09ac073_3 1a79e37_2 1b988ac_1 22cc237_2 23cf851_3 27e1026_1
29caf6f_2 2c544f9_2 2c544f9_3 2e9b91e_2 2ff4dfb_1 321ec38_3 325d6ec_1
37a8675_1 383a053_1 3b8fb7a_2 3c13f5a_2 3fcc458_1 4242c97_1 4441ee9_2
4ac4a8d_2 4d12842_2 552869a_2 5800354_3 5e27cd7_2 6474048_3 66b7899_2
68ee2c9_3 69ba40f_3 6ea6792_2 7238049_2 7434096_1 76f2c72_1 77bcb81_3
7d26579_1 7d26579_2 7e1be84_2 81be677_2 8a13317_1 8a13317_3 90adc3f_3
96bf160_2 988af8e_2 98d2608_1 998908e_1 9a16324_1 a3ba388_1 a7179fa_1
adb1060_2 af84964_2 afc0fce_2 b05f69a_1 b119b1f_1 b4707e8_2 b68eb83_3
b6d1104_1 b6d1104_2 b6d1104_3 b71318e_1 b71318e_2 b7a9ee9_3 b9c5c9a_1
ba46d91_1 bde252e_1 c77c005_3 c8f5f44_2 cb4ee7a_1 ccf4b82_1 ccf4b82_2
ccf4b82_3 d0b1f43_2 d6ac34d_1 d6ac34d_2 d6ac34d_3 dbc0276_1 dbc0276_2
dc5c5c6_2 e70b117_2 e7a10f8_2 e7a10f8_3 f323bae_3 f37e804_1 f3f60f0_1
f3f60f0_2 f3f60f0_3 f691597_3 f6936d4_2 f861c32_1 f86d850_2
```
