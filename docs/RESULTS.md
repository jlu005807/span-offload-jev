# 一轮实测结果（2026-10-18，本仓库代码）

90 个实例 × 1 轮 × 5 臂，按 `experiments/instances-90.txt` 顺序、4 并发、每臂一个空 M4 库。

## 实测

| 臂 | n | 成功率 | 卸载率 | 总 token | 每次调用 | 模型延迟 |
|---|---:|---:|---:|---:|---:|---:|
| native-large | 90 | 90.0% | 0.0% | 37.8M | 20.5 | 120.4s |
| native-small | 90 | 37.8% | 100% | 22.2M | 24.7 | 200.7s |
| offload-shadow | 90 | 86.7% | 0.0% | 35.0M | 20.1 | 125.3s |
| offload-active | 90 | 74.4% | 17.7% | 34.4M | 20.4 | 130.6s |
| m4-on | 90 | 77.8% | 21.0% | 31.9M | 20.4 | 121.2s |

## 与参考（5 轮 × 90，折算到每 90 个实例）

| 臂 | 成功率 | 卸载率 | 总 token | 每次调用 | 模型延迟 |
|---|---|---|---|---|---|
| native-large | 89.8 → **90.0** | 0 → 0 | 33.5M → 37.8M | 20.7 → 20.5 | 127.0 → 120.4 |
| native-small | 38.9 → **37.8** | 100 → 100 | 22.8M → 22.2M | 25.1 → 24.7 | 189.6 → 200.7 |
| offload-shadow | 88.4 → 86.7 | 0 → 0 | 34.9M → 35.0M | 20.6 → 20.1 | 139.0 → 125.3 |
| offload-active | 71.6 → 74.4 | 18.7 → 17.7 | 30.4M → 34.4M | 19.8 → 20.4 | 131.8 → 130.6 |
| m4-on | 69.6 → 77.8 | 19.6 → 21.0 | 30.1M → 31.9M | 20.1 → 20.4 | 127.6 → 121.2 |

**结论**：调用数/任务、模型延迟、卸载率三项在五个臂上都对得上（卸载率差 ≤1.4pp）；
成功率差 0–8pp，落在 90 样本的 95% 区间（±9pp）内。单轮样本小于参考的 450，这是主要差异来源。

## 需要记录的两件事

1. **offload-active 有 4 个任务曾整条退化成全大模型**（`miss_no_vector` 40/33/16/13）：
   并发下嵌入链路瞬时故障，同任务单跑时 miss=0。已用 `experiments/repair.py` 串行重跑修好，
   表中是该臂修好后的数（修前 76.7% / 16.3%）。
2. **native-small 的 `f86d850_2` 单任务跑了 >10 分钟**，被中止后由补跑重跑完成；
   `run_round.py` 已加 `--timeout`（默认 1800s）。

## 复现命令

```bash
for arm in native-large native-small offload-shadow offload-active m4-on; do
  python3 experiments/run_round.py --arm $arm --rounds 1 --workers 4
done
python3 experiments/repair.py --arm offload-active   # 有 miss_no_vector 时
python3 experiments/report.py --root runs/one_round
```

---

# 第二轮：带机制计数（`runs/round_counters`）

计数器接线后的重跑（native 两臂按要求未跑）。**每臂 90 个实例，0 失败行。**

| 臂 | n | 成功率 | 卸载率 | 总 token | 每次调用 | 模型延迟 |
|---|---:|---:|---:|---:|---:|---:|
| offload-active | 90 | 75.6% | 19.5% | 28.9M | 19.6 | 124.5s |
| offload-shadow | 90 | 92.2% | 0.0% | 32.3M | 19.8 | 115.3s |
| m4-on | 90 | 71.1% | 18.5% | 30.6M | 19.4 | 124.5s |

## 机制计数（summary 里现在能直接读到）

| 臂 | 准入通过 | checkpoint | commit | rollback 成功 | rollback 失败 | post-check 拒 | M4 记忆拒 | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offload-active | 343 | 182 | 155 | 27 | 0 | 32 | 0 | 32 |
| offload-shadow | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| m4-on | 316 | 165 | 146 | 19 | 0 | 31 | 4 | 36 |

三处自洽，说明计数不是乱加的：

1. **checkpoint = commit + rollback**：182 = 155 + 27；165 = 146 + 19
2. **fallback ≈ 守卫拒绝**：active 32 fallback / 32 post-check 拒；m4-on 36 fallback = 31 post-check + 4 记忆拒 + 1 次模型侧失败
3. **post-check 拒 ≠ rollback**：m4-on 有 31 次 post-check 拒但只有 19 次回滚 —— 差额是只读/未知效应，没有可回滚的东西

**shadow 臂全零**是对照：它执行全走大模型，守卫一次都不跑，所以六个计数都是 0。
这反证前两行不是"跑批就会涨"。

## 与参考对照

| 臂 | 成功率 | 卸载率 |
|---|---|---|
| offload-active | 71.6 → 75.6 | 18.7 → 19.5 |
| offload-shadow | 88.4 → 92.2 | 0 → 0 |
| m4-on | 69.6 → 71.1 | 19.6 → 18.5 |

三个臂的成功率都在参考值 +2~+4pp，卸载率差 ≤1.4pp。单轮 90 个样本的 95% 区间是 ±9pp，
这个量级的偏差属于抽样噪声，不是行为差异。

