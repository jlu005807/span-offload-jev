# 环境与 AppWorld 位置

这份文档回答一个问题：**克隆下来之后，把东西放在哪、装什么、怎么自检。**

## 目录约定

```
<span-offload>/                 ← 本仓库
├── appworld/                   ← 上游 AppWorld（**必须放在这里**，见下）
├── traceoffload/ router/ experiments/ configs/ tests/ docs/
└── runs/                       ← 跑批产物（git 忽略，可不存在）
```

## AppWorld 放哪、怎么装

本仓库**不包含** AppWorld 源码（上游 374 MB）。克隆到仓库根目录下的 `appworld/`：

```bash
cd <span-offload>
git clone https://github.com/StonyBrookNLP/appworld.git appworld
git -C appworld checkout 42b5bcf        # 本文档验证过的提交
pip install -e appworld
```

数据（task 定义、DB 种子、OpenAPI 规格）在 `appworld/data/`，约 202 MB。如果它不在仓库里：

```bash
cd appworld && appworld install && appworld download data
```

**为什么必须是 `<repo>/appworld`**：两处代码写死了这个相对位置 —

- `experiments/run_round.py` 会把 `<输出目录>/_root/data` 软链到 `<repo>/appworld/data`
- `router/build.py --openapi-dir appworld/data/api_docs/openapi` 需要 OpenAPI 规格来判读写

如果放在别处，用软链指过来即可：`ln -s /path/to/appworld appworld`。

## 三个本地服务

| 端口 | 服务 | 用途 |
|---|---|---|
| 8300 | Qwen3.8-27B（vLLM） | 大模型 |
| 8800 | Qwen3.5-9B（vLLM） | 小模型 |
| 11434 | Ollama `qwen3-embedding:0.6b` | 116 维里的两段嵌入（1024 维） |

```bash
curl -s http://127.0.0.1:8300/v1/models | head -c 120
curl -s http://127.0.0.1:8800/v1/models | head -c 120
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys;print([m['name'] for m in json.load(sys.stdin)['models']])"
```

⚠️ 嵌入服务**不是可选项**：拿不到 1024 维向量时预测器会 fail-open 到"走大模型"，
表现为卸载率莫名偏低（`miss_no_vector > 0`）。

## Python 依赖

```bash
pip install pydantic PyYAML numpy          # 运行 traceoffload + experiments
pip install scikit-learn                   # 只有 router/（训练）需要
pip install pytest                         # 跑测试
```

## 自检

```bash
python3 -c "import traceoffload.runner; print('runtime ok')"
python3 -m pytest tests -q                 # 应该是全绿
python3 experiments/run_appworld.py --help | head -5
```
