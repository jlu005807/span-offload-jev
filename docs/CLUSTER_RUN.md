# JEV 第二阶段：服务器部署与单任务验收

本文件只描述从服务器环境准备到一个 AppWorld 任务验收的步骤。JEV 使用原生 System One 服务；现有 LLM/SLM 可以继续使用 vLLM，JEV 不直接使用 vLLM 的 `/v1/chat/completions`。

## 1. 设置服务器路径

先只设置仓库路径；`.env` 要在代码同步后创建，避免 Git 克隆到一个已经被手工创建的非空目录：

```bash
export PROJECT=/share/home/$USER/span-offload-jev
mkdir -p /share/home/$USER
```

项目会在 `experiments/run_appworld.py` 启动时自动读取项目根目录的 `configs/.env`。建议把服务器路径、节点地址、端口和密钥放在这里；不要把模型路由阈值、`model_id` 或 revision 放在 `.env`，这些属于实验配置，应保留在服务器专用 YAML 中以便复现。不要在登录节点加载模型，后续服务命令应在已分配 GPU 的计算节点或共享 Slurm 作业中执行。

## 2. 同步项目代码

```bash
cd /share/home/$USER
if [ ! -d "$PROJECT/.git" ]; then
  git clone <项目仓库地址> "$PROJECT"
fi
cd "$PROJECT"
git fetch --all
git checkout feat/jev-router
git pull --ff-only
git status

umask 077
if [ ! -f configs/.env ]; then
cat > configs/.env <<'EOF'
PROJECT=/share/home/REPLACE_WITH_USER/span-offload-jev
APPWORLD_ROOT=/share/home/REPLACE_WITH_USER/span-offload-jev/appworld-root

# JEV 原生服务
JEV_MODEL_ID=alanhuangya/OpenJev-1.7B
JEV_REVISION=7418ae8b6e3b6a6507a8a58729ae491cd1c732d9
JEV_HOST=REPLACE_WITH_JEV_HOST
JEV_PORT=8900

# 已有或将要启动的 OpenAI-compatible vLLM 服务
LLM_HOST=REPLACE_WITH_LLM_HOST
LLM_PORT=8300
LLM_MODEL_ID=Qwen3.8-27B
SLM_HOST=REPLACE_WITH_SLM_HOST
SLM_PORT=8800
SLM_MODEL_ID=Qwen3.5-9B

OPENAI_API_KEY=dummy
MODEL_SERVER_URL=
# 仅当 JEV 服务需要认证时填写
JEV_API_KEY=
EOF
fi

# 首次运行前编辑 configs/.env，把所有 REPLACE_WITH_* 替换为实际值。
set -a
source configs/.env
set +a
mkdir -p "$APPWORLD_ROOT"
```

如果项目已有服务器副本，使用 `git fetch`、`git checkout` 和 `git pull` 更新；不要覆盖服务器已有实验目录。

## 3. 安装 AppWorld 和 Python 环境

```bash
cd "$PROJECT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[jev,dev]'

if [ ! -d appworld ]; then
  git clone https://github.com/StonyBrookNLP/appworld.git appworld
  git -C appworld checkout 42b5bcf
fi

python -m pip install -e appworld
APPWORLD_ROOT="$APPWORLD_ROOT" appworld install
APPWORLD_ROOT="$APPWORLD_ROOT" appworld download data
```

检查环境：

```bash
python -c 'import traceoffload, appworld, httpx, yaml; print("environment ok")'
python -m pytest tests -q -o addopts=
```

## 4. 复用或部署已有 LLM 和 SLM

LLM 和 SLM 继续使用 OpenAI-compatible vLLM；JEV 是单独的原生 System One 服务，不替换这两个模型。先检查集群上是否已有服务：

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 15 -fsS "http://${LLM_HOST}:${LLM_PORT}/v1/models"
curl --noproxy '*' --connect-timeout 5 --max-time 15 -fsS "http://${SLM_HOST}:${SLM_PORT}/v1/models"
```

再各发送一次最短聊天请求，确认服务不只是端口可达：

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 60 -fsS \
  -H 'Content-Type: application/json' \
  "http://${LLM_HOST}:${LLM_PORT}/v1/chat/completions" \
  -d "{\"model\":\"${LLM_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}"

curl --noproxy '*' --connect-timeout 5 --max-time 60 -fsS \
  -H 'Content-Type: application/json' \
  "http://${SLM_HOST}:${SLM_PORT}/v1/chat/completions" \
  -d "{\"model\":\"${SLM_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}"
```

两个请求都成功时，记录响应中的实际 `id`，后续直接复用服务，不要重复加载模型。服务必须同时支持：

```text
GET  /v1/models
POST /v1/chat/completions
```

如果某个服务不存在，优先使用集群已有的 vLLM Slurm 脚本，而不是重新写启动脚本。典型操作如下（将路径、节点和脚本名替换为服务器实际值）：

```bash
# Qwen3.8-27B：已有脚本通常位于模型目录
sbatch -w <llm计算节点> <LLM_MODEL_DIR>/04_serve.sbatch
# 若集群要求固定 TP/并发，使用已有的 04_serve_tp4_*.sbatch

# Qwen3.5-9B
sbatch -w <slm计算节点> <SLM_MODEL_DIR>/04_serve.sbatch
squeue -u "$USER"
```

提交后从脚本日志读取真实 `VLLM_HOST`、`VLLM_IP`、`VLLM_PORT` 和 `SERVED_MODEL_NAME`，填回 `configs/.env`，然后重复两个 `curl`。如果只能从登录节点访问计算节点，建立 SSH 隧道并让 `.env` 使用隧道本地端口：

```bash
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L 8300:<llm计算节点IP>:8300 \
  <登录用户名>@<登录跳板机>
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L 8800:<slm计算节点IP>:8800 \
  <登录用户名>@<登录跳板机>
```

此时把 `LLM_HOST=127.0.0.1`、`SLM_HOST=127.0.0.1` 写入 `.env`，并再次检查 `http://127.0.0.1:8300/v1/models` 和 `http://127.0.0.1:8800/v1/models`。不要在同一端口同时启动两个服务。

然后复制服务器配置并写入真实 endpoint 和模型 ID：

```bash
cp configs/arms/jev-shadow.yaml configs/arms/server-jev-shadow.yaml
cp configs/arms/jev-active.yaml configs/arms/server-jev-active.yaml
```

两个文件中的 LLM/SLM 字段都要与服务一致（当前 YAML 加载器不会展开 `${VAR}`）：

```yaml
models:
  large:
    model_id: Qwen3.8-27B       # /v1/models 返回的 id
    served_model_name: Qwen3.8-27B
    api_base: http://<LLM_HOST>:8300/v1
  small:
    model_id: Qwen3.5-9B
    served_model_name: Qwen3.5-9B
    api_base: http://<SLM_HOST>:8800/v1
```

若服务返回的 `id` 不同，以服务返回值为准；`model_id` 和 `served_model_name` 应保持与服务启动参数一致。`api_key: dummy` 仅适用于无认证的内部服务。

## 5. 申请 Jev GPU 资源

如果使用 Slurm，先申请一张 GPU：

```bash
srun --partition=<partition> \
     --account=<account> \
     --qos=<qos> \
     --gres=gpu:A800:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=01:00:00 \
     --pty bash
```

进入计算节点后确认：

```bash
hostname
nvidia-smi
```

如果 LLM/SLM 与 Jev 共享一张 GPU，必须在同一个 Slurm 作业内启动，并先降低 SLM 的显存预算；不要为 Jev 再启动一个独立 GPU 作业。

## 6. 安装并启动 Jev 原生服务

```bash
export PROJECT=/share/home/$USER/span-offload-jev
cd "$PROJECT"
set -a
source configs/.env
set +a
source .venv/bin/activate
python -m pip install 'wev-ai[serve]'
wev serve --help
```

确认参数后启动首个候选模型。模型 revision 必须固定为 `$JEV_REVISION`；如果当前 `wev` 版本支持 `--revision`，显式传入该参数：

```bash
wev serve \
  --model "$JEV_MODEL_ID" \
  --revision "$JEV_REVISION" \
  --host 0.0.0.0 \
  --port "$JEV_PORT"
```

如果 `--help` 不支持 `--revision`，先将该 revision 下载到固定本地目录，再把 `--model` 改为本地权重目录。不要使用未固定 revision 的 latest 权重。

JEV 不使用 vLLM 的聊天接口。服务必须提供：

```text
GET  /health
POST /v1/systemone
```

## 7. 检查 Jev 服务

另开终端，在服务节点或可达节点执行：

```bash
export PROJECT=/share/home/$USER/span-offload-jev
cd "$PROJECT"
set -a
source configs/.env
set +a
export JEV_ENDPOINT="http://${JEV_HOST}:${JEV_PORT}"
curl --noproxy '*' --connect-timeout 5 --max-time 15 -fsS "$JEV_ENDPOINT/health"
curl --noproxy '*' --connect-timeout 5 --max-time 15 -fsS "$JEV_ENDPOINT/v1/models"
```

发送一次原生短请求：

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 30 \
  -H 'Content-Type: application/json' \
  "$JEV_ENDPOINT/v1/systemone" \
  -d "$(python - <<'PY'
import json, os
print(json.dumps({
    'model': os.environ['JEV_MODEL_ID'],
    'state': {'model_info': {
        'large_model': os.environ['LLM_MODEL_ID'],
        'small_model': os.environ['SLM_MODEL_ID'],
    }, 'context': {'goal_text': 'Read a known record', 'history': []}},
    'questions': {'route': {
        'type': 'choice',
        'instructions': 'Choose the execution model for the next Agent call.',
        'criteria': {
            'LLM': 'Use the large model.',
            'SLM': 'Use the small model with existing guards and fallback.',
        },
    }},
}, ensure_ascii=False))
PY
)"
```

响应必须包含 `model`、`answers.route.type=choice` 以及有限且归一化的 `LLM`/`SLM` 概率。

## 8. 配置真实 Jev 服务地址和模型身份

编辑第 4 步生成的两个服务器配置文件，至少修改：

```yaml
runner:
  jev:
    enabled: true
    endpoint: http://<JEV_HOST>:8900
    model_id: <v1/models和响应中确认的实际model>
    expected_revision: 7418ae8b6e3b6a6507a8a58729ae491cd1c732d9
    api_key_env: null  # 认证服务才改为 JEV_API_KEY
```

`endpoint` 只写 origin，不要追加 `/v1`。如果项目在本机通过 SSH 隧道运行，才使用 `http://127.0.0.1:8900`。

## 9. 接入真实 tokenizer 预检

正式任务前必须使用 Jev 服务实际 tokenizer、原生模板和最大输入限制完成预检。项目客户端在没有预检器时会记录 `input_unverified` 并拒绝发送，这是安全保护。

预检器必须：

1. 接收即将发送的完整 `model/state/questions` 请求；
2. 使用与服务相同的 tokenizer 和模板计算完整 input token 数；
3. 确认不会被服务截断；
4. 返回真实 token 数，超限时抛出 `input_too_long`。

当前命令行入口没有通用的 tokenizer 参数；需要在部署适配代码中调用 `AppWorldPilotRunner(..., jev_input_preflight=...)`，或为当前模型补一个 CLI 适配器。完成后先用客户端测试确认 JSONL 中 `input_metadata.input_tokens` 有值，再执行下面的任务命令。

## 10. 运行一个 shadow 任务

shadow 需要旧 add4 路由及其嵌入服务；如果嵌入服务不可用，先执行 active 验收。

```bash
export PROJECT=/share/home/$USER/span-offload-jev
cd "$PROJECT"
set -a
source configs/.env
set +a
source .venv/bin/activate
export APPWORLD_ROOT="$APPWORLD_ROOT"
RUN_ID=$(date +%Y%m%d-%H%M%S)

python experiments/run_appworld.py \
  --config configs/arms/server-jev-shadow.yaml \
  --mode classifier_route \
  --task-id 099174d_1 \
  --output-dir "runs/jev-shadow-$RUN_ID" \
  --experiment-name "jev-shadow-099174d_1-$RUN_ID"
```

检查：

```bash
ls "runs/jev-shadow-$RUN_ID"
cat "runs/jev-shadow-$RUN_ID"/routing-decisions-*.jsonl
cat "runs/jev-shadow-$RUN_ID"/appworld-classifier_route-summary.json
```

确认 Jev 请求成功、输入 token 数有记录、shadow 最终仍执行旧路由计划。

## 11. 运行一个 active 任务

必须使用新的输出目录：

```bash
export PROJECT=/share/home/$USER/span-offload-jev
cd "$PROJECT"
set -a
source configs/.env
set +a
source .venv/bin/activate
RUN_ID=$(date +%Y%m%d-%H%M%S)

python experiments/run_appworld.py \
  --config configs/arms/server-jev-active.yaml \
  --mode classifier_route \
  --task-id 099174d_1 \
  --output-dir "runs/jev-active-$RUN_ID" \
  --experiment-name "jev-active-099174d_1-$RUN_ID"
```

检查：

```bash
cat "runs/jev-active-$RUN_ID"/routing-decisions-*.jsonl
cat "runs/jev-active-$RUN_ID"/appworld-classifier_route-summary.json
```

验收以下结果：

- 高 `P(SLM)` 进入 `GUARDED_SLM`；
- 低分、超时、非法响应或输入预检失败进入 `jev_unavailable_large` 或 LLM 路由；
- SLM 守卫失败后调用 LLM fallback；
- `framework_model_calls_by_target` 记录实际模型派发；
- JEV 请求、耗时、输入长度和错误写入 JSONL。

## 12. 第二阶段完成条件

保存以下信息后，才进入第三阶段批量对照：

```text
SLURM_JOB_ID=
SERVICE_NODE=
JEV_MODEL_ID=
JEV_REVISION=
JEV_ENDPOINT=
TOKENIZER_REVISION=
LLM_ENDPOINT=
SLM_ENDPOINT=
SHADOW_RUN_DIR=
ACTIVE_RUN_DIR=
```

如果任一步失败，保留完整日志和新输出目录，修复后重新运行单任务；不要覆盖旧输出，也不要直接启动全量批跑。
