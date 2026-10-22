# Jev路由模型首轮选型

日期：2026-10-21，更新：2026-10-22。状态：完成公开资料与权重元数据核对；本地JEV客户端和路由接线已实现并通过假服务测试。尚未下载权重或在A800上验证真实服务。

## 1. 首轮选择

先接入 **TypeCastLM-Qwen3.5-3.8B**；第二款对照选择 **Docket-3B**。两者完整权重均严格小于4B且接近4B，均标注Apache-2.0。这是依据规模和接入条件作出的工程选择，不是已经证实的Span路由效果排名。先完成一款，避免同时引入多套部署依赖。

参数量来自2026-10-21 Hugging Face模型API的 `safetensors.total`，不是仓库昵称或LoRA适配器大小。

| 定位 | 模型 | 完整参数量 | 固定权重revision |
|---|---|---:|---|
| 首个接入 | [mihailgribov/typecastlm-qwen3.5-3.8b](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b) | 3,759,594,944，约3.760B | `8a0ec192f90b374bad42cfc0adbbe533a0dc1686` |
| 接近4B的对照 | [shahjenil/Docket-3B](https://huggingface.co/shahjenil/Docket-3B) | 3,659,737,600，约3.660B | `53805ff27b8ec18c11e23619338d873ff3eeef56` |
| 后续不同架构备选 | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 3,123,483,888，约3.123B | `051bcc464b01b9f92942b364d9586b0ef5912432` |
| 后续较小规模对照 | [internlm/Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B) | 2,213,241,664，约2.213B | `8797836c65fc91a2435b1fb6850b5f0aabd75cc3` |

Docket虽然名称是3B，实际接近3.66B，符合当前偏好。d1使用LFM许可证，不将它写成Apache模型；Intern-Decision-2B标注Apache-2.0，规模离4B较远，先作为后备。

API证据：[TypeCastLM元数据](https://huggingface.co/api/models/mihailgribov/typecastlm-qwen3.5-3.8b)、[Docket元数据](https://huggingface.co/api/models/shahjenil/Docket-3B)、[d1元数据](https://huggingface.co/api/models/LiquidAI/d1-3B)、[Intern-Decision元数据](https://huggingface.co/api/models/internlm/Intern-Decision-2B)。部署下载使用固定revision，另行固定推理代码与依赖版本，不能仅固定权重后继续任意升级reader。

## 2. 为什么先选TypeCastLM

以下首轮TypeCastLM选择保留；用户要求按原生接口进一步筛选后，新增的完整候选清单见§6。若先验证小模型路径，优先OpenJev-1.7B，不再把Intern-Decision直接当同路径服务。

- 大小接近4B，完整BF16权重按参数估算约7.52GB（7.00GiB），无需额外加载一个教师模型才能推理。
- 是候选决策模型；使用 `choice` 对LLM/SLM打分，适合当前二选一路由。
- 已有[作者推理代码](https://github.com/mihail-gribov/typecastlm)，直接部署其原生System One服务。
- 权重架构配置已核对为 `Qwen3_5ForSequenceClassification`、39个输出标签；不是直接拿普通聊天completion解析文本。

先使用BF16和作者的choice读出方式，不同时改变精度、校准和模板。实际显存、导入速度、A800算子与SLM共存尚需验证。作者提到的 `flash-linear-attention`、`fla-core` 是加速库，不是额外模型；版本需要和现有容器核对，不能直接升级共享sandbox。

## 3. 直接使用原生API，不做协议转换

按用户2026-10-21要求，项目直连 **`POST /v1/systemone`**。原生请求只含 `model`、`state`、`questions`，客户端读取 `answers.route.probabilities`；不用自定义端点，不编写模型HTTP转换层。

TypeCastLM已有这个接口，使用[上游服务程序](https://github.com/mihail-gribov/typecastlm)部署即可；不把Reader另外包成第二个自有服务。其他候选只有验证同一原生接口后才接入，Intern-Decision的不同路径不能只改URL后声称兼容。

客户端保留必要处理：criteria按LLM、SLM顺序发送，检查响应model和候选概率；本地阈值、超时、审计和失败回退。TypeCastLM的超长state默认可能被截断，正式调用前做保守token预检，无法确认完整输入时不发送。原生API不保证返回revision或接受deadline；权重版本通过部署记录核实，超时由客户端执行，不强加服务器字段。

模型config的262144位置上限不等于原生服务已验证256K输入。Jev只需读取Span投影。原生服务的推理依赖和SLM共存仍待实际测试。

参考：[TypeSafe原生API](https://docs.typesafe.ai/api/system-one)、[TypeCastLM API](https://github.com/mihail-gribov/typecastlm/blob/main/docs/API.md)。

## 4. 接入现有设计

以下是选型后拟采用的部署标识，不代表当前已有可调用服务：

```yaml
runner:
  jev:
    enabled: true
    mode: shadow
    endpoint: http://127.0.0.1:8900
    model_id: typecastlm-qwen3.5-3.8b
    expected_revision: 8a0ec192f90b374bad42cfc0adbbe533a0dc1686
    input_mode: combined  # context | four_features | combined
```

这是配置片段，按[完整设计](superpowers/specs/2026-10-20-jev-router-design.md)合并到完整实验臂；主配置的预算、阈值、M1/M4与开关规则不变。input_mode选择纯上下文、纯四特征或组合；每种均只调用一次原生接口，共用固定角色提示词。执行模型名由models配置自动读取，和这里的Jev model_id分开。模型身份核对实际响应；固定权重revision另由部署记录验证，不能只相信原生API可能忽略的model字段。跨节点endpoint改为实际节点IP，不追加 `/v1`。

保留SLM/Jev同一作业内共享一张A800的方向：SLM先尝试60%–70%预算并保留256K，Jev短输入、串行推理；能否共存以实际峰值为准。Docket对照在另一轮替换模型及其原生部署，不同时常驻多款Jev，也不为选型阶段默认额外占一张A800。

## 5. 下一步验收

1. 固定权重revision，取得对应Reader源码、配置、分词器及许可证；确认依赖版本。先读取代码，不运行网页自动生成的聊天示例。
2. 在独立环境验证一条choice输入，检查输出标签、概率、候选交换的映射及超限拒绝；先用短样例验证接口，不据此宣称路由准确。
3. 直接部署原生System One服务，模型只加载一次；客户端验证模型身份、概率、本地超时、错误返回和本地请求hash，不增加转换层。
4. 与SLM同载测显存、真实Span长度和延迟；未通过不启动正式对照。
5. 当前优先直接跑已有AppWorld任务，比较全大模型、原路由、Jev三种输入模式；用开发任务调整提示词/阈值，再用独立任务正式测试。若后续取得学长的原始前缀/标签包，可补充同前缀离线评价，不作为在线实验前置。任务成功、局部正确性与净节省分别报告；单route Choice不等于原有双Logit门控，候选模型支持多问题也不代表两者语义自动相同。

完整集群操作见[CLUSTER_RUN.md](CLUSTER_RUN.md)。当前仓库已包含本地JEV路由接入和 `configs/arms/jev-active.yaml`、`configs/arms/jev-shadow.yaml`；真实模型服务仍需按本文件固定revision后部署验证。

## 6. 按原生 /v1/systemone 重新筛选

本轮硬条件是：完整模型小于4B，上游已经提供 `POST /v1/systemone`，请求支持model/state/questions，choice答案支持候选概率。这里“兼容”依据作者模型卡/服务代码文档，不是本项目的在线实测；不把同名路径当成所有错误码、截断、校准、usage字段均相同的保证。

| 定位 | 模型 | 参数量 | 上游现成服务与核对依据 |
|---|---|---:|---|
| 小模型优先验证 | [alanhuangya/OpenJev-1.7B](https://huggingface.co/alanhuangya/OpenJev-1.7B) | 1,720,574,976，约1.721B | `wev serve`；模型卡明确为System One兼容请求/响应，支持choice/noul/score |
| 极小开销对照 | [AnkitAI/TinyJev-0.6B](https://huggingface.co/AnkitAI/TinyJev-0.6B) | 596,049,920，约0.596B | [tinyjev项目](https://github.com/ankit-aglawe/tinyjev)提供 `tinyjev serve TinyJev-0.6B`，默认8077的System One端点 |
| 接近4B主候选 | [TypeCastLM](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b) | 3,759,594,944，约3.760B | 上游 `typecastlm-serve`，API文档明确System One接口；见§3 |
| 接近4B对照 | [Docket-3B](https://huggingface.co/shahjenil/Docket-3B) | 3,659,737,600，约3.660B | [docket-serve API](https://github.com/shah-jenil/docket-serve#api)明确提供System One请求/响应 |
| 低优先级备选 | [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne) | 752,674,883，约0.753B | 自带 `openthai_systemone.server:app`，模型卡给出System One HTTP命令；偏泰语/英语训练定位 |
| 兼容但部署更复杂 | [StandardThinking/StandardOne-3B](https://huggingface.co/StandardThinking/StandardOne-3B) | 3,849,090,048，约3.849B | 模型卡明确System One，但上游方案由SGLang引擎+jev-adapter两进程组成，当前不优先采用 |

模型许可证：OpenJev、TypeCastLM、Docket、OpenThai和StandardOne标注Apache-2.0；AnkitAI/TinyJev-0.6B权重元数据标注MIT，不能将所有候选统一写成Apache。参数量来自对应HF元数据，不是仅凭型号估计。TinyJev-0.6B是AnkitAI项目，和此前提到的lostargon/Tiny-Jev不是同一个仓库，服务命令不能混用。

新核对的固定revision：

| 模型 | 权重revision |
|---|---|
| OpenJev-1.7B | `7418ae8b6e3b6a6507a8a58729ae491cd1c732d9` |
| AnkitAI/TinyJev-0.6B | `c559c2f7ea95069f92af8b999f512345bbb87d22` |
| OpenThai-SystemOne | `dc67b0295ab7ac94b05d27187150d9a4b5784554` |
| StandardOne-3B | `70023495760b0cefecda7395377377f4d85a64fb` |

对应元数据：[OpenJev](https://huggingface.co/api/models/alanhuangya/OpenJev-1.7B)、[TinyJev](https://huggingface.co/api/models/AnkitAI/TinyJev-0.6B)、[OpenThai](https://huggingface.co/api/models/iapp/OpenThai-SystemOne)、[StandardOne](https://huggingface.co/api/models/StandardThinking/StandardOne-3B)。

### 建议实验顺序

先跑 **OpenJev-1.7B + TypeCastLM-3.8B** 两款：前者作为较小决策模型，后者满足接近4B的偏好。需要极小开销基线再加TinyJev-0.6B；Docket作为后续接近4B对照。各轮分别加载，不要求多款Jev同时驻留GPU。不使用作者不同数据集上的分数替代本项目效果结论。

首个小模型的上游部署入口是：

```bash
# 这是上游安装/启动形式，需在独立服务环境和获配计算资源内验证。
# 正式部署须先固定包版本/源码revision，并使用固定权重快照；不能在共享sandbox里直接升级依赖。
pip install 'wev-ai[serve]'
wev serve --model alanhuangya/OpenJev-1.7B --port 8900
```

命令来自作者模型卡，尚未在本集群执行。默认监听地址、实际响应model名和设备设置需用固定版本的 `wev serve --help` 核对，不编造其参数。若只绑定127.0.0.1，其他节点与经manage的隧道不能直接访问，按上游支持的配置处理。

客户端保持统一：origin形式的endpoint + `/v1/systemone`，原生 `criteria`按LLM、SLM顺序发送，读 `answers.route.probabilities`；更换实际服务model_id，不在本项目新建转换层。仍验证超时、输入完整性、模型身份、候选顺序、usage缺失情况和服务错误码。

此前的Intern-Decision已查到主要HTTP路径为 `/v1/decisions`，不列为本轮无需转换的候选；LiquidAI/d1目前只核实了原生Python `system_one`，没有据此认定它已提供目标HTTP端点。后续如上游提供并验证同协议服务，再补进候选清单。
