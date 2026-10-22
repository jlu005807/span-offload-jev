# Jev 可选路由设计评审与修订记录

日期：2026-10-20。

评审对象：[可选 Jev 决策模型路由设计](2026-10-20-jev-router-design.md)。

## 1. 评审方式与结论

先进行设计自审，再由独立评审代理对照当前代码只读检查，主代理修订后由同一评审者复核。评审没有启动模型或改变业务代码。

最新接口决定（2026-10-21）：按用户要求直连 `/v1/systemone`，取消自定义HTTP协议和转换服务。§2–5保留历史评审背景；其中options数组、服务端revision/deadline等要求已被§6取代。

最新输入与实验修订（2026-10-21）：按后续讨论增加context/four_features/combined三种输入、统一角色提示词和执行模型信息自动读取；直接使用AppWorld任务在线对照，Span标签包不作为前置。当前有效约定以设计正文及本记录§7为准；历史“两种输入”不再作为实施范围。本轮仅主代理自审，没有新增独立代理评审或部署验收。

第二轮结论：**方向可行，发现并修订了开关恢复、选项顺序和补跑覆盖问题；同时收敛首版范围。** 新接口、真实模型adapter、完整网络超时和在线实验效果仍待实现验证。第二轮细节见§5；§2–4保留首轮评审过程，不作为已被替换设计的实施要求。

评审覆盖默认关闭兼容性、配置互斥、原始文本传递、M1/M4初始化与特征依赖、shadow/active语义、HTTP协议与deadline、严格核心契约、失败任务日志、离线回放和报告分母。

## 2. 独立评审发现与修订

### R1. 冻结时间与DNS可能突破总超时预算

严重性：Important，已修订。

AppWorld/freezegun会冻结普通事件循环使用的时钟。异步请求即使被取消，单次循环关闭也可能等待默认executor中的DNS解析线程。

修订：Jev专属 `asyncio.Runner(loop_factory=...)`，实例loop.time使用项目real_time；UTC期限使用real_utcnow。首版endpoint只接受IP字面量，HTTP trust_env=False，不跟随重定向。明确网络预算不是操作系统硬实时保证，不用普通read_timeout代替总时限。

依据：`traceoffload/clock.py`、当前freezegun与asyncio实现。落点：设计§5.4、§9、§14。

### R2. 路由预算遗漏M3的更紧期限

严重性：Important，已修订。

只看caller剩余时间会遗漏 `episode开始时间 + route.timeout_ms`，M1/embedding可能已经消耗完后者。

修订：按Coordinator实际公式计算两者最小值，扣除执行保留预算；shadow在旧计划计算完成后重新读真实时间。若已耗尽则不请求Jev，不延长原deadline。

依据：`coordinator.py::orchestrate_episode`、`m2/routers.py::_ModeRouter._plan`。落点：设计§4.1、§9、§14。

### R3. 移除add4会连带移除M4配置约束

严重性：Important，已修订。

原add4加载块承担m1_online和span-embed-v2校验。Jev active关闭add4后，若不独立保留，M4可能没有当前52维向量，或向量文本配方错位，最终静默放行。

修订：Jev+M4开启时独立要求同步在线嵌入和V2配方；Profiler数值列取add4、Jev四特征、M4消费者的并集；M4关闭不强制嵌入。

依据：`runner.py::__init__/_make_profiler/_observe_m4_profile/_m4_action_guard_lookup`。落点：设计§4.2、§5.4、§14。

### R4. 只靠SpanRecord不能完整重建路由输入

严重性：Important，已修订。

原SpanRecord没有完整InvocationContext、任务目标和初始化后的模板消息跳过数，三元组仅能定位，不能保证重建。

修订：新增每进程版本化provenance旁路，保存目标来源、实际skip、模板与builder版本、窗口和执行模型说明；每条决策绑定provenance revision，回放必须核对输入hash。缺数据明确不可回放，不用当前数据偷偷补齐。

依据：`contracts/records.py::SpanRecord`、`coordinator.py::_build_span`、`m1/span_embedding.py::build_v2_text`、`run_appworld.py`的初始化接线。落点：设计§10、§14。

### R5. 失败任务可能丢失已发生的Jev统计

严重性：Important，已修订。

run_task异常路径直接返回_failure_row，跳过正常任务末尾的汇总。仅在finally清缓存并不够，会丢掉故障期间的请求和审计失败。

修订：成功、失败统一形成row并经过Jev finalizer，先写入真实统计与完整性状态，再清理上下文。未调用Jev的早期初始化失败可保持字段缺失。

依据：`experiments/run_appworld.py::run_task`。落点：设计§4.1、§10、§14。

### R6. M4关闭时强制绕过Router的Span不可见

严重性：Important，已修订。

原span_observer只在M4开启时注册；恢复/循环/fallback又绕过M2，因此新审计不能仅在Jev Router里写。

修订：组合span observer独立于M4，按完整三元组幂等记录明确旁路的bypassed状态；审计写失败和未知缺行不得被猜为绕过。pending未完成Span不提前判缺行。

依据：`run_appworld.py::_build_coordinator`、`coordinator.py::_observe_span`。落点：设计§4.1、§14。

### 其他修订

- 纠正目标字段路径为 `context.workflow_context.workflow_goal.goal_text`。
- 明确认证为Authorization Bearer，密钥仅从环境变量读取。
- 新文本构建器统一注册，防止被嵌入初始化重新赋值覆盖。
- four_features_v1固定窗口10，不以同一个模板ID暗改输入。
- shadow只运行一次旧Router，不能执行两份计划，也不能与原强制DIRECT shadow混用。
- 不把新模型P(SLM)冒充旧gate/net概率；详细审计旁路，保留核心契约兼容。

## 3. 本地验证证据

本轮新增业务代码：无。

设计可行性探针：`runs/setup/check_jev_deadline.py`。在现有WSL Linux环境中执行，结果：

```text
default loop clock frozen: True
real-clock loop timeout: OK (0.050s)
```

它只验证冻结环境下专属时钟的超时机制，不等于真实HTTP客户端已经实现或通过测试。完整慢速分块、连接取消、关闭耗时和代理隔离测试列入设计验收。

文档检查：5个JSON/YAML示例均可解析；设计内已有相对文档链接均存在。没有据此重新宣称旧373项测试覆盖新功能。

## 4. 实现前后的边界

下一阶段可在没有GPU和服务器时实现开关、假服务、Router接线和回归测试。真实模型adapter与在线效果仍需模型revision、部署环境、服务地址，以及学长提供的Span文本、标签和划分。

设计通过后仍需在实现中逐项验证：关闭零新请求；active无add4可运行；M4独立；shadow旧计划保持；失败任务统计完整；远程异常保守走LLM；实际卸载率不受观察意向污染。

## 5. 第二轮：问题与最小化范围

本轮按用户“评审但不要过度设计”的要求，再次对照源码并进行独立只读评审。没有修改业务代码或部署服务。

### S1. 总开关不能单独恢复旧路径

级别：需要修正，已修订设计§5。

原方案要求active时把 `l2_add4.enabled` 设为false，之后只关闭Jev，会触发 `traceoffload/runner.py:1068` 的add4必需校验。

最小修正：保留原add4配置；初始化先解析Jev，active跳过旧预测器加载、校验和计算，off/shadow才走原add4加载分支。配置切换测试必须使用同一份旧配置，只修改Jev开关。恢复基线仍要求原权重和嵌入服务可用。

### S2. 固定选项顺序没有写进数据结构

级别：需要修正，已修订设计§8。

原请求用JSON对象表达选项，却要求LLM、SLM固定顺序。适配器若按对象遍历编译候选，重序列化可能改变顺序，影响输入或概率对应关系。

最小修正：使用有序数组 `[{id: LLM, text: ...}, {id: SLM, text: ...}]`，校验ID与顺序；响应概率仍按名称返回。接口测试必须检查模型原生选项与概率的映射。

### S3. 旧补跑逻辑可能覆盖Jev失败

级别：正式批跑前需要修正，已写入设计§10、§13、§14。

`experiments/run_round.py:77` 的 `_missing_ids` 和 `experiments/repair.py:52` 会补跑add4缺向量任务；Jev shadow仍有add4。若同一次还出现Jev超时，原目录重跑会经 `run_appworld.py::_merge_summary_rows` 按任务覆盖结果，让该故障从最终summary消失。原文只有“不覆盖故障”的原则，未明确拦住这条已有调用路径。

最小修正：第一阶段用单任务、新目录；启用Jev的批跑臂禁用原目录自动补跑，列出缺失/失败，由新目录显式重跑。旧臂不变，不新建重试框架。

### S4. 输入回放和工程拆分超过首版需要

级别：范围简化，已修订设计§1、§4、§10–14。

此前的provenance多文件日志、初始化版本、历史builder重建以及全Span旁路审计，需要新增状态管理，才能完成本来很简单的“查看这次给Jev什么输入”。

最小修正：

- 同一份决策JSONL直接保存实际请求输入、响应、阈值和耗时；删除provenance重建链。
- 只记录进入Jev决策的调用；强制恢复等绕过不进入配对分母。保留现有SpanRecord用于额外轨迹分析。
- 八个新模块合并为五个职责模块；测试按行为组织，不固定九套文件。
- 保留一个必需POST接口；ready检查可选。多模型靠更换配置，不做插件注册平台。
- 离线评估、正式报告和批跑保护分阶段做；GPU成本对账、吞吐队列和全部轨迹回放延后。

首轮R4指出的“SpanRecord不足以重建输入”事实仍成立，本轮通过直接保存输入解决；首轮R6针对“所有Span都有新审计”的要求，现已主动撤销该额外要求。R1–R3及R5的真实时钟、M3期限、M4独立与失败统计约束继续保留，这些是当前代码的实际接线要求。

### 其他小修订

- 文档YAML明确为片段；实际arm从完整旧配置复制。`_prepare_runner_config`只做一层默认合并，没有旧实验臂继承功能。
- 起始提示改为判断当前步骤是否适合卸载；没有成本数据时不要求模型猜净节省，也不预设LLM一定单次更贵。
- 请求失败率、质量和耗时可以先评估；缺少费用口径不阻塞路由开发，费用未知写null。

### 收敛后的实施范围

默认关闭开关、shadow/active、两种输入模板、可注入HTTP客户端、原受控执行计划、M4独立、受真实时钟约束的请求预算、成功/失败统一统计和单份决策JSONL。先接一款真实模型，再用相同接口替换模型比较。

第二轮修订后，独立评审复核未发现必须再修的实质问题。5个JSON/YAML示例解析通过，并检查了选项数组顺序、active保留旧配置及相对文档链接；`git diff --check`通过。本轮只改两份文档，没有运行模型或把旧业务测试结果当作新功能验证。


## 6. 用户确认：原生System One直连

2026-10-21，用户明确要求不进行接口转换。实际官方路径为 `/v1/systemone`，用户消息中的systemon按该拼写纠正。同步设计、集群指南和模型选型，业务代码仍未实现。

- 取消自定义路由HTTP端点、wire schema和模型HTTP转换器；客户端直接发送model/state/questions，读取answers.route.probabilities。
- criteria采用原生对象，保留LLM、SLM插入顺序。首轮S2指出的候选顺序风险继续通过原始JSON顺序测试解决，不再要求自定义options数组。
- 身份/模板hash/请求ID/deadline留在客户端审计；revision由部署固定和核对，不要求原生响应返回。缺少证据明确标记未核实。
- 不再要求服务端拒绝本地过期请求或承诺取消GPU计算；本地真实时钟总预算仍保留，迟到结果不卸载。
- 不假定原生服务拒绝超限：TypeCastLM有截断行为，客户端在发送前预检安全长度，无法确认完整输入则不发送。原生扩展字段允许存在，缺失服务计时记null。
- 保留默认关闭、shadow/active、原SLM守卫、M4独立和任务失败统计。当前修改仅为设计一致性修订，不是新的实现评审或部署通过证明。

## 7. 三种输入、统一提示词与在线实验修订

本轮依据用户后续确认更新设计，不实现业务代码、不下载模型。沿用原生接口和既有执行语义，修改点与自审结论如下。

| 检查项 | 风险与最小修订 |
|---|---|
| 两类模式容易混淆 | mode仍是shadow/active；新增input_mode=context/four_features/combined，内部映射三个版本模板，不同时暴露input_template |
| combined变成两次调用 | 明确每个被路由Span最多一次HTTP尝试、一个route问题；没有分开打分、投票或二次重试 |
| Jev和执行模型身份混用 | 顶层model取runner.jev.model_id；state.model_info从models.large/small.model_id自动读取；不手工重复维护型号、不编造能力画像 |
| 角色称号掩盖判断目标 | 三模式共用固定route_v1角色、信息充分性/难度标准和LLM/SLM选项；只判断下一次调用，不要求生成代码或猜精确净节省 |
| 四特征被误当四个数 | 明确是两标量+两原始文本，不做embedding/PCA，也不把116维向量交给Jev；严格保留旧文本配方 |
| 组合输入冗余或丢观察 | 只按源Span/消息对应关系去除确定重复，不做语义摘要；features不变，工具真实返回与用户更正不能因名称重复而删除 |
| 组合超限后偷偷换输入 | 统一按完整state序列化字符预算和原生完整输入token预检；超限只删允许移除的旧context完整单元，仍超限就unavailable，不切换模式 |
| 不同提示词掩盖输入贡献 | 三模式固定同一prompt/criteria、模型、窗口和示例阈值；分别记录长度、超限与错误。单阈值不代表不同分数已校准，各自调优另做开发/验证实验 |
| 等标签包导致在线实验停滞 | 当前先接真服务并跑AppWorld单任务，再五组独立在线比较；取得Span标签后才补充离线评价 |
| 指标名称超出证据 | AppWorld任务成功不自动提供逐Span局部正确标签，守卫通过不等于正确；Jev自身开销、同卡资源影响和失败不隐去，成本未知记null |

同步文件：设计正文、CLUSTER_RUN.md、JEV_MODELS.md。保留原开关兼容、真实时钟预算、M4独立、任务失败统计及禁止原目录覆盖补跑的约束。不新增模板平台、服务转换层、模型投票、经济第二问题或逐Span标注框架。

实施验收增加：三模式请求内容与提示词一致性、自动模型身份、原始四特征一致性、combined不误删观察、超限不改输入模式、至多一次HTTP尝试，以及日志保存input_mode/prompt版本/实际输入长度。文档检查与未来实现测试分开；本轮文档示例检查不能替代服务器模型兼容性或任务效果实验。

本轮文档验证通过：四份文件中的5个YAML块、2个JSON块可解析，1个Python请求示例语法及组装结果正确，13个Bash块通过bash -n；相对文件链接有效。执行示例使用虚构配置和固定提示词，不发送HTTP，并核对顶层Jev型号、执行SLM型号、仅有route一个问题、criteria按LLM/SLM顺序且与提示词章节一致。git diff --check通过。未运行项目业务测试或模型实验，因为本轮没有实现业务功能；既有未实现状态保持明确。
