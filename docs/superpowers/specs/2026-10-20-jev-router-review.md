# Jev 可选路由设计评审与修订记录

日期：2026-10-20。

评审对象：[可选 Jev 决策模型路由设计](2026-10-20-jev-router-design.md)。

## 1. 评审方式与结论

先进行设计自审，再由独立评审代理对照当前代码只读检查，主代理修订后由同一评审者复核。评审没有启动模型或改变业务代码。

最终结论：**未发现剩余 Critical 或 Important 设计问题，可进入实现。** 这不表示新接口、真实模型adapter、完整网络超时或在线实验效果已经验证。

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
