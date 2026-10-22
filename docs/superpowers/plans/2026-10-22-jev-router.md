# Jev 路由分阶段实施计划

> 实施方式：按 executing-plans 推进，输入和客户端并行开发，接线与验收顺序执行；每个阶段完成测试及独立评审。

**目标：** 实现默认关闭的Jev M2后端，支持三种输入、原生System One接口、shadow/active、原有受控执行和可追溯日志。

**架构：** 沿用2026-10-20设计及10月21日三输入修订。新代码集中于m2/jev五个模块，Runner准备上下文，M2消费并生成原有RoutePlan，不改变Coordinator/M3协议。

**环境：** Python 3.11+，Pydantic，httpx可选依赖；本地HTTP测试无需GPU。Linux/WSL执行涉及真实时钟的验收。

## 阶段一：本地可测试运行时

- [x] 基线：功能分支保留现有改动；WSL基线和Windows时钟限制已记录。
- [x] 类型/配置与HTTP：完成严格校验、原生单次请求、取消/无重试和完整输入预检；未提供预检时安全返回input_unverified。
- [x] 三输入：完成context、four_features、combined，保留原特征配方、工具结果、few-shot跳过、字符上限和三元组缓存。
- [x] 路由与审计：完成active/shadow、受控SLM、LLM fallback、deadline扣除、JSONL和summary统计。
- [x] Runner接线：完成active绕过旧artifact、关闭无Jev工作、消费者并集、M4独立和profile observer接线。
- [x] AppWorld收尾：成功/异常统一Jev汇总与finally清理，异常任务保留已发生统计。
- [x] 配置/文档：新增jev-shadow/active完整配置、httpx可选依赖和部署说明。
- [x] 验收：JEV测试143项、全仓WSL测试516项通过；完成独立规格评审并修复 token 审计缺口。

测试命令（项目根目录，WSL的现有虚拟环境）：

```bash
.venv-wsl/bin/python -m pytest tests/test_jev_client.py tests/test_jev_input.py tests/test_jev_routing.py tests/test_jev_runner.py -q -o addopts=
.venv-wsl/bin/python -m pytest tests -q -o addopts=
```

## 阶段二：真实原生服务与单任务

- [ ] 固定模型/源码/依赖revision；核对服务真实模板、tokenizer与最大输入，接入经验证的完整输入预检。
- [ ] 获配计算节点验证原生接口、SLM/Jev共享显存及长输入策略。
- [ ] 一个开发任务分别验证shadow和active，确认实际HTTP/派发/守卫/结果记录。

阶段二只在真实服务、可达地址及计算资源就绪时执行，不以假服务测试声称部署验收通过。

## 阶段三：在线比较

- [x] 接report列和Jev实验禁止覆盖补跑的保护；旧实验行为不变。
- [ ] 固定开发清单，分别运行全大模型、原路由、Jev三输入五组，独立状态和输出目录。
- [ ] 正式测试使用不参与提示词/阈值调整的任务；任务成功率、开销及逐Span标签边界遵循设计。

不自动提交或推送既有用户改动，不覆盖服务器已有项目或共享容器。
