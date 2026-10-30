# Jev fixed Span implementation plan

Goal: 用同一前缀构建并比较三种 Jev 输入，不引入标签推断。

Architecture: experiments/jev_fixed_span_data.py 只读抽取、采样和完整性检查；experiments/jev_fixed_span.py 执行与统计，环境变量入口沿用 deploy/jev 风格。复用既有输入构建器及客户端。

Tech stack: Python 3.11+, sqlite3, pydantic, PyYAML, httpx, 已有 OpenJev tokenizer preflight。

- [x] 写数据关联、完整输入重建、三模式及统计验收测试，确认新功能缺失时失败。
- [x] 实现数据导出、严格原输入复现校验与哈希清单。
- [x] 实现实验执行和配对/全样本汇总，mock 服务覆盖失败路径。
- [x] 运行针对性回归与本地端到端验收，自评并修正问题。
- [x] 上传代码，在服务器实际历史数据上离线导出和验收，不请求模型。
- [x] 验收通过后提交有时限的 Jev 单独服务作业；真实实验不需要 LLM/SLM 服务。
- [ ] 保存实验结果、限制及复现说明。

在当前授权分支顺序实施，不自动提交或推送。
