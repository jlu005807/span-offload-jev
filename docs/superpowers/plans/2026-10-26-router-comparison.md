# Router comparison implementation plan

Goal: 自动完成四组技术验收及 20 任务交错对照。

Architecture: 新增 experiments/compare_routers.py 生成配置、调度独立 worker 和汇总；复用 run_native_task 与 AppWorldPilotRunner。新增部署脚本准备 embedding 依赖并管理自己的 Jev/Ollama 服务，不改 M1–M4。

- [x] 先写配置隔离、交错执行、技术门控和失败继续的测试并观察失败。
- [x] 实现控制器、隔离 worker 和可追溯报告。
- [ ] 准备固定版本 Ollama 与模型，下载不占 GPU，哈希校验。
- [x] 本地及服务器代码测试，冻结 20 任务清单；官方划分由运行入口核对。
- [x] 提交自动服务、单任务验收与 80 格实验作业，设定资源时限及清理。
- [ ] 确认作业实际状态并保存结果说明。

本地验收和独立评审完成；首轮 3 项重要问题已修复并复评通过。评审后再进行服务器复测，服务器先前的 29 项检查不能替代当前版本复测。

评审后服务器作业 560900 已验证 61 项通过。560896 准备 embedding，560902 依赖它成功后自动启动服务并进行真实四组验收。
