"""TraceOffload 数据契约包。

所有跨组件通信对象都定义在这里，按用途分文件：

| 模块 | 内容 |
|---|---|
| base | 类型别名、29 个枚举、ContractModel 基类、通用引用、信封基类 |
| capabilities | 受限能力引用（安全边界核心） |
| prior | AdapterSession 投影的历史执行事实（M1/embedding 旁路输入） |
| requests | Agent 模型请求对象（消息/工具/解码参数） |
| context | workflow 上下文与调用入口（WorkflowContext / InvocationContext） |
| responses | 模型响应对象 |
| profiling | M1 画像（SpanDescriptor / RuntimeProfile） |
| routing | M2 路由（验证方案 / RoutePlan / RouteOption / RoutePolicy） |
| execution | M3 执行（ExecutionRequest / ExecutionResult / 尝试与守卫记录） |
| records | Coordinator 完整记录（SpanRecord / ExecutionSummary） |
| outcomes | M4 反馈（OutcomeEvent / ExperienceProfile / FeedbackSnapshot） |
| links | 主链对象之间的关联校验函数 |

依赖方向单向无环：base → capabilities → context/profiling → routing → execution
→ records，outcomes 独立于 routing 之上；links 依赖全部。
"""

from .base import *  # noqa: F401,F403
from .base import __all__ as _base_exports
from .capabilities import *  # noqa: F401,F403
from .capabilities import __all__ as _capabilities_exports
from .prior import *  # noqa: F401,F403
from .prior import __all__ as _prior_exports
from .context import *  # noqa: F401,F403
from .context import __all__ as _context_exports
from .execution import *  # noqa: F401,F403
from .execution import __all__ as _execution_exports
from .recovery import *  # noqa: F401,F403
from .recovery import __all__ as _recovery_exports
from .links import *  # noqa: F401,F403
from .links import __all__ as _links_exports
from .outcomes import *  # noqa: F401,F403
from .outcomes import __all__ as _outcomes_exports
from .profiling import *  # noqa: F401,F403
from .profiling import __all__ as _profiling_exports
from .records import *  # noqa: F401,F403
from .records import __all__ as _records_exports
from .requests import *  # noqa: F401,F403
from .requests import __all__ as _requests_exports
from .responses import *  # noqa: F401,F403
from .responses import __all__ as _responses_exports
from .routing import *  # noqa: F401,F403
from .routing import __all__ as _routing_exports

# 统一导出清单：等价于原 contracts.py 的 __all__（含修复的补充项）
__all__ = [
    *_base_exports,
    *_capabilities_exports,
    *_prior_exports,
    *_context_exports,
    *_execution_exports,
    *_recovery_exports,
    *_links_exports,
    *_outcomes_exports,
    *_profiling_exports,
    *_records_exports,
    *_requests_exports,
    *_responses_exports,
    *_routing_exports,
]
