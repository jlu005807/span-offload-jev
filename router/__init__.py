"""TraceOffload 离线路由器：116 维特征 -> 训练 -> 验证集选阈值 -> 导出 artifact。

数据流
------
  runs/{训练集,验证集}/<task>/original/snapshots/k####/prefix_trace.json
      | features.py         两个前缀标量 + V2_goal_recent 文本
      | actions.py          最近 10 个动作签名（C_bare_w10）
      | embed.py            Ollama qwen3-embedding:0.6b -> 1024 维 x 2
      | train.py            [2 标量] + PCA50(v2) + PCA64(actions) = 116 维，两个 logit
      | select_threshold.py 验证集上最大化 sum(净节省) -> tau_gate / tau_net
      | export.py           写 artifact（PCA/标准化系数 + 两个 logit + tau）

在线侧 traceoffload/components/l2_add4_predictor.py 读同一份 artifact，
用同一套公式重建 116 维，因此离线与在线逐位一致。
"""

__all__ = ["constants", "signature", "actions", "features", "embed", "labels", "stats"]
