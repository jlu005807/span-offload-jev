"""离线预测器的冻结常量。

只保留 116 维配方真正读到的四个签名参数 + token 估算比例。这些值在训练前冻结，
改动会让线上的动作文本与离线不一致（静默失准），所以单独成文件、显式登记。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConstants:
    #: 一次动作里最多保留多少个 kwargs（按键名排序后截断）。
    signature_max_kwargs: int = 12
    #: 单个 kwarg 值的最大字符数。
    signature_value_truncate: int = 120
    #: 字面量容器最多保留几个元素。
    signature_list_cap: int = 8
    #: 非字面量参数的渲染策略：named（保留变量名）或 opaque。
    signature_var_policy: str = "named"
    #: est_prompt_tokens = total_chars / token_chars_ratio（与线上 M1 同一口径）。
    token_chars_ratio: float = 4.0


DEFAULT = FeatureConstants()
