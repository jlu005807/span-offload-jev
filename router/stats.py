"""评估用的小工具：池化 AUC、家族均衡 AUC、家族级 bootstrap。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def pooled_auc(y: Sequence[float], p: Sequence[float]) -> float:
    y_arr, p_arr = np.asarray(y, float), np.asarray(p, float)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, p_arr))


def family_equal_auc(y: Sequence[float], p: Sequence[float], family: Sequence[str]) -> float:
    """先在每个家族内算 AUC，再对家族取等权平均。

    池化 AUC 会被大任务带偏；家族均衡是 2 节评估口径。
    """
    y_arr, p_arr, f_arr = np.asarray(y, float), np.asarray(p, float), np.asarray(family)
    aucs = []
    for f in sorted(set(f_arr.tolist())):
        m = f_arr == f
        if len(np.unique(y_arr[m])) < 2:
            continue
        aucs.append(roc_auc_score(y_arr[m], p_arr[m]))
    return float(np.mean(aucs)) if aucs else float("nan")
