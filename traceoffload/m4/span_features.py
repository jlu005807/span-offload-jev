"""M4 检索用的 52 维 span 表示 —— **变换与 ``L2V4Predictor`` 共用一份 artifact**。

为什么单独一个模块
------------------
这个变换有两个消费者，而且它们**必须看到同一个向量**：

* ``L2V4Predictor`` —— 决定这一步卸不卸（M2 的门1）
* M4 动作守卫 —— 决定这一步拦不拦（M3 的 verifier）

两边一旦漂移，M4 嘴里的「相似场景」和预测器眼里的「相似场景」就不是一回事，
**而且不会报错** —— 这一类错误只能靠钉住同一个 artifact 来防。

⚠️ 52 维里混了 2 个**非嵌入的标量**（``prefix_n_A`` / ``log_est_prompt_tokens``），
而且整体过了 ``(raw − norm_mu) / norm_sd`` 标准化。所以它**不是「PCA 后的嵌入」**，
是「L2_v4 的输入表示」。当嵌入用会不对。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, Sequence

import numpy as np

from ..contracts import ToolExecutionStatus

__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "action_signature",
    "derive_span_features",
    "failed_action_signatures",
    "guard_vector",
    "guard_vector_dimension",
]


def failed_action_signatures(
    response: Any,
    tool_results: Iterable[Any],
) -> frozenset[str]:
    """从「模型响应 + 工具执行结果」里抽出**失败**动作的签名。

    参数在 ``response.tool_calls`` 上，成败在 ``tool_results`` 上 ——
    **两张表都没有对方的字段**（``ToolExecutionResult`` 里根本没有 ``arguments``）。
    唯一能对齐两者的是 ``tool_call_id``。

    三条语义，各自都有理由：

    * **响应里有、结果里没有 ⇒ 不算失败。** 守卫拦下的动作正是这种：它没跑，
      所以没有成败。当成失败会让「被拦下的」回写进库，下次拦得更狠 —— 自激。
    * **结果里有、响应里没有 ⇒ 丢弃。** 对不上号，宁可漏记不可记错。
    * **``status`` 成功但 ``returncode`` 非零 ⇒ 仍算失败** —— 与
      ``m4_memory.compute_local_success`` 同口径，两处判定必须一致。
    """

    by_call_id: dict[str, Any] = {}
    for result in tool_results:
        call_id = getattr(result, "tool_call_id", None)
        if call_id is not None:
            by_call_id[str(call_id)] = result

    failed: set[str] = set()
    for call in getattr(response, "tool_calls", ()) or ():
        call_id = getattr(call, "tool_call_id", None)
        if call_id is None:
            continue
        result = by_call_id.get(str(call_id))
        if result is None or not _tool_result_failed(result):
            continue
        failed.add(
            action_signature(
                str(getattr(call, "tool_name", "")),
                getattr(call, "arguments", None) or {},
            )
        )
    return frozenset(failed)


def _tool_result_failed(result: Any) -> bool:
    """与 ``compute_local_success`` 逐条同义：error_type / status / returncode。"""

    if getattr(result, "error_type", None):
        return True
    if getattr(result, "status", None) != ToolExecutionStatus.SUCCESS:
        return True
    returncode = getattr(result, "returncode", None)
    return returncode is not None and returncode != 0


def action_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """``(工具名, 参数)`` 的**规范化**签名 —— 键序无关、可哈希、可跨进程比。

    为什么不能直接用参数字典当钥匙：

    * 字典不可哈希，装不进 ``frozenset``；
    * **键序不同的两个等值字典必须签出同一个名字** —— 否则同一个动作会因为
      ``json.dumps`` 的键序差异被当成两个，命中率凭空掉一半。

    为什么把工具名也放进去：``(工具, 参数)`` 是一个整体，
    同一份参数交给不同工具是两回事。

    形状取 ``[工具名, 参数]`` 这个二元列表而不是 ``{"tool":…, "args":…}`` ——
    后者的键会和用户参数里的同名键撞车。
    """

    return json.dumps(
        [str(tool_name), arguments],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

#: 与线上 L2_v4 用的是同一个文件。改这里 = 换向量空间 ⇒ **M4 库里已存的 52 维全部作废**。
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "router" / "artifact" / "l2_new_v4_mlp64x32.json"
)


@lru_cache(maxsize=4)
def _params(path: str) -> tuple[Any, ...]:
    """读一次 artifact 并冻结成数组。**只读 `preprocessing`，不碰模型权重。**"""

    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    pre = spec["preprocessing"]
    return (
        tuple(str(name) for name in spec["prefix_features"]),
        np.asarray(pre["emb_mu"], dtype=np.float64),
        np.asarray(pre["emb_sd"], dtype=np.float64),
        np.asarray(pre["pca_mean"], dtype=np.float64),
        np.asarray(pre["pca_components"], dtype=np.float64),
        np.asarray(pre["norm_mu"], dtype=np.float64),
        np.asarray(pre["norm_sd"], dtype=np.float64),
    )


def derive_span_features(
    embedding: Sequence[float],
    prefix_values: Sequence[float],
    *,
    artifact_path: str | Path | None = None,
) -> tuple[float, ...]:
    """``[前缀标量...] ⊕ PCA(嵌入)`` → 标准化后的 52 维。

    变换顺序**逐字照抄** ``L2V4Predictor._project`` + ``predict``：
    先按 ``emb_mu/emb_sd`` 标准化嵌入，再减 ``pca_mean`` 后乘成分，
    拼上前缀标量，最后整体减 ``norm_mu`` 除 ``norm_sd``。
    """

    names, emb_mu, emb_sd, pca_mean, pca_components, norm_mu, norm_sd = _params(
        str(artifact_path or DEFAULT_ARTIFACT_PATH)
    )
    values = np.asarray(prefix_values, dtype=np.float64)
    if values.shape != (len(names),):
        raise ValueError(
            f"prefix_values 应为 {len(names)} 维（{list(names)}），收到 {values.shape}"
        )
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != emb_mu.shape:
        raise ValueError(f"embedding 应为 {emb_mu.shape[0]} 维，收到 {vector.shape}")

    centered = (vector - emb_mu) / emb_sd
    projected = (centered[None, :] - pca_mean[None, :]) @ pca_components.T
    raw = np.concatenate([values, projected[0]])
    return tuple(float(value) for value in (raw - norm_mu) / norm_sd)


def guard_vector(
    embedding: Sequence[float],
    online_features: Mapping[str, Any],
    *,
    artifact_path: str | Path | None = None,
) -> tuple[float, ...] | None:
    """在线入口：从 1024 维嵌入 + M1 的在线特征派生守卫用的 52 维。

    与 :func:`derive_span_features` 的关系：这个只是**取值适配器** ——
    按 artifact 声明的 ``prefix_features`` 顺序从 ``online_features`` 里取列，
    再交给底层变换。变换本身只有一份实现。

    **缺列 / 值为 None ⇒ 返回 ``None``，不抛异常。** 在线特征由 M1 的
    builder 填；它换名字、或这一版没填，都会让这里取不到值。那时守卫应当
    **拿不到向量而放行**（fail-open），而不是让决策路径炸掉 ——
    这个库是历史证据，不是正确性依赖。

    ⚠️ ``None`` **不是 0**：不能把缺列填 0 再投影，那会得到一个
    「看起来正常但语义错误」的向量，而且不报错。
    """

    names, *_ = _params(str(artifact_path or DEFAULT_ARTIFACT_PATH))
    values: list[float] = []
    for name in names:
        value = online_features.get(name)
        if value is None:
            return None
        values.append(float(value))
    return derive_span_features(embedding, values, artifact_path=artifact_path)


def guard_vector_dimension(*, artifact_path: str | Path | None = None) -> int:
    """守卫向量的维度（= artifact 的 ``input_dim``，当前 52）。

    **这个数只能有一个来源。** 类中心的维度、store 的维度闸、派生出来的向量 ——
    三者只要有一个没跟着 artifact 换，就会出现维度不符或误读，
    而 fail-open 会把「查不到」伪装成「没证据」，日志里看不出来。
    """

    names, _, _, _, pca_components, _, _ = _params(
        str(artifact_path or DEFAULT_ARTIFACT_PATH)
    )
    # 从**实际参与运算的数组**推，而不是读 artifact 声明的 `input_dim` ——
    # 声明值有可能和数组对不上，那时我们要的是「真的会算出多长」。
    return len(names) + len(pca_components)
