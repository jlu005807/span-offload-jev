"""`add4` 两阶段预测器：两段嵌入各自的 PCA，喂给**两个** logit，判两次。

## 它算什么

```
e_v2   (1024)  ← M1 用 `SpanEmbeddingTextBuilderV2` 算的那次 embedding
e_act  (1024)  ← 同一时刻用 `SpanActionsTextBuilder` 算的「最近 10 个动作签名」

x = [ prefix_n_A, log_est_prompt_tokens ]      # 2，各自标准化
  ⊕ PCA50((e_v2  − mu2) / sd2)                 # 50
  ⊕ PCA64((e_act − mu3) / sd3)                 # 64
                                               # ⇒ 116 维

p_gate = sigmoid(gate_coef · x + gate_b)
p_net  = sigmoid(net_coef  · x + net_b)

卸载 ⟺ p_gate ≥ τ_g **且** p_net ≥ τ_t          # 与，不是或
```

两段嵌入**都要**算，因为两个模型读的是同一个 116 维向量。

## 与 v4 的三处关键差别

1. **两个 PCA、两个嵌入。** v4 只有一段嵌入、一个 PCA。
2. **没有最后那道联合归一化。** v4 的 `x` 最后过一遍 `(x − norm_mu)/norm_sd`；
   `add4` 的离线配方（`sf_train_four_feat.fit_transform_prep`）**不过**。
   artifact 里用 `joint_normalization: false` 显式声明 —— 抄 v4 的 `norm_*`
   会强加一道归一化，**不报错但每个 p 都会偏**。
3. **判定是与，不是单个阈值。** 所以对外给的是 `Add4Decision`（含两个 p、两个 τ），
   而不是一个标量 —— 让调用方去重新实现那个与，迟早会出现「只比了一个」的版本。

## 与 M4 的关系

**不重算 embedding。** 两段都在 M1 的 `profile_observer` 里算过一次并缓存。
但**缓存是分开的**：M4 守卫读的是 52 维那份（`guard_vector` 的产物），
本预测器读 116 维那份 —— 两者的向量空间不同，共用会静默失准。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ARTIFACT_KIND",
    "Add4Decision",
    "L2Add4Predictor",
    "load_l2_add4_predictor",
]

#: artifact 的 `kind` 标记。**必须**匹配 —— 拿一份 v4 artifact 喂进来时，
#: 字段缺失会报错，但报的会是「少 preprocessing」这种不看根因的消息。
ARTIFACT_KIND = "l2_add4_two_stage"


def _floats(values: Any, *, expected: int | None = None, what: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"add4 artifact: {what} must be a list, got {type(values).__name__}")
    out = tuple(float(value) for value in values)
    if expected is not None and len(out) != expected:
        raise ValueError(f"add4 artifact: {what} has {len(out)} entries, expected {expected}")
    return out


def _matrix(rows: Any, *, what: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f"add4 artifact: {what} must be a non-empty list of rows")
    out = tuple(_floats(row, what=f"{what}[{index}]") for index, row in enumerate(rows))
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError(f"add4 artifact: {what} rows have inconsistent widths")
    return out


def _to_float32(values: Sequence[float]) -> list[float]:
    """把一串数**先降成 float32 再升回 float64**。

    为什么不是画蛇添足：离线那两段嵌入的来源精度**不一样** ——
    `aw3_emb_v2goal.jsonl` 是 float64，而 `emb_C_bare_w10.npz` 存的是 **float32**
    （训练脚本 `V = np.load(...)["v"].astype(np.float64)`，是先有 float32 再升格）。
    在线嵌入客户端返回 float64，直接喂进去就与训练时的向量差一个舍入。

    差别小到不影响任何决策，但它让「在线向量与离线逐位相同」这句话不成立 ——
    而那句话正是 `tests/components/test_l2_add4_golden_features.py` 要钉的东西。
    所以按 artifact 声明的精度还原，而不是靠容差把问题盖过去。
    """
    return [float(np.float32(value)) for value in values]


@dataclass(frozen=True)
class Add4Decision:
    """一次两阶段判定。**两个 p 和两个 τ 都留着** —— 只留 `offload` 就没法审计
    「是门控挡的还是经济挡的」。"""

    p_gate: float
    p_net: float
    tau_gate: float
    tau_net: float
    offload: bool


class L2Add4Predictor:
    """`add4` 前向。构造时把 artifact 全部参数**校验并冻结**，之后只做算术。

    纯 Python，不依赖 sklearn —— 116 维的两个点积只有几百次乘加，微秒级。
    """

    def __init__(self, spec: Mapping[str, Any]) -> None:
        kind = str(spec.get("kind") or "").strip()
        if kind != ARTIFACT_KIND:
            raise ValueError(
                f"add4 artifact: kind={kind!r}，期望 {ARTIFACT_KIND!r}"
                "（这个加载器只认两阶段 add4，喂 v4 的 artifact 进来是配置错）"
            )
        pre = spec.get("preprocessing")
        if not isinstance(pre, Mapping):
            raise ValueError("add4 artifact: missing 'preprocessing'")

        prefix_names = spec.get("prefix_features")
        if not isinstance(prefix_names, (list, tuple)) or not prefix_names:
            raise ValueError("add4 artifact: 'prefix_features' must be a non-empty list")
        self.prefix_features: tuple[str, ...] = tuple(str(name) for name in prefix_names)

        legacy = pre.get("legacy")
        if not isinstance(legacy, Mapping):
            raise ValueError("add4 artifact: preprocessing.legacy is missing")
        self.legacy_mu = _floats(
            legacy.get("mu"), expected=len(self.prefix_features), what="legacy.mu"
        )
        self.legacy_sd = _floats(
            legacy.get("sd"), expected=len(self.prefix_features), what="legacy.sd"
        )
        if any(scale == 0.0 for scale in self.legacy_sd):
            raise ValueError("add4 artifact: legacy.sd contains 0 (not invertible)")

        # --- 联合归一化：**必须显式为 false** ---------------------------------
        # 缺字段就当 false 是危险的：一份抄了 v4 格式、带 norm_mu/norm_sd 的
        # artifact 会被静默地**不**归一化。所以要显式声明。
        joint = spec.get("joint_normalization")
        if joint is not False:
            raise ValueError(
                "add4 artifact: joint_normalization 必须显式为 false"
                f"（收到 {joint!r}）。add4 的离线配方不做最后那道 (x−μ)/σ —— "
                "要启用它必须同时给出 116 长度的 norm_mu/norm_sd，而这个加载器还没实现。"
            )

        self.gate = self._branch(pre.get("gate"), what="gate")
        self.net = self._branch(pre.get("net"), what="net")

        declared = spec.get("input_dim")
        if declared is not None and int(declared) != self.input_dim:
            raise ValueError(
                f"add4 artifact: input_dim={declared} 与 "
                f"前缀 {len(self.prefix_features)} + gate {self.gate_pca_nc} "
                f"+ net {self.net_pca_nc} = {self.input_dim} 不一致"
            )

        models = spec.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("add4 artifact: missing 'models'")
        self.gate_model_type, self.gate_coef, self.gate_intercept, self.tau_gate = self._model(
            models.get("gate"), what="models.gate"
        )
        self.net_model_type, self.net_coef, self.net_intercept, self.tau_net = self._model(
            models.get("net"), what="models.net"
        )
        for name, tau in (("models.gate.tau", self.tau_gate), ("models.net.tau", self.tau_net)):
            if not 0.0 <= tau <= 1.0:
                raise ValueError(f"add4 artifact: {name}={tau} 不在 [0,1]")

        self.metrics = dict(spec.get("metrics") or {})
        self.trained_on = str(spec.get("trained_on") or "")
        self.validated_on = str(spec.get("validated_on") or "")

    # -- 构造期校验 ------------------------------------------------------------
    def _branch(self, block: Any, *, what: str) -> dict[str, Any]:
        if not isinstance(block, Mapping):
            raise ValueError(f"add4 artifact: preprocessing.{what} is missing")
        mu = _floats(block.get("emb_mu"), what=f"{what}.emb_mu")
        sd = _floats(block.get("emb_sd"), expected=len(mu), what=f"{what}.emb_sd")
        if any(scale == 0.0 for scale in sd):
            raise ValueError(f"add4 artifact: {what}.emb_sd contains 0 (not invertible)")
        mean = _floats(block.get("pca_mean"), expected=len(mu), what=f"{what}.pca_mean")
        components = _matrix(block.get("pca_components"), what=f"{what}.pca_components")
        if len(components[0]) != len(mu):
            raise ValueError(
                f"add4 artifact: {what}.pca_components width {len(components[0])} "
                f"!= embedding dim {len(mu)}"
            )
        nc = len(components)
        if block.get("pca_nc") is not None and int(block["pca_nc"]) != nc:
            raise ValueError(
                f"add4 artifact: {what}.pca_nc={block['pca_nc']} "
                f"与 pca_components 行数 {nc} 不一致"
            )
        # 训练时那一段嵌入的存储精度。`emb_C_bare_w10.npz` 是 float32，
        # `aw3_emb_v2goal.jsonl` 是 float64 —— 见 `_to_float32` 的说明。
        dtype = str(block.get("embedding_dtype") or "float64").strip().lower()
        if dtype not in {"float32", "float64"}:
            raise ValueError(f"add4 artifact: {what}.embedding_dtype={dtype!r} 不支持")
        return {
            "segment": str(block.get("embedding_segment") or ""),
            "mu": mu,
            "sd": sd,
            "mean": mean,
            "components": components,
            "nc": nc,
            "dtype": dtype,
        }

    def _model(self, block: Any, *, what: str) -> tuple[str, tuple[float, ...], float, float]:
        if not isinstance(block, Mapping):
            raise ValueError(f"add4 artifact: {what} is missing")
        model_type = str(block.get("model_type") or "").strip().lower()
        if model_type != "logistic":
            raise ValueError(f"add4 artifact: {what}.model_type={model_type!r}，只支持 logistic")
        coef = _floats(block.get("coef"), expected=self.input_dim, what=f"{what}.coef")
        intercept = float(block.get("intercept", 0.0))
        tau_raw = block.get("tau")
        if not isinstance(tau_raw, (int, float)) or isinstance(tau_raw, bool):
            raise ValueError(f"add4 artifact: {what}.tau 必须是数字，收到 {tau_raw!r}")
        return model_type, coef, intercept, float(tau_raw)

    # -- 派生属性 --------------------------------------------------------------
    @property
    def gate_pca_nc(self) -> int:
        return int(self.gate["nc"])

    @property
    def net_pca_nc(self) -> int:
        return int(self.net["nc"])

    @property
    def input_dim(self) -> int:
        return len(self.prefix_features) + self.gate_pca_nc + self.net_pca_nc

    @property
    def gate_embedding_dim(self) -> int:
        """第 ③ 段嵌入的维度（1024）。"""
        return len(self.gate["mu"])

    @property
    def net_embedding_dim(self) -> int:
        """第 ④ 段嵌入的维度（1024）。**空动作文本要用这个长度填零向量。**"""
        return len(self.net["mu"])

    @property
    def gate_segment(self) -> str:
        return str(self.gate["segment"])

    @property
    def net_segment(self) -> str:
        return str(self.net["segment"])

    # -- 前向 ------------------------------------------------------------------
    @staticmethod
    def _project(
        embedding: Sequence[float],
        *,
        mu: Sequence[float],
        sd: Sequence[float],
        mean: Sequence[float],
        components: Sequence[Sequence[float]],
    ) -> list[float]:
        if len(embedding) != len(mu):
            raise ValueError(f"add4: embedding dim {len(embedding)} != artifact dim {len(mu)}")
        centered = [(float(value) - m) / s for value, m, s in zip(embedding, mu, sd)]
        out: list[float] = []
        for component in components:
            total = 0.0
            for value, weight, c in zip(centered, component, mean):
                total += (value - c) * weight
            out.append(total)
        return out

    def features_from(
        self,
        *,
        gate_embedding: Sequence[float],
        net_embedding: Sequence[float],
        prefix_values: Sequence[float],
    ) -> tuple[float, ...]:
        """两段嵌入 + 前缀标量 → **116 维**。与离线 `fit_transform_prep` 的 `Xa` 同构。"""
        if len(prefix_values) != len(self.prefix_features):
            raise ValueError(
                f"add4: got {len(prefix_values)} prefix values, "
                f"artifact expects {len(self.prefix_features)}"
            )
        legacy = [
            (float(value) - mu) / sd
            for value, mu, sd in zip(prefix_values, self.legacy_mu, self.legacy_sd)
        ]
        projected = self._project(
            self._as_stored(gate_embedding, self.gate),
            mu=self.gate["mu"], sd=self.gate["sd"],
            mean=self.gate["mean"], components=self.gate["components"],
        ) + self._project(
            self._as_stored(net_embedding, self.net),
            mu=self.net["mu"], sd=self.net["sd"],
            mean=self.net["mean"], components=self.net["components"],
        )
        return tuple(legacy + projected)

    @staticmethod
    def _as_stored(embedding: Sequence[float], branch: Mapping[str, Any]) -> list[float]:
        """按 artifact 声明的精度还原该段嵌入（见 `_to_float32`）。"""
        values = [float(value) for value in embedding]
        return _to_float32(values) if branch["dtype"] == "float32" else values

    def predict_from_span_features(self, features: Sequence[float]) -> tuple[float, float]:
        """从**已经算好的 116 维**前向，返回 `(p_gate, p_net)`。

        存在的意义：让在线缓存只留 116 维（两段 1024 维用完即弃）。

        ⚠️ 输入必须已经是 `features_from` 的产物 —— 少一步标准化会得到一个
        「看起来像向量」的垃圾，而且不报错。
        """
        values = [float(value) for value in features]
        if len(values) != self.input_dim:
            raise ValueError(
                f"add4: expected {self.input_dim} span features, got {len(values)}"
            )
        return (
            self._logit(values, self.gate_coef, self.gate_intercept),
            self._logit(values, self.net_coef, self.net_intercept),
        )

    def decide_from_span_features(
        self,
        features: Sequence[float],
        *,
        tau_gate: float | None = None,
        tau_net: float | None = None,
    ) -> Add4Decision:
        """两阶段判定。**与**语义：任一条不过 ⇒ 不卸载 ⇒ 走大模型。

        τ 允许覆盖（阈值敏感性实验用）；不给就用 artifact 里的。
        """
        gate_tau = self.tau_gate if tau_gate is None else float(tau_gate)
        net_tau = self.tau_net if tau_net is None else float(tau_net)
        p_gate, p_net = self.predict_from_span_features(features)
        return Add4Decision(
            p_gate=p_gate,
            p_net=p_net,
            tau_gate=gate_tau,
            tau_net=net_tau,
            offload=bool(p_gate >= gate_tau and p_net >= net_tau),
        )

    @staticmethod
    def _logit(values: Sequence[float], coef: Sequence[float], intercept: float) -> float:
        total = intercept
        for value, weight in zip(values, coef):
            total += value * weight
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, total))))


def load_l2_add4_predictor(path: str | Path) -> L2Add4Predictor | None:
    """读 add4 artifact。

    **文件不存在返回 None**（与 `load_l2_v4_predictor` 一致，让调用方决定怎么办）；
    **文件存在但格式不对则抛异常** —— 不静默。
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    spec = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError(f"add4 artifact {candidate} is not a JSON object")
    return L2Add4Predictor(spec)
