"""M4 Memory 的轻量 SQLite 持久化。

只存 M4 所需的最小字段：lineage、class_key、embedding BLOB、labels、
LRU 时间和 full_span 的 JSON 投影。不引入 ORM，不做复杂查询。
"""

from __future__ import annotations

from array import array
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterable


class SQLiteM4MemoryStore:
    """轻量 SQLite 存储；线程安全，供 M4MemoryWorker 后台调用。"""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_dimension: int | None = None,
    ) -> None:
        """``embedding_dimension`` 给定时，`load_records` 默认跳过维度不符的行。

        为什么要在**构造时**定：M4Memory 启动时会自己 `load_records()` 恢复历史，
        那次调用没有「调用方记得传维度」这回事 —— 每次调用都要记得传，
        迟早会漏，而漏了的坏法是**静默读到换空间之前的向量**。
        """

        self.path = str(path)
        self._embedding_dimension = embedding_dimension
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def count_records(self) -> int:
        """库里现在几条。

        T2 用它填 `m4_records_at_start` / `m4_records_at_end`（spec §3.2）。
        **没有这两个数，「seed 没加载上」和「守卫就是不触发」看起来一样** ——
        消融里就分不清「机制没效果」和「机制没跑」。

        ⚠️ 不收 `embedding_dimension` 过滤：这里要的是「库里有几条」，
        不是「有几条能用」。维度不匹配是另一件事，别混进这个数。
        """
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM m4_records"
            ).fetchone()
        return int(row[0]) if row else 0

    def save_pending(self, seed: Any) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO m4_pending(record_id, embedding, verifier_refs)
                VALUES (?, ?, ?)
                """,
                (
                    str(seed.record_id),
                    _encode_embedding(seed.embedding),
                    json.dumps(sorted(seed.verifier_refs), ensure_ascii=True),
                ),
            )
            self._connection.commit()

    def delete_pending(self, record_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM m4_pending WHERE record_id = ?",
                (str(record_id),),
            )
            self._connection.commit()

    def load_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_id, embedding, verifier_refs FROM m4_pending"
            ).fetchall()
        return [
            {
                "record_id": str(row[0]),
                "embedding": _decode_embedding(row[1]),
                "verifier_refs": _decode_string_list(row[2]),
            }
            for row in rows
        ]

    def save_record(self, record: Any) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO m4_records(
                    record_id, workflow_run_id, episode_id, logical_invocation_id,
                    class_key, embedding, full_span_json, local_success, danger_tier,
                    verifier_refs, entered_at, last_access_at, pair_id,
                    switch_direction, forced_class, assignment_status,
                    embedding_latency_ms, embedding_source, template_version,
                    action, decision_reason, policy_version, predictor_p,
                    predictor_threshold, predictor_version, predictor_feature_version,
                    router_features_json, task_success, label_scope,
                    span_input_tokens, span_output_tokens, span_cost, span_latency_ms,
                    suffix_cost, trajectory_cost, position_ratio, trajectory_span_count,
                    step_index, experiment_id, failed_actions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.record_id),
                    str(record.workflow_run_id),
                    str(record.episode_id),
                    str(record.logical_invocation_id),
                    int(record.class_key),
                    _encode_embedding(record.embedding),
                    _serialize_full_span(record.full_span_record),
                    int(bool(record.local_success)),
                    int(record.danger_tier),
                    json.dumps(sorted(record.verifier_refs), ensure_ascii=True),
                    float(record.entered_at),
                    float(record.last_access_at),
                    record.pair_id,
                    record.switch_direction,
                    int(bool(record.forced_class)),
                    str(record.assignment_status),
                    float(getattr(record, "embedding_latency_ms", 0.0)),
                    str(getattr(record, "embedding_source", "unknown")),
                    str(getattr(record, "template_version", "unknown")),
                    str(getattr(record, "action", "unknown")),
                    str(getattr(record, "decision_reason", "unknown")),
                    str(getattr(record, "policy_version", "unknown")),
                    _optional_float(getattr(record, "predictor_p", None)),
                    _optional_float(getattr(record, "predictor_threshold", None)),
                    _optional_str(getattr(record, "predictor_version", None)),
                    _optional_str(getattr(record, "predictor_feature_version", None)),
                    _optional_str(getattr(record, "router_features_json", None)),
                    _optional_int(getattr(record, "task_success", None)),
                    _optional_str(getattr(record, "label_scope", None)),
                    _optional_int(getattr(record, "span_input_tokens", None)),
                    _optional_int(getattr(record, "span_output_tokens", None)),
                    _optional_float(getattr(record, "span_cost", None)),
                    int(getattr(record, "span_latency_ms", 0) or 0),
                    _optional_float(getattr(record, "suffix_cost", None)),
                    _optional_float(getattr(record, "trajectory_cost", None)),
                    _optional_float(getattr(record, "position_ratio", None)),
                    _optional_int(getattr(record, "trajectory_span_count", None)),
                    _optional_int(getattr(record, "step_index", None)),
                    str(getattr(record, "experiment_id", "unknown")),
                    json.dumps(
                        sorted(str(item) for item in record.failed_actions),
                        ensure_ascii=True,
                    ),
                ),
            )
            self._connection.commit()

    def delete_record(self, record_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM m4_records WHERE record_id = ?",
                (str(record_id),),
            )
            self._connection.commit()

    def load_records(self, *, embedding_dimension: int | None = None) -> list[dict[str, Any]]:
        """读出全部记录；``embedding_dimension`` 给定时跳过维度不符的行。

        为什么要这道闸：换 PCA = **换向量空间**，老库里的坐标与新向量之间
        没有任何换算关系。不跳过的话它们会被解码、拿去算 `_cosine_distance` ——
        要么抛异常，要么**静默算出毫无意义的相似度**。后者更糟：守卫会基于
        垃圾证据拒绝动作，而且不报错。

        不传 = 不过滤（既有调用方的行为）。
        """

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT record_id, workflow_run_id, episode_id, logical_invocation_id,
                       class_key, embedding, full_span_json, local_success,
                       danger_tier, verifier_refs, entered_at, last_access_at,
                       pair_id, switch_direction, forced_class, assignment_status,
                       embedding_latency_ms, embedding_source, template_version,
                       action, decision_reason, policy_version, predictor_p,
                       predictor_threshold, predictor_version, predictor_feature_version,
                       router_features_json, task_success, label_scope,
                       span_input_tokens, span_output_tokens, span_cost, span_latency_ms,
                       suffix_cost, trajectory_cost, position_ratio, trajectory_span_count,
                       step_index, experiment_id, failed_actions
                FROM m4_records
                """
            ).fetchall()
        wanted = (
            embedding_dimension
            if embedding_dimension is not None
            else self._embedding_dimension
        )
        records = [
            {
                "record_id": str(row[0]),
                "workflow_run_id": str(row[1]),
                "episode_id": str(row[2]),
                "logical_invocation_id": str(row[3]),
                "class_key": int(row[4]),
                "embedding": _decode_embedding(row[5]),
                "full_span_record": _deserialize_full_span(row[6]),
                "local_success": bool(row[7]),
                "danger_tier": int(row[8]),
                "verifier_refs": _decode_string_list(row[9]),
                "entered_at": float(row[10]),
                "last_access_at": float(row[11]),
                "pair_id": row[12],
                "switch_direction": row[13],
                "forced_class": bool(row[14]),
                "assignment_status": str(row[15]),
                "embedding_latency_ms": float(row[16]),
                "embedding_source": str(row[17]),
                "template_version": str(row[18]),
                "action": str(row[19]),
                "decision_reason": str(row[20]),
                "policy_version": str(row[21]),
                "predictor_p": _optional_float(row[22]),
                "predictor_threshold": _optional_float(row[23]),
                "predictor_version": _optional_str(row[24]),
                "predictor_feature_version": _optional_str(row[25]),
                "router_features_json": _optional_str(row[26]),
                "task_success": _optional_bool(row[27]),
                "label_scope": _optional_str(row[28]),
                "span_input_tokens": _optional_int(row[29]),
                "span_output_tokens": _optional_int(row[30]),
                "span_cost": _optional_float(row[31]),
                "span_latency_ms": int(row[32] or 0),
                "suffix_cost": _optional_float(row[33]),
                "trajectory_cost": _optional_float(row[34]),
                "position_ratio": _optional_float(row[35]),
                "trajectory_span_count": _optional_int(row[36]),
                "step_index": _optional_int(row[37]),
                "experiment_id": str(row[38]),
                "failed_actions": frozenset(_decode_string_list(row[39])),
            }
            for row in rows
        ]
        if wanted is None:
            return records
        return [row for row in records if len(row["embedding"]) == wanted]

    #: 旧库缺失时逐列补齐：列名 -> 列定义。
    #: 「决策 / 结局 / 成本」三组字段用于 M4 的「决策 -> 结局」配对账本；
    #: 旧库里的行补为默认值后，无法追溯的历史决策保持 'unknown' / NULL，
    #: 不会被误读成一个有效决策。
    _ADDED_COLUMNS: dict[str, str] = {
        "failed_actions": "failed_actions TEXT NOT NULL DEFAULT '[]'",
        "embedding_latency_ms": "embedding_latency_ms REAL NOT NULL DEFAULT 0.0",
        "embedding_source": "embedding_source TEXT NOT NULL DEFAULT 'unknown'",
        #: 产出这条向量的**文本配方版本**。与 embedding_source（谁算的）正交：
        #: 同一个 builder 换配方后，老记录必须还能被解释，所以两者都要记。
        "template_version": "template_version TEXT NOT NULL DEFAULT 'unknown'",
        # --- 观测成本（结局到达后回填；线上可实测，不是跨臂的 Δ）---------------
        #: 从**本 Span** 起到轨迹结束的花费。检索「贵不贵」的主信号。
        #: ⚠️ 强烈依赖 Span 位置（越早后缀越长）→ 必须配合 position_ratio 归一化，
        #: 不能直接横向比。
        "suffix_cost": "suffix_cost REAL",
        #: 整条轨迹成本（与 Span 无关，同一 workflow 的每行相同）。
        "trajectory_cost": "trajectory_cost REAL",
        #: 本 Span 在轨迹里的相对位置 k/n ∈ (0,1]，用于 suffix_cost 的归一化。
        "position_ratio": "position_ratio REAL",
        #: 该轨迹的 Span 总数，便于审计位置归一化的分母。
        "trajectory_span_count": "trajectory_span_count INTEGER",
        #: 本 Span 的**绝对步号** k。位置归一化用它当分桶坐标 ——
        #: **不能用 `position_ratio = k/n`**：M2 决策时轨迹还没跑完、n 未知，
        #: 查询点算不出这个比值，而参照点算得出 —— 两边坐标不一致，
        #: 就会出现「查询点和参照点不可比」的静默错误。
        "step_index": "step_index INTEGER",
        #: 本次**运行**的唯一标识（`PilotRunner._resolve_experiment_id`）。
        #: 为什么需要它：`record_id = workflow-{task_id}|invocation-N`
        #: **同一 task 重跑会得到同一个主键** → `INSERT OR REPLACE` 覆盖，
        #: 库永远只有那几条，「用历史经验」这件事根本不会发生（实测
        #: n_neighbors 恒为 0）。检索时靠它排掉本轮自己那些 Span。
        "experiment_id": "experiment_id TEXT NOT NULL DEFAULT 'unknown'",
        "action": "action TEXT NOT NULL DEFAULT 'unknown'",
        "decision_reason": "decision_reason TEXT NOT NULL DEFAULT 'unknown'",
        "policy_version": "policy_version TEXT NOT NULL DEFAULT 'unknown'",
        "predictor_p": "predictor_p REAL",
        "predictor_threshold": "predictor_threshold REAL",
        "predictor_version": "predictor_version TEXT",
        "predictor_feature_version": "predictor_feature_version TEXT",
        "router_features_json": "router_features_json TEXT",
        "task_success": "task_success INTEGER",
        "label_scope": "label_scope TEXT",
        "span_input_tokens": "span_input_tokens INTEGER",
        "span_output_tokens": "span_output_tokens INTEGER",
        "span_cost": "span_cost REAL",
        "span_latency_ms": "span_latency_ms INTEGER NOT NULL DEFAULT 0",
    }

    def _ensure_columns(self) -> None:
        """兼容旧 SQLite 文件：缺少新增列时 ALTER TABLE 补齐。"""

        with self._lock:
            existing = {
                str(row[1])
                for row in self._connection.execute("PRAGMA table_info(m4_records)")
            }
            for column, definition in self._ADDED_COLUMNS.items():
                if column not in existing:
                    self._connection.execute(
                        f"ALTER TABLE m4_records ADD COLUMN {definition}"
                    )
            self._connection.commit()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS m4_pending(
                    record_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    verifier_refs TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS m4_records(
                    record_id TEXT PRIMARY KEY,
                    workflow_run_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    logical_invocation_id TEXT NOT NULL,
                    class_key INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    full_span_json TEXT NOT NULL,
                    local_success INTEGER NOT NULL,
                    danger_tier INTEGER NOT NULL,
                    verifier_refs TEXT NOT NULL,
                    failed_actions TEXT NOT NULL DEFAULT '[]',
                    entered_at REAL NOT NULL,
                    last_access_at REAL NOT NULL,
                    pair_id TEXT,
                    switch_direction TEXT,
                    forced_class INTEGER NOT NULL,
                    assignment_status TEXT NOT NULL,
                    embedding_latency_ms REAL NOT NULL DEFAULT 0.0,
                    embedding_source TEXT NOT NULL DEFAULT 'unknown',
                    action TEXT NOT NULL DEFAULT 'unknown',
                    decision_reason TEXT NOT NULL DEFAULT 'unknown',
                    policy_version TEXT NOT NULL DEFAULT 'unknown',
                    predictor_p REAL,
                    predictor_threshold REAL,
                    predictor_version TEXT,
                    predictor_feature_version TEXT,
                    router_features_json TEXT,
                    task_success INTEGER,
                    label_scope TEXT,
                    span_input_tokens INTEGER,
                    span_output_tokens INTEGER,
                    span_cost REAL,
                    span_latency_ms INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_m4_records_class
                    ON m4_records(class_key);
                CREATE INDEX IF NOT EXISTS idx_m4_records_lineage
                    ON m4_records(workflow_run_id, episode_id, logical_invocation_id);
                """
            )
            self._connection.commit()
            self._ensure_columns()


def _encode_embedding(embedding: Iterable[float]) -> bytes:
    return array("f", (float(value) for value in embedding)).tobytes()


def _decode_embedding(blob: bytes) -> tuple[float, ...]:
    values = array("f")
    values.frombytes(blob)
    return tuple(float(value) for value in values)


def _decode_string_list(raw: Any) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


def _optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    return text or None


def _optional_bool(raw: Any) -> bool | None:
    """SQLite 的 NULL 表示「未知」，必须与 False 区分开。"""

    if raw is None:
        return None
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("", "none", "null", "unknown"):
            return None
        return normalized in ("1", "true", "yes")
    return bool(raw)


def _serialize_full_span(value: Any) -> str:
    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())
    return json.dumps(value, default=str, ensure_ascii=True)


def _deserialize_full_span(raw: Any) -> Any:
    text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


__all__ = ["SQLiteM4MemoryStore"]
