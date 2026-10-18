"""Ollama Qwen3-Embedding 客户端（显式调用，不进入 M1 热路径）。

默认只提供可注入的 HTTP 适配层；调用方在离线/异步场景下显式使用。
客户端不做缓存、不写入 Agent 上下文，也不修改任何消息。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from urllib.request import Request, urlopen

from ..clock import real_time

DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 30.0

JsonPost = Callable[[str, Mapping[str, object], float], Mapping[str, object]]


@dataclass(frozen=True)
class EmbeddingVector:
    """一次 embedding 的 1024 维结果与审计元数据。"""

    values: tuple[float, ...]
    model: str
    dimension: int
    prompt_eval_count: int | None = None
    latency_ms: float = 0.0


class OllamaEmbeddingError(RuntimeError):
    """Ollama 接口不可用、返回结构异常或向量非法时抛出。"""


class OllamaEmbeddingClient:
    """调用 Ollama ``/api/embed`` 的原生客户端。

    这只是一个旁路工具，不是 M1 的必调组件。若后续要在线使用，必须额外
    设计缓存、超时、fail-closed 和异步策略，避免给主链路增加不可控时延。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        normalize: bool = True,
        post_json: JsonPost | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._normalize = normalize
        self._post_json = post_json or _post_json

    @property
    def model(self) -> str:
        return self._model

    def embed_one(self, text: str) -> EmbeddingVector:
        return self.embed_many((text,))[0]

    def embed_many(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if not texts:
            raise ValueError("texts cannot be empty")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("all texts must be strings")
        started = real_time()
        response = self._post_json(
            f"{self._base_url}/api/embed",
            {"model": self._model, "input": list(texts)},
            self._timeout_seconds,
        )
        latency_ms = (real_time() - started) * 1000.0
        raw_vectors = response.get("embeddings")
        if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
            raise OllamaEmbeddingError("ollama response embeddings count mismatch")
        vectors = tuple(
            _parse_vector(raw_vector, normalize=self._normalize)
            for raw_vector in raw_vectors
        )
        prompt_eval_count = response.get("prompt_eval_count")
        if isinstance(prompt_eval_count, bool) or not isinstance(prompt_eval_count, int):
            prompt_eval_count = None
        returned_model = response.get("model")
        model = returned_model if isinstance(returned_model, str) and returned_model else self._model
        return tuple(
            EmbeddingVector(
                values=vector,
                model=model,
                dimension=len(vector),
                prompt_eval_count=prompt_eval_count,
                latency_ms=latency_ms,
            )
            for vector in vectors
        )


def _parse_vector(raw: object, *, normalize: bool) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise OllamaEmbeddingError("ollama embedding vector must be a non-empty list")
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise OllamaEmbeddingError("ollama embedding vector contains a non-number")
        value = float(item)
        if not math.isfinite(value):
            raise OllamaEmbeddingError("ollama embedding vector contains a non-finite number")
        values.append(value)
    if not normalize:
        return tuple(values)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise OllamaEmbeddingError("ollama embedding vector has zero norm")
    return tuple(value / norm for value in values)


def _post_json(
    url: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> Mapping[str, object]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # pragma: no cover - network-specific path
        raise OllamaEmbeddingError(f"ollama embedding request failed: {error}") from error
    if not isinstance(decoded, dict):
        raise OllamaEmbeddingError("ollama embedding response must be a JSON object")
    return decoded


__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingVector",
    "OllamaEmbeddingClient",
    "OllamaEmbeddingError",
]
