from traceoffload.m1.embedding_client import (
    OllamaEmbeddingClient,
    OllamaEmbeddingError,
)

import pytest


def test_embed_many_normalizes_and_returns_metadata() -> None:
    calls = []

    def fake_post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {
            "model": "qwen3-embedding:0.6b",
            "embeddings": [[3.0, 4.0], [0.0, 2.0]],
            "prompt_eval_count": 12,
        }

    client = OllamaEmbeddingClient(post_json=fake_post, timeout_seconds=5.0)

    results = client.embed_many(("hello", "world"))

    assert calls[0][0].endswith("/api/embed")
    assert calls[0][1] == {"model": "qwen3-embedding:0.6b", "input": ["hello", "world"]}
    assert results[0].values == (0.6, 0.8)
    assert results[1].values == (0.0, 1.0)
    assert all(result.model == "qwen3-embedding:0.6b" for result in results)
    assert all(result.dimension == 2 for result in results)
    assert all(result.prompt_eval_count == 12 for result in results)


def test_embed_one_rejects_zero_norm_vector() -> None:
    def fake_post(url, payload, timeout):
        return {"embeddings": [[0.0, 0.0]]}

    client = OllamaEmbeddingClient(post_json=fake_post)

    with pytest.raises(OllamaEmbeddingError):
        client.embed_one("hello")


def test_embed_many_rejects_empty_input() -> None:
    client = OllamaEmbeddingClient(post_json=lambda url, payload, timeout: {})

    with pytest.raises(ValueError):
        client.embed_many(())


def test_embed_many_requires_matching_response_count() -> None:
    client = OllamaEmbeddingClient(post_json=lambda url, payload, timeout: {"embeddings": [[1.0]]})

    with pytest.raises(OllamaEmbeddingError):
        client.embed_many(("a", "b"))
