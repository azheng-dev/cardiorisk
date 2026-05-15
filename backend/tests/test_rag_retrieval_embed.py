"""Tests for the embedder Protocol + concrete embedders + EmbedCache.

Skips MiniLM and BGE-M3 model-load tests if their wheels aren't on
the runner (lets the unit-test pass in environments without
sentence-transformers or FlagEmbedding installed). The fixture-CI
job has both deps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cardiorisk.rag.retrieval.embed import (
    BGE_M3_DIM,
    MINILM_DIM,
    MOCK_DIM,
    BaseEmbedder,
    EmbedCache,
    MockEmbedder,
    get_embedder,
)


def test_mock_embedder_satisfies_protocol() -> None:
    emb = MockEmbedder()
    assert isinstance(emb, BaseEmbedder)
    assert emb.dim == MOCK_DIM


def test_mock_embedder_returns_unit_norm_vectors() -> None:
    emb = MockEmbedder()
    out = emb.encode(["alpha", "beta", "gamma"])
    assert out.shape == (3, MOCK_DIM)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)


def test_mock_embedder_deterministic() -> None:
    emb = MockEmbedder()
    a = emb.encode(["alpha"])
    b = emb.encode(["alpha"])
    np.testing.assert_array_equal(a, b)


def test_mock_embedder_distinct_inputs_distinct_vectors() -> None:
    emb = MockEmbedder()
    a = emb.encode(["alpha"])
    b = emb.encode(["beta"])
    # Different texts should produce different vectors with high probability.
    assert not np.allclose(a, b)


def test_mock_embedder_empty_input() -> None:
    emb = MockEmbedder()
    out = emb.encode([])
    assert out.shape == (0, MOCK_DIM)


def test_get_embedder_factory() -> None:
    assert isinstance(get_embedder("mock"), MockEmbedder)
    with pytest.raises(ValueError):
        get_embedder("not-a-real-embedder")


def test_get_embedder_factory_constants_match() -> None:
    """Constants in embed.py reflect the real embedder dims."""
    assert BGE_M3_DIM == 1024
    assert MINILM_DIM == 384
    assert MOCK_DIM == 64


def test_embed_cache_round_trip(tmp_path: Path) -> None:
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    chunk_ids = ["c1", "c2", "c3"]
    texts = ["alpha text", "beta text", "gamma text"]

    first = cache.encode(chunk_ids, texts)
    assert first.shape == (3, MOCK_DIM)

    second = cache.encode(chunk_ids, texts)
    np.testing.assert_array_equal(first, second)


def test_embed_cache_persists_to_disk(tmp_path: Path) -> None:
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    cache.encode(["c1"], ["alpha"])
    expected = tmp_path / "mock-64" / "c1.npy"
    assert expected.exists()


def test_embed_cache_partial_hit(tmp_path: Path) -> None:
    """Cache hits avoid re-encoding; misses go to the embedder."""
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    cache.encode(["c1"], ["alpha"])

    # Now request c1 (cache hit) and c2 (cache miss); both should
    # come back with the right vectors.
    out = cache.encode(["c1", "c2"], ["alpha", "beta"])
    assert out.shape == (2, MOCK_DIM)
    direct = emb.encode(["alpha", "beta"])
    np.testing.assert_array_equal(out, direct)


def test_embed_cache_clear(tmp_path: Path) -> None:
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    cache.encode(["c1"], ["alpha"])
    cache.clear()
    assert not list((tmp_path / "mock-64").glob("*.npy"))


def test_embed_cache_length_mismatch_raises(tmp_path: Path) -> None:
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    with pytest.raises(ValueError):
        cache.encode(["c1", "c2"], ["alpha"])


def test_embed_cache_query_not_cached(tmp_path: Path) -> None:
    """encode_query embeds without writing to disk."""
    emb = MockEmbedder()
    cache = EmbedCache(tmp_path, emb)
    q = cache.encode_query("some query string")
    assert q.shape == (MOCK_DIM,)
    assert not list((tmp_path / "mock-64").glob("*.npy"))


# Conditional MiniLM smoke (skipped on environments without
# sentence-transformers wheels installed; the CI hot path does have
# them).
def _have_st() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_st(), reason="sentence-transformers not installed")
def test_minilm_round_trip_smoke() -> None:
    """Real-embedder smoke; downloads ~80 MB on first run."""
    from cardiorisk.rag.retrieval.embed import MiniLMEmbedder

    emb = MiniLMEmbedder()
    assert emb.dim == MINILM_DIM
    out = emb.encode(["the patient is at high cardiovascular risk"])
    assert out.shape == (1, MINILM_DIM)
    norm = np.linalg.norm(out[0])
    assert 0.9 <= norm <= 1.1
