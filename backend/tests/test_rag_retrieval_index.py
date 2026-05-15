"""Tests for the HNSW vector index wrapper.

Builds a small synthetic index and asserts: (a) recall@k saturates
on the planted-near-neighbour query; (b) save / load round-trip is
exact; (c) empty / single-row degenerate cases don't crash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cardiorisk.rag.retrieval.embed import MockEmbedder
from cardiorisk.rag.retrieval.index import HNSWIndex

DIM = 32


def _seeded_unit_vectors(n: int, dim: int = DIM, seed: int = 20260507) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    out: np.ndarray = raw / norms
    return out


def test_build_then_search_returns_self_at_top() -> None:
    vecs = _seeded_unit_vectors(50)
    chunk_ids = [f"c{i:03d}" for i in range(50)]
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=chunk_ids, vectors=vecs)
    # Querying with the exact vector for c042 should return c042 first.
    hits = idx.search(vecs[42], top_k=5)
    assert hits[0].chunk_id == "c042"
    assert hits[0].score > 0.99


def test_recall_at_k_above_threshold() -> None:
    """For 100 random vectors, recall@10 against brute-force top-10 ≥ 0.9."""
    vecs = _seeded_unit_vectors(100)
    chunk_ids = [f"c{i:03d}" for i in range(100)]
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=chunk_ids, vectors=vecs)

    rng = np.random.default_rng(20260507)
    queries = rng.standard_normal((10, DIM)).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    n_match = 0
    n_total = 0
    for q in queries:
        # Brute-force ground truth.
        sims = vecs @ q
        gt_top10 = set(np.argsort(-sims)[:10].tolist())
        # HNSW top-10.
        hnsw_hits = {int(chunk_ids.index(h.chunk_id)) for h in idx.search(q, top_k=10)}
        n_match += len(gt_top10 & hnsw_hits)
        n_total += 10
    recall = n_match / n_total
    assert recall >= 0.9, recall


def test_save_load_round_trip(tmp_path: Path) -> None:
    vecs = _seeded_unit_vectors(20)
    chunk_ids = [f"c{i:03d}" for i in range(20)]
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=chunk_ids, vectors=vecs)
    idx.save(tmp_path)

    idx_loaded = HNSWIndex.load(tmp_path, dim=DIM)
    assert len(idx_loaded) == 20
    assert idx_loaded.chunk_ids == idx.chunk_ids

    # Same query gives same top-1 chunk_id (and approximately the same score).
    hits_orig = idx.search(vecs[5], top_k=3)
    hits_loaded = idx_loaded.search(vecs[5], top_k=3)
    assert hits_orig[0].chunk_id == hits_loaded[0].chunk_id


def test_save_load_dim_mismatch_raises(tmp_path: Path) -> None:
    vecs = _seeded_unit_vectors(5)
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=["a", "b", "c", "d", "e"], vectors=vecs)
    idx.save(tmp_path)
    with pytest.raises(ValueError):
        HNSWIndex.load(tmp_path, dim=999)


def test_load_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HNSWIndex.load(tmp_path, dim=DIM)


def test_search_top_k_caps_at_index_size() -> None:
    vecs = _seeded_unit_vectors(3)
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=["a", "b", "c"], vectors=vecs)
    hits = idx.search(vecs[0], top_k=10)
    assert len(hits) == 3


def test_empty_index_search_returns_empty() -> None:
    idx = HNSWIndex(dim=DIM)
    idx.build(chunk_ids=[], vectors=np.empty((0, DIM), dtype=np.float32))
    q = _seeded_unit_vectors(1)[0]
    assert idx.search(q, top_k=5) == []


def test_query_dim_mismatch_raises() -> None:
    idx = HNSWIndex(dim=DIM)
    idx.build(
        chunk_ids=["a"],
        vectors=_seeded_unit_vectors(1),
    )
    with pytest.raises(ValueError):
        idx.search(np.zeros(99, dtype=np.float32), top_k=1)


def test_build_shape_mismatch_raises() -> None:
    idx = HNSWIndex(dim=DIM)
    with pytest.raises(ValueError):
        idx.build(chunk_ids=["a", "b"], vectors=_seeded_unit_vectors(3))


def test_save_before_build_raises(tmp_path: Path) -> None:
    idx = HNSWIndex(dim=DIM)
    with pytest.raises(RuntimeError):
        idx.save(tmp_path)


def test_works_with_mock_embedder_dim() -> None:
    emb = MockEmbedder()
    texts = ["alpha", "beta", "gamma"]
    vecs = emb.encode(texts)
    chunk_ids = list(texts)
    idx = HNSWIndex(dim=emb.dim)
    idx.build(chunk_ids=chunk_ids, vectors=vecs)
    hits = idx.search(emb.encode(["alpha"])[0], top_k=1)
    assert hits[0].chunk_id == "alpha"
