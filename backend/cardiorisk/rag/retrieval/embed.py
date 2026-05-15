"""Embedders + on-disk cache for the dense retrieval leg.

Three concrete embedders behind one Protocol:

- :class:`BGEM3Embedder` — production embedder, :mod:`FlagEmbedding`'s
  ``BGEM3FlagModel``. 1024-d output, ~2.27 GB weights downloaded once
  to the Hugging Face cache. Used for the local headline run.
- :class:`MiniLMEmbedder` — CI-smoke embedder,
  ``sentence-transformers/all-MiniLM-L6-v2`` (~80 MB, 384-d output).
  Real embedder so the wiring exercises end-to-end; light enough to
  cache on a CI runner.
- :class:`MockEmbedder` — pure-Python deterministic embedder. Returns
  ``hash(chunk_id) -> 64-d unit vector``. Used by unit tests that
  require byte-identical fixtures with no model load.

All three implement :class:`BaseEmbedder`. The :class:`EmbedCache`
class wraps any :class:`BaseEmbedder` with a disk-backed
``(model_name, chunk_id) -> np.ndarray`` cache so re-runs of the
eval don't re-encode unchanged chunks.

Determinism: BGEM3 + MiniLM are deterministic given fixed inputs and
the same model weights; the MockEmbedder is deterministic by
construction. The eval orchestrator pins ``torch.set_num_threads(1)``
before importing this module to keep results bit-stable across runs.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

#: Pinned RNG seed (matches the rest of the project).
SEED: int = 20260505

#: BGE-M3 produces 1024-d dense vectors.
BGE_M3_DIM: int = 1024
#: MiniLM-L6-v2 produces 384-d vectors.
MINILM_DIM: int = 384
#: Mock embedder uses a small fixed dim so unit tests run fast.
MOCK_DIM: int = 64


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol every embedder satisfies.

    Implementations must be **stateless across calls** apart from the
    cached model weights — no per-call mutation of internal buffers.
    """

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Return an ``(n, dim)`` float32 array of L2-normalised vectors."""
        ...


def _l2_normalise(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-wise L2 normalise; zero-rows stay zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


class MockEmbedder:
    """Deterministic, dep-free embedder used by unit tests.

    Embeds each text into a 64-d unit vector seeded by sha256(text).
    Two identical texts produce identical vectors; near-duplicate
    texts produce uncorrelated vectors (this is *not* a semantic
    embedder). The `RRF` and `pipeline` tests use it because they
    care about the wiring, not the semantics.
    """

    name: str = "mock-64"
    dim: int = MOCK_DIM

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**31)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return _l2_normalise(out)


class MiniLMEmbedder:
    """``sentence-transformers/all-MiniLM-L6-v2`` wrapper.

    Used for the CI smoke (~80 MB weights, 384-d output, fast on a
    CPU runner). Lazily imports :mod:`sentence_transformers` so the
    unit-test import path doesn't pay the dep cost.
    """

    name: str = "all-MiniLM-L6-v2"
    dim: int = MINILM_DIM

    def __init__(self, *, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class BGEM3Embedder:
    """``BAAI/bge-m3`` wrapper via :class:`FlagEmbedding.BGEM3FlagModel`.

    Production embedder. 1024-d dense output. We use only the dense
    head; bge-m3's sparse and ColBERT heads are research escape
    hatches documented in ADR-016 §1.
    """

    name: str = "bge-m3"
    dim: int = BGE_M3_DIM

    def __init__(self, *, batch_size: int = 8, use_fp16: bool = False) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        out = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense: Any = out["dense_vecs"]
        arr = np.asarray(dense, dtype=np.float32)
        return _l2_normalise(arr)


def get_embedder(name: str, **kwargs: Any) -> BaseEmbedder:
    """Factory for the eval orchestrator's ``--embedder`` flag."""
    if name == "mock":
        return MockEmbedder()
    if name == "minilm":
        return MiniLMEmbedder(**kwargs)
    if name in ("bge-m3", "bgem3"):
        return BGEM3Embedder(**kwargs)
    raise ValueError(f"unknown embedder {name!r}; known: mock, minilm, bge-m3")


class EmbedCache:
    """Disk-backed ``(model_name, chunk_id) -> np.ndarray`` cache.

    Layout::

        <root>/<embedder_name>/<chunk_id>.npy

    A cache hit avoids re-encoding the chunk; the eval can sweep
    chunkers and rerank conditions without re-paying the embedder
    cost on every run. Concurrent writes use atomic ``.part`` →
    ``rename`` so a partial write never poisons the cache.
    """

    def __init__(self, root: Path, embedder: BaseEmbedder) -> None:
        self._root = root / embedder.name
        self._root.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder

    def _path_for(self, chunk_id: str) -> Path:
        return self._root / f"{chunk_id}.npy"

    def encode(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Encode ``texts`` (with ``chunk_ids`` for cache keys); cache misses go to the embedder.

        Args:
            chunk_ids: One id per row in ``texts``. Must be the same length.
            texts: Texts to embed. ``chunk_ids[i]`` is the cache key for ``texts[i]``.

        Returns:
            ``(n, dim)`` float32 array, rows aligned with ``texts``.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) length mismatch"
            )
        n = len(texts)
        out = np.empty((n, self._embedder.dim), dtype=np.float32)

        miss_idx: list[int] = []
        for i, chunk_id in enumerate(chunk_ids):
            cache_path = self._path_for(chunk_id)
            if cache_path.exists():
                out[i] = np.load(cache_path)
            else:
                miss_idx.append(i)

        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            miss_vecs = self._embedder.encode(miss_texts)
            for j, i in enumerate(miss_idx):
                out[i] = miss_vecs[j]
                self._save_atomic(self._path_for(chunk_ids[i]), miss_vecs[j])

        return out

    def encode_query(self, query: str) -> npt.NDArray[np.float32]:
        """Embed a query string; queries are not cached (one per call)."""
        out: npt.NDArray[np.float32] = self._embedder.encode([query])[0]
        return out

    def _save_atomic(self, path: Path, arr: npt.NDArray[np.float32]) -> None:
        # np.save() automatically appends ".npy" to non-".npy" paths;
        # writing through an open file handle bypasses that footgun.
        tmp = path.with_name(path.name + ".part")
        with tmp.open("wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)

    def clear(self) -> None:
        """Remove all cached vectors for this embedder. Used by tests."""
        for f in self._root.glob("*.npy"):
            f.unlink()
        for f in self._root.glob("*.part"):
            f.unlink()
