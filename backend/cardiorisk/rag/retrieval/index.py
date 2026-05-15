"""HNSW vector index — :mod:`hnswlib` thin wrapper.

Phase 3.2's vector leg. The interface is deliberately small (build /
save / load / search) so Phase 4 can swap the implementation for a
``PgVectorIndex`` against the same surface.

Tuning (from ADR-016 §2):

- ``space="cosine"`` — matches the L2-normalised vectors the
  :class:`BaseEmbedder` Protocol returns.
- ``M=16`` — the hnswlib default; balances recall and graph size.
  Doubling M to 32 gains <1% recall on our scale and doubles the
  graph build time.
- ``ef_construction=200`` — twice the published default; the build
  is one-shot per chunker so we pay it gladly for the recall.
- ``ef`` (search-time) — set to ``max(2 * top_k, 50)`` to keep
  recall@top_k near saturation while staying cheap.

File format: hnswlib persists a single binary blob; we save it to
``vector.bin`` next to a sibling ``ids.json`` that records the row
ordering (hnswlib internally uses integer labels; we map them back
to chunk_ids).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class HNSWHit:
    """One result row from :meth:`HNSWIndex.search`."""

    chunk_id: str
    score: float


class HNSWIndex:
    """In-memory cosine HNSW index keyed by ``chunk_id``.

    Build pattern::

        idx = HNSWIndex(dim=1024)
        idx.build(chunk_ids=["a", "b", ...], vectors=embeddings)
        idx.save(path)

    Query pattern::

        idx = HNSWIndex.load(path, dim=1024)
        hits = idx.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        *,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self._dim = dim
        self._m = m
        self._ef_construction = ef_construction
        self._index: Any | None = None
        self._chunk_ids: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(self._chunk_ids)

    def build(
        self,
        *,
        chunk_ids: list[str],
        vectors: npt.NDArray[np.float32],
    ) -> None:
        """Build the index from ``(chunk_id, vector)`` pairs.

        ``vectors`` must be ``(len(chunk_ids), self.dim)``. Vectors
        are assumed already L2-normalised (the cosine space treats
        the L2 distance correctly only when both query and index
        vectors are unit-norm — see hnswlib README §"Cosine
        similarity").
        """
        n = len(chunk_ids)
        if vectors.shape != (n, self._dim):
            raise ValueError(f"vectors shape {vectors.shape} != expected ({n}, {self._dim})")
        if n == 0:
            self._index = hnswlib.Index(space="cosine", dim=self._dim)
            self._index.init_index(max_elements=1, ef_construction=self._ef_construction, M=self._m)
            self._chunk_ids = []
            return
        self._index = hnswlib.Index(space="cosine", dim=self._dim)
        self._index.init_index(
            max_elements=n,
            ef_construction=self._ef_construction,
            M=self._m,
        )
        labels = np.arange(n, dtype=np.int64)
        self._index.add_items(vectors.astype(np.float32, copy=False), labels)
        self._chunk_ids = list(chunk_ids)

    def search(
        self,
        query: npt.NDArray[np.float32],
        *,
        top_k: int,
    ) -> list[HNSWHit]:
        """Return the top-``top_k`` nearest chunks to ``query``.

        ``query`` is a 1-d vector of shape ``(dim,)``. The returned
        ``score`` is ``1 - cosine_distance`` ∈ ``[-1, 1]`` (so a
        higher score means closer). For unit-norm vectors this equals
        the cosine similarity.
        """
        if self._index is None or not self._chunk_ids:
            return []
        if query.shape != (self._dim,):
            raise ValueError(f"query shape {query.shape} != expected ({self._dim},)")
        k = min(top_k, len(self._chunk_ids))
        ef = max(2 * top_k, 50)
        self._index.set_ef(ef)
        labels, distances = self._index.knn_query(query.astype(np.float32, copy=False), k=k)
        labels = labels[0]
        distances = distances[0]
        return [
            HNSWHit(chunk_id=self._chunk_ids[int(label)], score=float(1.0 - dist))
            for label, dist in zip(labels, distances, strict=True)
        ]

    def save(self, path: Path) -> None:
        """Persist the index to ``path/vector.bin`` + ``path/ids.json``."""
        if self._index is None:
            raise RuntimeError("index has not been built; call .build() first")
        path.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path / "vector.bin"))
        payload = {"chunk_ids": self._chunk_ids, "dim": self._dim}
        (path / "ids.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, dim: int | None = None) -> HNSWIndex:
        """Load a previously-saved index from ``path``.

        ``dim`` is read from the persisted ``ids.json`` if not given;
        passing it explicitly is a defensive cross-check used by the
        pipeline tests.
        """
        ids_path = path / "ids.json"
        bin_path = path / "vector.bin"
        if not ids_path.exists() or not bin_path.exists():
            raise FileNotFoundError(f"HNSW index not found at {path} (need vector.bin + ids.json)")
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        loaded_dim = int(payload["dim"])
        if dim is not None and dim != loaded_dim:
            raise ValueError(f"dim mismatch: caller passed {dim}, file has {loaded_dim}")
        idx = cls(dim=loaded_dim)
        idx._index = hnswlib.Index(space="cosine", dim=loaded_dim)
        n = len(payload["chunk_ids"])
        idx._index.load_index(str(bin_path), max_elements=max(n, 1))
        idx._chunk_ids = list(payload["chunk_ids"])
        return idx
