"""Cross-encoder reranker on top of the RRF-fused candidate set.

Phase 3.2's reranker stage. ADR-016 §4 picks
:class:`BAAI/bge-reranker-v2-m3` as the production reranker; this
module wraps :class:`FlagEmbedding.FlagReranker` behind a small
Protocol so unit tests can swap in :class:`MockReranker`.

Latency: cross-encoder over ~50 candidates is ~150 ms on CPU. The
Phase 3.3 LLM call dwarfs this, but a future low-latency surface
will profile it. ONNX export + quantisation are the natural follow-on
steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseReranker(Protocol):
    """Pluggable reranker; consumes (query, passages), returns scores."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return a relevance score per passage. Higher = more relevant."""
        ...


class MockReranker:
    """Deterministic, dep-free reranker used by unit tests.

    Returns a score equal to the count of shared lowercase tokens
    between query and passage, divided by passage token count. Not
    semantically meaningful, but exercises the pipeline wiring with
    a monotonically-sensible ordering for fixture queries.
    """

    name: str = "mock-token-overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        scores: list[float] = []
        for p in passages:
            p_tokens = p.lower().split()
            if not p_tokens:
                scores.append(0.0)
                continue
            shared = sum(1 for t in p_tokens if t in q_tokens)
            scores.append(float(shared) / float(len(p_tokens)))
        return scores


class BGEReranker:
    """``BAAI/bge-reranker-v2-m3`` wrapper via :class:`sentence_transformers.CrossEncoder`.

    We use sentence-transformers' ``CrossEncoder`` rather than
    ``FlagEmbedding.FlagReranker`` because the latter still calls the
    deprecated ``Tokenizer.prepare_for_model`` API which was removed in
    transformers 5.x. ``CrossEncoder`` wraps the same checkpoint
    (``BAAI/bge-reranker-v2-m3`` is published in cross-encoder format on
    the Hub) and tracks current transformers releases.
    """

    name: str = "bge-reranker-v2-m3"

    def __init__(self, *, batch_size: int = 8) -> None:
        import contextlib

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        # Cross-encoder forward pass has no need for gradient tracking
        # at inference time. Without an explicit ``torch.inference_mode``
        # in ``score`` the autograd graph is materialised every call;
        # over the 50 Q x 3 rerank-cell matrix this leaks several GB of
        # activations and eventually thrashes on macOS. We also flip the
        # underlying nn.Module to eval mode defensively (skip silently
        # if a future sentence-transformers release renames the attr).
        with contextlib.suppress(AttributeError):
            self._model.model.eval()
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        import torch

        pairs = [[query, p] for p in passages]
        with torch.inference_mode():
            raw: Any = self._model.predict(pairs, batch_size=self._batch_size)
        # CrossEncoder.predict returns a numpy 1-d array; normalise to list[float].
        return [float(x) for x in raw]


def get_reranker(name: str, **kwargs: Any) -> BaseReranker:
    """Factory mirroring :func:`embed.get_embedder`."""
    if name == "mock":
        return MockReranker()
    if name in ("bge", "bge-reranker", "bge-reranker-v2-m3"):
        return BGEReranker(**kwargs)
    raise ValueError(f"unknown reranker {name!r}; known: mock, bge-reranker-v2-m3")
