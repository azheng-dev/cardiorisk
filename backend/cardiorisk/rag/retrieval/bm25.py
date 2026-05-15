"""BM25 sparse-retrieval leg — :mod:`rank_bm25` thin wrapper.

Pickle-backed save / load round-trip mirrors the HNSW index pattern.
A small vendored English stopword list keeps the dep footprint
identical to Phase 3.1 (no NLTK runtime download).

Tokenisation (deliberately simple, per ADR-016 §3): unicode-aware
whitespace split, lowercase, strip ASCII punctuation, drop tokens
shorter than 2 characters and tokens in :data:`STOPWORDS`. The honest
weakness — medical compound terms ("ACE-I", "BP", "CKD") get
tokenised non-uniformly — is documented in the research note (§8).
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rank_bm25 import BM25Okapi

#: Short vendored English stopword list. Intentionally small — we
#: keep medically-relevant function words ("not", "no") because they
#: invert clinical meaning. Lifted from the union of the rank_bm25
#: docs example list and the NLTK English list, restricted to words
#: that don't carry clinical meaning.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "of",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "doing",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "their",
        "his",
        "her",
        "its",
        "our",
        "your",
        "my",
        "me",
        "us",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "into",
        "than",
        "then",
        "so",
        "such",
        "about",
    }
)

# Token pattern: keep alphanumeric runs (medical numbers like "140"
# and "10%" matter); the regex captures word characters and a
# trailing optional '%' character so percentage strings survive.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+%?")


def tokenise(text: str) -> list[str]:
    """Tokenise ``text`` for the BM25 sparse leg.

    Steps: extract alphanumeric (+ trailing percent) runs, lowercase,
    drop length-1 tokens (except numeric digits, which we keep
    because '5' / '10' / '45' are clinically meaningful), drop tokens
    in :data:`STOPWORDS`.
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.lower()
        # Keep numeric-only tokens of length 1+ (digits matter
        # clinically); drop alphabetic singletons.
        if len(tok) < 2 and not tok.isdigit():
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


@dataclass(frozen=True)
class BM25Hit:
    """One result row from :meth:`BM25Index.search`."""

    chunk_id: str
    score: float


class BM25Index:
    """``BM25Okapi`` over ``(chunk_id, text)`` pairs.

    Build pattern mirrors :class:`HNSWIndex`::

        idx = BM25Index()
        idx.build(chunk_ids=["a", "b", ...], texts=["...", "..."])
        idx.save(path)

    Query pattern::

        idx = BM25Index.load(path)
        hits = idx.search("query string", top_k=5)
    """

    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._tokenised: list[list[str]] = []

    def __len__(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(self._chunk_ids)

    def build(self, *, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) length mismatch"
            )
        self._chunk_ids = list(chunk_ids)
        self._tokenised = [tokenise(t) for t in texts]
        if not self._tokenised:
            # rank_bm25 raises on empty corpus; we keep self._bm25 = None
            # and short-circuit search().
            self._bm25 = None
            return
        # rank_bm25 raises if any document tokenises to []. Replace
        # empty token lists with a single sentinel token that will
        # never match a real query.
        prepared = [toks if toks else ["__empty__"] for toks in self._tokenised]
        self._bm25 = BM25Okapi(prepared)

    def search(self, query: str, *, top_k: int) -> list[BM25Hit]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        toks = tokenise(query)
        if not toks:
            return []
        scores: Any = self._bm25.get_scores(toks)
        # Argsort descending, take top_k. We deliberately do NOT
        # filter on score > 0: BM25Okapi can produce zero or even
        # negative IDFs on small corpora where a query term appears
        # in most documents. Filtering on score > 0 would hide
        # valid-but-uninformative ranking signal; the eval scorer
        # decides whether a hit "counts" via the doc_id + page_range
        # + keyword check, not via score thresholding.
        order = scores.argsort()[::-1][:top_k]
        return [
            BM25Hit(chunk_id=self._chunk_ids[int(i)], score=float(scores[int(i)])) for i in order
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self._chunk_ids,
            "tokenised": self._tokenised,
        }
        with (path / "bm25.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=4)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        bm25_path = path / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        with bm25_path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        idx = cls()
        idx._chunk_ids = list(payload["chunk_ids"])
        idx._tokenised = [list(t) for t in payload["tokenised"]]
        if idx._tokenised:
            prepared = [toks if toks else ["__empty__"] for toks in idx._tokenised]
            idx._bm25 = BM25Okapi(prepared)
        return idx
