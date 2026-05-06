"""Source-of-truth list of guideline PDFs the corpus pipeline ingests.

Per the user's Phase 3.1 corpus-scope decision (AGENTS §2 "Open
decisions" / Phase 3.1 plan), this is **RACGP Red Book + NVDPA
absolute-CVD-risk materials only**. The AusCVDRisk calculator and
Therapeutic Guidelines (eTG) cardiac chapters are deferred to "future
scope" (AGENTS §8) and ADR-015.

Each entry is a :class:`CorpusSource` carrying:

- ``doc_id`` — stable identifier; used as the join key everywhere
  downstream (parse output, chunk records, eval-set
  ``expected_doc_id``).
- ``title`` — human-readable, surfaced in the manifest and in the
  retrieval UI later.
- ``publisher`` — ``"RACGP"`` or ``"NVDPA"``; lets the citation layer
  attribute claims correctly.
- ``url`` — public download URL the fetcher will hit.
- ``out_filename`` — local filename under
  ``data/external/corpus/raw/``.
- ``checksum_filename`` — name of the sha256 lockfile under
  ``data/checksums/`` that pins the fetched bytes.

The first-run-pin pattern (`fetch.py`) means the lockfiles do **not**
need to be populated up-front: the maintainer runs
``backend/scripts/fetch_corpus.py`` once, the first run writes the
sha256, subsequent runs verify against it. This mirrors the UCI HFP
fetcher contract (see :mod:`cardiorisk.data.fetch`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CorpusSource:
    """One guideline PDF the pipeline knows how to fetch + ingest.

    Attributes:
        doc_id: Stable identifier (snake_case, no extension). Used as
            the join key in :class:`~cardiorisk.rag.ingest.parse.ParsedDoc`,
            :class:`~cardiorisk.rag.ingest.chunkers.base.Chunk`, the
            manifest, and ``eval/retrieval/questions.jsonl``.
        title: Human-readable document title (cite-friendly).
        publisher: ``"RACGP"`` or ``"NVDPA"``. Drives attribution in
            the citation layer (Phase 3.3).
        url: Publicly downloadable URL of the PDF. Bytes are not
            redistributed by this repo (per ADR-015).
        out_filename: Local filename under
            ``data/external/corpus/raw/``.
        checksum_filename: Lockfile name under ``data/checksums/``.
    """

    doc_id: str
    title: str
    publisher: str
    url: str
    out_filename: str
    checksum_filename: str


# Three sources for the v1 corpus. Deliberately kept small: the
# Phase 3.1 deliverable is the *pipeline*, and Phase 3.2's chunking +
# retrieval eval works just as well on a 3-doc corpus as on a 30-doc
# one. Adding more chapters is an O(1) line-of-code change here.
CORPUS_SOURCES: Final[tuple[CorpusSource, ...]] = (
    CorpusSource(
        doc_id="racgp_redbook_cvd",
        title=(
            "RACGP Red Book — Guidelines for preventive activities in general "
            "practice (10th edn): Cardiovascular disease prevention chapter"
        ),
        publisher="RACGP",
        url=(
            "https://www.racgp.org.au/clinical-resources/clinical-guidelines/"
            "key-racgp-guidelines/view-all-racgp-guidelines/red-book/"
            "prevention-of-cardiovascular-disease.pdf"
        ),
        out_filename="racgp_redbook_cvd.pdf",
        checksum_filename="corpus_racgp_redbook_cvd.sha256",
    ),
    CorpusSource(
        doc_id="nvdpa_2023_australian_cvd_risk_guideline",
        title=(
            "Australian Guideline for assessing and managing cardiovascular "
            "disease risk (2023, NVDPA / Heart Foundation)"
        ),
        publisher="NVDPA",
        url=(
            "https://www.cvdcheck.org.au/sites/default/files/2023-07/"
            "AustCVDRisk_FullGuideline_2023.pdf"
        ),
        out_filename="nvdpa_2023_australian_cvd_risk_guideline.pdf",
        checksum_filename="corpus_nvdpa_2023_full_guideline.sha256",
    ),
    CorpusSource(
        doc_id="nvdpa_2023_quick_reference_guide",
        title=(
            "Australian CVD Risk Assessment 2023 — Quick reference guide (NVDPA / Heart Foundation)"
        ),
        publisher="NVDPA",
        url=(
            "https://www.cvdcheck.org.au/sites/default/files/2023-07/"
            "AustCVDRisk_QuickReferenceGuide_2023.pdf"
        ),
        out_filename="nvdpa_2023_quick_reference_guide.pdf",
        checksum_filename="corpus_nvdpa_2023_quick_reference.sha256",
    ),
)


_BY_DOC_ID: Final[dict[str, CorpusSource]] = {s.doc_id: s for s in CORPUS_SOURCES}


def get_source(doc_id: str) -> CorpusSource:
    """Return the :class:`CorpusSource` with the given ``doc_id``.

    Raises :class:`KeyError` with the list of known ids on miss, so
    typos in CLI ``--source`` arguments fail loudly with a useful
    message.
    """
    try:
        return _BY_DOC_ID[doc_id]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_DOC_ID))
        raise KeyError(f"unknown doc_id {doc_id!r}; known: {known}") from exc
