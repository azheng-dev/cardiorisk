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
        # Direct PDF download surfaced from the RACGP "Cardiovascular disease
        # (CVD) risk" chapter page; see ADR-015 (Phase 3.2 amendment) for the
        # URL-resolution audit that swapped the previous
        # ``red-book/prevention-of-cardiovascular-disease.pdf`` path (now
        # 404). The ``getattachment/<guid>/...aspx`` URL is the canonical
        # download surface RACGP exposes for the full Red Book PDF as of
        # 2026-05-15.
        url=(
            "https://www.racgp.org.au/getattachment/"
            "9755764e-25f8-4799-bbca-29ddaf8c6d65/"
            "Guidelines-for-preventive-activities-in-general-practice.aspx"
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
        # cvdcheck.org.au moved to a Next.js front-end after July 2023 and
        # the original ``sites/default/files/2023-07/...`` paths are 404.
        # PDFs are now hosted on the CloudFront origin
        # ``d35rj4ptypp2hd.cloudfront.net`` and surfaced through JS. See
        # ADR-015 amendment for the URL-resolution audit.
        url=(
            "https://d35rj4ptypp2hd.cloudfront.net/pdf/"
            "Guideline-for-assessing-and-managing-CVD-risk_20230522.pdf"
        ),
        out_filename="nvdpa_2023_australian_cvd_risk_guideline.pdf",
        checksum_filename="corpus_nvdpa_2023_full_guideline.sha256",
    ),
    CorpusSource(
        doc_id="nvdpa_2023_summary_of_recommendations",
        title=(
            "Australian CVD Risk Assessment 2023 — Summary of "
            "recommendations (NVDPA / Heart Foundation)"
        ),
        publisher="NVDPA",
        # The 2023 ``QuickReferenceGuide`` PDF was retired with the cvdcheck
        # site rebuild; the ``Summary-of-recommendations`` PDF is the
        # current quick-reference-shaped surface (~310 KB; one-page-per
        # recommendation block, identical citation discipline to the full
        # guideline).
        url=(
            "https://d35rj4ptypp2hd.cloudfront.net/pdf/"
            "CVD-Risk-Guideline-Document-Summary-of-recommendations.pdf"
        ),
        out_filename="nvdpa_2023_summary_of_recommendations.pdf",
        checksum_filename="corpus_nvdpa_2023_summary_of_recommendations.sha256",
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
