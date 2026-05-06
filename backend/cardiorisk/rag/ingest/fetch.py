"""Idempotent fetcher for the Australian CVD-risk guideline corpus.

Mirrors the contract of :mod:`cardiorisk.data.fetch` (the HFP UCI
fetcher): stream-download with a 60-second timeout, write atomically
(``.part`` then rename), compute SHA-256, and verify against a pinned
lockfile in ``data/checksums/``. Failure modes are explicit:

- First run for a source (no lockfile): write the lockfile from the
  observed digest and log ``"downloaded"``. The maintainer commits the
  resulting ``corpus_<doc_id>.sha256`` file.
- Subsequent run with matching pin: log ``"reused"`` (file already on
  disk) or ``"redownloaded"`` (re-fetched at the user's request via
  ``--force``).
- Subsequent run with **mismatched** pin: raise :class:`FetchError`.
  This is how upstream silent-edits to RACGP / NVDPA PDFs surface.

Both publishers serve the PDFs over plain HTTPS without
authentication, so the fetcher is a thin :mod:`requests` wrapper. The
``--use-fixture`` short-circuit (handled by
``backend/scripts/fetch_corpus.py``) bypasses the network entirely
and reads the markdown fixture directly; that's the CI path.

PDFs are never committed to the repo. The .gitignore rule
``data/external/*`` plus the global ``*.pdf`` rule enforce this; the
``scripts/no_raw_data.sh`` pre-commit hook is the third line of
defence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import requests

from cardiorisk.data.paths import CORPUS_RAW, DATA_CHECKSUMS

from .sources import CORPUS_SOURCES, CorpusSource

HTTP_TIMEOUT_SECONDS: Final[float] = 60.0


class FetchError(RuntimeError):
    """Raised when a corpus download or checksum verification fails."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of ``path``'s contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pinned_checksum(checksum_path: Path) -> str | None:
    """Read the pinned hex SHA-256 from a ``.sha256`` lockfile, or ``None``.

    Same lockfile format as :func:`cardiorisk.data.fetch.read_pinned_checksum`:
    one non-comment line containing the hex digest; ``#`` introduces a
    trailing comment.
    """
    if not checksum_path.exists():
        return None
    for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            return line.lower()
    return None


def write_pinned_checksum(checksum_path: Path, hex_digest: str, source_url: str) -> None:
    """Write a ``.sha256`` lockfile with the digest and a source-URL comment."""
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(
        f"{hex_digest.lower()}  # source: {source_url}\n",
        encoding="utf-8",
    )


def download_to(url: str, out_path: Path) -> None:
    """Stream ``url`` into ``out_path`` atomically (write-then-rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def fetch_one(
    source: CorpusSource,
    *,
    force: bool,
    raw_dir: Path = CORPUS_RAW,
    checksum_dir: Path = DATA_CHECKSUMS,
) -> tuple[Path, str, str]:
    """Fetch one corpus PDF and verify against its pinned checksum.

    Returns ``(out_path, observed_digest, action)`` where ``action`` is
    one of ``"reused"``, ``"downloaded"``, or ``"redownloaded"``.

    Raises :class:`FetchError` if the downloaded file's digest does not
    match the pinned lockfile.
    """
    out_path = raw_dir / source.out_filename
    checksum_path = checksum_dir / source.checksum_filename
    pinned = read_pinned_checksum(checksum_path)

    if not force and out_path.exists() and pinned is not None:
        observed = sha256_of(out_path)
        if observed == pinned:
            return out_path, observed, "reused"

    download_to(source.url, out_path)
    observed = sha256_of(out_path)

    if pinned is None:
        write_pinned_checksum(checksum_path, observed, source.url)
        return out_path, observed, "downloaded"

    if observed != pinned:
        raise FetchError(
            f"checksum mismatch for {source.doc_id}: pinned={pinned}, "
            f"observed={observed}. Either upstream changed (verify and "
            f"update {checksum_path}) or the download was corrupted."
        )
    return out_path, observed, "redownloaded"


def fetch_all(
    *,
    force: bool,
    sources: Iterable[CorpusSource] = CORPUS_SOURCES,
    raw_dir: Path = CORPUS_RAW,
    checksum_dir: Path = DATA_CHECKSUMS,
) -> list[tuple[CorpusSource, Path, str, str]]:
    """Fetch every corpus PDF sequentially.

    The fetch loop is deliberately serial: there are O(3) PDFs and the
    upstream publishers (RACGP, NVDPA) are well-behaved but not
    obviously friendly to parallel hammering. A retry / backoff layer
    can be added if the corpus grows past O(20).
    """
    return [
        (source, *fetch_one(source, force=force, raw_dir=raw_dir, checksum_dir=checksum_dir))
        for source in sources
    ]
