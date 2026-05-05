"""Idempotent fetcher for the UCI Heart Disease subsets and the optional
Kaggle HFP combined CSV.

Verifies SHA-256 of every download against pinned values in
``data/checksums/``, refusing to keep a file whose hash diverges from the
lockfile (failure mode is the maintainer noticing upstream drift in CI).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

from .paths import DATA_CHECKSUMS, DATA_RAW, FIXTURE_PATH

UCI_BASE: Final[str] = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
KAGGLE_DATASET: Final[str] = "fedesoriano/heart-failure-prediction"
KAGGLE_FILE: Final[str] = "heart.csv"
HTTP_TIMEOUT_SECONDS: Final[float] = 60.0


@dataclass(frozen=True)
class UciSource:
    """One UCI Heart Disease processed subset."""

    name: str
    url: str
    out_filename: str
    checksum_filename: str


UCI_SOURCES: Final[tuple[UciSource, ...]] = (
    UciSource(
        name="cleveland",
        url=f"{UCI_BASE}/processed.cleveland.data",
        out_filename="processed.cleveland.data",
        checksum_filename="uci_cleveland.sha256",
    ),
    UciSource(
        name="hungarian",
        url=f"{UCI_BASE}/processed.hungarian.data",
        out_filename="processed.hungarian.data",
        checksum_filename="uci_hungarian.sha256",
    ),
    UciSource(
        name="switzerland",
        url=f"{UCI_BASE}/processed.switzerland.data",
        out_filename="processed.switzerland.data",
        checksum_filename="uci_switzerland.sha256",
    ),
    UciSource(
        name="va",
        url=f"{UCI_BASE}/processed.va.data",
        out_filename="processed.va.data",
        checksum_filename="uci_va.sha256",
    ),
)

KAGGLE_CHECKSUM_FILENAME: Final[str] = "kaggle_hfp.sha256"
KAGGLE_OUT_FILENAME: Final[str] = "hfp_kaggle.csv"


class FetchError(RuntimeError):
    """Raised when a download or checksum verification fails."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of ``path``'s contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pinned_checksum(checksum_path: Path) -> str | None:
    """Read the pinned hex SHA-256 from a ``.sha256`` lockfile, or None.

    File format: a single non-comment line containing the hex digest
    (case-insensitive). Anything after a ``#`` is treated as a comment.
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
    source: UciSource,
    *,
    force: bool,
    raw_dir: Path = DATA_RAW,
    checksum_dir: Path = DATA_CHECKSUMS,
) -> tuple[Path, str, str]:
    """Fetch one UCI source and verify against its pinned checksum.

    Returns ``(out_path, observed_digest, action)`` where ``action`` is one
    of ``"reused"``, ``"downloaded"``, or ``"redownloaded"``.

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
            f"checksum mismatch for {source.name}: pinned={pinned}, "
            f"observed={observed}. Either upstream changed (verify and "
            f"update {checksum_path}) or the download was corrupted."
        )
    return out_path, observed, "redownloaded"


def fetch_all_uci(
    *,
    force: bool,
    sources: Iterable[UciSource] = UCI_SOURCES,
    raw_dir: Path = DATA_RAW,
    checksum_dir: Path = DATA_CHECKSUMS,
) -> list[tuple[UciSource, Path, str, str]]:
    """Fetch every UCI subset sequentially."""
    return [
        (source, *fetch_one(source, force=force, raw_dir=raw_dir, checksum_dir=checksum_dir))
        for source in sources
    ]


def fetch_kaggle(
    *,
    force: bool,
    raw_dir: Path = DATA_RAW,
    checksum_dir: Path = DATA_CHECKSUMS,
) -> tuple[Path, str, str]:
    """Fetch the Kaggle HFP combined CSV.

    Lazy-imports ``kaggle`` because that package calls ``sys.exit(1)`` at
    import time when no API credentials are present.
    """
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        raise FetchError(
            "KAGGLE_USERNAME and KAGGLE_KEY env vars required for Kaggle fetch. "
            "Get them from https://www.kaggle.com/settings -> API -> Create New Token."
        )

    out_path = raw_dir / KAGGLE_OUT_FILENAME
    checksum_path = checksum_dir / KAGGLE_CHECKSUM_FILENAME
    pinned = read_pinned_checksum(checksum_path)

    if not force and out_path.exists() and pinned is not None:
        observed = sha256_of(out_path)
        if observed == pinned:
            return out_path, observed, "reused"

    raw_dir.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        dataset=KAGGLE_DATASET,
        file_name=KAGGLE_FILE,
        path=str(raw_dir),
        force=True,
    )

    downloaded_zip = raw_dir / f"{KAGGLE_FILE}.zip"
    if downloaded_zip.exists():
        import zipfile

        with zipfile.ZipFile(downloaded_zip) as zf:
            zf.extractall(raw_dir)
        downloaded_zip.unlink()

    extracted = raw_dir / KAGGLE_FILE
    if not extracted.exists():
        raise FetchError(f"expected {extracted} after Kaggle download; not found")
    extracted.replace(out_path)

    observed = sha256_of(out_path)
    if pinned is None:
        write_pinned_checksum(
            checksum_path,
            observed,
            f"kaggle://{KAGGLE_DATASET}/{KAGGLE_FILE}",
        )
        return out_path, observed, "downloaded"

    if observed != pinned:
        raise FetchError(
            f"Kaggle checksum mismatch: pinned={pinned}, observed={observed}. "
            f"Verify upstream and update {checksum_path}."
        )
    return out_path, observed, "redownloaded"


def use_fixture(
    *,
    fixture_path: Path = FIXTURE_PATH,
    raw_dir: Path = DATA_RAW,
) -> Path:
    """Copy the synthetic fixture into ``raw_dir`` and return its path.

    CI uses this to exercise downstream code without needing network access.
    """
    if not fixture_path.exists():
        raise FetchError(
            f"fixture not found at {fixture_path}. "
            "Run `uv run python backend/scripts/generate_fixture.py` first."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "hfp_mini.csv"
    shutil.copyfile(fixture_path, target)
    return target
