"""Token-window chunker: cl100k_base, 512 tokens, 64-token stride.

Robust baseline: ignores document structure entirely and walks the
token stream in fixed-size windows with overlap. This is the
strategy Phase 3.2's retrieval eval will use as the reference point;
the semantic and hybrid chunkers have to *beat* it on hit@k or
they're not paying for their additional complexity.

Implementation: tiktoken does not expose per-token character offsets
directly. We compute them by accumulating the byte length of each
token's bytes-decoding (``decode_single_token_bytes``), then mapping
byte offsets to char offsets via a one-pass UTF-8 walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..parse import ParsedDoc
from .base import Chunk, _enc, make_chunk

DEFAULT_WINDOW_TOKENS: Final[int] = 512
DEFAULT_STRIDE_TOKENS: Final[int] = 448  # 512 - 64 overlap


@dataclass(frozen=True)
class TokenWindowChunker:
    """Fixed-size token-window chunker."""

    name: str = "token"
    window_tokens: int = DEFAULT_WINDOW_TOKENS
    stride_tokens: int = DEFAULT_STRIDE_TOKENS

    def __post_init__(self) -> None:
        if self.window_tokens <= 0:
            raise ValueError(f"window_tokens must be > 0, got {self.window_tokens}")
        if self.stride_tokens <= 0:
            raise ValueError(f"stride_tokens must be > 0, got {self.stride_tokens}")
        if self.stride_tokens > self.window_tokens:
            raise ValueError(
                f"stride_tokens ({self.stride_tokens}) must not exceed "
                f"window_tokens ({self.window_tokens})"
            )

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        text = doc.full_text()
        if not text:
            return []
        enc = _enc()
        token_ids = enc.encode(text)
        if not token_ids:
            return []

        # Per-token byte lengths -> cumulative byte offsets.
        per_token_bytes = [enc.decode_single_token_bytes(t) for t in token_ids]
        byte_offsets = [0]
        for b in per_token_bytes:
            byte_offsets.append(byte_offsets[-1] + len(b))
        # Byte offset -> char offset map. For ASCII (the common case)
        # byte_to_char is the identity; for Latin-1 + UTF-8 prose this
        # walk is O(len(text_bytes)) once.
        text_bytes = text.encode("utf-8")
        byte_to_char = _build_byte_to_char_map(text_bytes)

        chunks: list[Chunk] = []
        n = len(token_ids)
        for window_start in range(0, n, self.stride_tokens):
            window_end = min(window_start + self.window_tokens, n)
            byte_start = byte_offsets[window_start]
            byte_end = byte_offsets[window_end]
            char_start = byte_to_char[byte_start]
            char_end = byte_to_char[byte_end]
            if char_end <= char_start:
                continue
            chunk_text = text[char_start:char_end]
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.name,
                    char_start=char_start,
                    char_end=char_end,
                    text=chunk_text,
                )
            )
            if window_end == n:
                break
        return chunks


def _build_byte_to_char_map(text_bytes: bytes) -> list[int]:
    """Return ``byte_to_char`` of length ``len(text_bytes) + 1``.

    ``byte_to_char[i]`` is the char position corresponding to byte
    position ``i`` in a valid UTF-8 byte string. Continuation-byte
    positions inside a multi-byte char map to the *start* of that
    char.
    """
    n = len(text_bytes)
    byte_to_char = [0] * (n + 1)
    char_idx = 0
    i = 0
    while i < n:
        b = text_bytes[i]
        if b < 0x80:
            width = 1
        elif b < 0xC0:
            # stray continuation byte; treat as 1 to keep the walk linear
            width = 1
        elif b < 0xE0:
            width = 2
        elif b < 0xF0:
            width = 3
        else:
            width = 4
        for k in range(width):
            if i + k < n:
                byte_to_char[i + k] = char_idx
        i += width
        char_idx += 1
    byte_to_char[n] = char_idx
    return byte_to_char
