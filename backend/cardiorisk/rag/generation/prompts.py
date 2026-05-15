"""File-backed prompt loader for the citation-mandatory generator.

Templates live alongside this module under ``./prompts/``. The file
name carries the version: ``citation_required.v1.md`` is the v1
template, and a future amendment that changes wording ships as
``citation_required.v2.md`` rather than mutating v1 in place.

Why a hand-rolled mini-renderer rather than ``jinja2``: the templates
are short, the substitutions are trivial (``{{ question }}`` and
``{% for passage in passages %}``), and dragging in ``jinja2`` for two
substitution forms is wildly disproportionate. The renderer rejects
any template token it does not understand, so a future template that
needs proper Jinja will fail loudly rather than silently passing
unrendered placeholders to the LLM.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

PROMPT_DIR: Final[Path] = Path(__file__).resolve().parent / "prompts"

#: Default citation prompt version. Bump when ``ADR-017`` decides to
#: cut a v2 prompt; the v1 template stays committed for reproducibility.
DEFAULT_PROMPT: Final[str] = "citation_required.v1.md"

_VAR_RE = re.compile(r"\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_FOR_RE = re.compile(
    r"\{%\s*for\s+(?P<var>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"\s+in\s+(?P<iterable>[a-zA-Z_][a-zA-Z0-9_]*)\s*%\}"
    r"(?P<body>.*?)"
    r"\{%\s*endfor\s*%\}",
    re.DOTALL,
)
# After loop expansion the body uses {{ var.attr }} access; matched here.
_VAR_ATTR_RE = re.compile(
    r"\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<attr>[a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
)


@dataclass(frozen=True)
class PromptPassage:
    """One retrieved passage shaped for the prompt.

    Mirrors :class:`cardiorisk.rag.retrieval.pipeline.RetrievedChunk`
    but only the fields the prompt template needs. The generator
    constructs these from the retrieval results immediately before
    rendering.
    """

    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


@lru_cache(maxsize=8)
def load_prompt(name: str = DEFAULT_PROMPT) -> str:
    """Read a prompt template by file name.

    Args:
        name: File name under ``./prompts/``. Must include the
            ``.vX.md`` suffix; the loader rejects anything else so a
            stale callsite cannot accidentally mute the version
            check.

    Returns:
        Raw template text, unrendered.

    Raises:
        FileNotFoundError: if the named template does not exist.
        ValueError: if the file name lacks a ``.vN.md`` suffix.
    """
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*\.v[0-9]+\.md", name):
        raise ValueError(
            f"prompt name {name!r} must match '<slug>.v<int>.md' (e.g. 'citation_required.v1.md')"
        )
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _expand_for(template: str, context: Mapping[str, Any]) -> str:
    """Expand ``{% for x in xs %}...{% endfor %}`` blocks once.

    Nested loops are not supported — the prompt grammar is small on
    purpose. The match is non-greedy so multiple sibling loops in one
    template each expand independently.
    """

    def _expand(match: re.Match[str]) -> str:
        var = match.group("var")
        iterable_name = match.group("iterable")
        body = match.group("body")
        if iterable_name not in context:
            raise KeyError(f"prompt iterable {iterable_name!r} not in context")
        items: Iterable[Any] = context[iterable_name]
        out: list[str] = []
        for item in items:
            out.append(_render_loop_body(body, var, item))
        return "".join(out)

    return _FOR_RE.sub(_expand, template)


def _render_loop_body(body: str, var: str, item: Any) -> str:
    """Substitute ``{{ var.attr }}`` against one loop item.

    Pulled out of the closure so ruff's B023 (function definition does
    not bind loop variable) check does not flag the lambda; the
    function's parameters explicitly capture both ``var`` and ``item``.
    """
    return _VAR_ATTR_RE.sub(
        lambda m: _resolve_attr(m.group("name"), m.group("attr"), var, item),
        body,
    )


def _resolve_attr(name: str, attr: str, loop_var: str, item: Any) -> str:
    if name != loop_var:
        return f"{{{{ {name}.{attr} }}}}"  # leave un-rendered; will fail below
    if hasattr(item, attr):
        return str(getattr(item, attr))
    if isinstance(item, Mapping) and attr in item:
        return str(item[attr])
    raise AttributeError(f"prompt loop item {type(item).__name__} has no attribute {attr!r}")


def render_prompt(template: str, **context: Any) -> str:
    """Render a template with the given context.

    Supports two grammar forms only:

    - ``{{ name }}`` — flat substitution from ``context``.
    - ``{% for var in iterable %}...{{ var.attr }}...{% endfor %}``
      — single-level loop, attribute-access inside the body.

    Anything else raises :class:`ValueError`. The strictness is
    deliberate: the LLM's reliability depends on the prompt rendering
    cleanly, so silent template bugs are not allowed to propagate.
    """
    expanded = _expand_for(template, context)

    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in context:
            raise KeyError(f"prompt variable {name!r} not in context")
        return str(context[name])

    rendered = _VAR_RE.sub(_sub, expanded)

    leftover = re.search(r"\{\{|\}\}|\{%|%\}", rendered)
    if leftover:
        raise ValueError(
            f"prompt rendering left an unparsed token at offset {leftover.start()}: "
            f"{rendered[leftover.start() : leftover.start() + 30]!r}"
        )
    return rendered


def render_citation_prompt(
    *,
    question: str,
    passages: list[PromptPassage],
    template_name: str = DEFAULT_PROMPT,
) -> str:
    """Convenience wrapper for the citation prompt."""
    template = load_prompt(template_name)
    return render_prompt(template, question=question, passages=passages)
