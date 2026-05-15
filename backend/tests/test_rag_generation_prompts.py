"""Tests for the prompt loader + mini-renderer."""

from __future__ import annotations

import pytest

from cardiorisk.rag.generation.prompts import (
    DEFAULT_PROMPT,
    PromptPassage,
    load_prompt,
    render_citation_prompt,
    render_prompt,
)


def _passage(chunk_id: str = "x:1", text: str = "Lorem ipsum.") -> PromptPassage:
    return PromptPassage(
        chunk_id=chunk_id,
        doc_id="docA",
        page_start=1,
        page_end=1,
        text=text,
    )


def test_default_prompt_loads_and_has_required_directives() -> None:
    text = load_prompt(DEFAULT_PROMPT)
    assert "[chunk_id" in text
    assert "REFUSE" in text
    assert "{{ question }}" in text
    assert "{% for passage in passages %}" in text


def test_load_prompt_rejects_unsuffixed_name() -> None:
    with pytest.raises(ValueError, match="must match"):
        load_prompt("citation_required.md")


def test_load_prompt_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.v1.md")


def test_render_prompt_substitutes_flat_variable() -> None:
    rendered = render_prompt("Hello {{ name }}!", name="world")
    assert rendered == "Hello world!"


def test_render_prompt_expands_for_loop() -> None:
    template = "{% for x in xs %}- {{ x.label }} = {{ x.value }}\n{% endfor %}"
    items = [
        type("Item", (), {"label": "a", "value": 1})(),
        type("Item", (), {"label": "b", "value": 2})(),
    ]
    rendered = render_prompt(template, xs=items)
    assert rendered == "- a = 1\n- b = 2\n"


def test_render_prompt_for_loop_supports_dict_items() -> None:
    template = "{% for x in xs %}{{ x.k }}|{% endfor %}"
    rendered = render_prompt(template, xs=[{"k": "alpha"}, {"k": "beta"}])
    assert rendered == "alpha|beta|"


def test_render_prompt_raises_on_unknown_variable() -> None:
    with pytest.raises(KeyError, match="missing_var"):
        render_prompt("{{ missing_var }}")


def test_render_prompt_raises_on_unparsed_token() -> None:
    template = "{{ a }} {% bad-syntax %}"
    with pytest.raises(ValueError, match="unparsed token"):
        render_prompt(template, a="x")


def test_render_citation_prompt_includes_chunk_id_in_passage_block() -> None:
    rendered = render_citation_prompt(
        question="What is the threshold?",
        passages=[_passage(chunk_id="docA:p1:c1", text="The threshold is 10%.")],
    )
    assert "[chunk_id=docA:p1:c1]" in rendered
    assert "(doc=docA, page=1-1)" in rendered
    assert "The threshold is 10%." in rendered
    assert "What is the threshold?" in rendered


def test_render_citation_prompt_with_multiple_passages_emits_each() -> None:
    rendered = render_citation_prompt(
        question="?",
        passages=[
            _passage(chunk_id="a:1", text="alpha"),
            _passage(chunk_id="b:1", text="beta"),
        ],
    )
    assert rendered.count("[chunk_id=") == 2
    assert "alpha" in rendered
    assert "beta" in rendered
