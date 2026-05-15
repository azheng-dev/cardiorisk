"""Tests for the mock NLI verifier (DeBERTa wrapper is dep-gated)."""

from __future__ import annotations

import pytest

from cardiorisk.rag.generation.nli import (
    BaseNLIVerifier,
    EntailmentResult,
    MockNLIVerifier,
    get_nli_verifier,
)


def test_mock_satisfies_protocol() -> None:
    v: BaseNLIVerifier = MockNLIVerifier()
    assert isinstance(v, BaseNLIVerifier)
    assert v.name == "mock-token-overlap"


def test_full_overlap_yields_entailment() -> None:
    res = MockNLIVerifier().entails(
        premise="Statins are first-line therapy for primary prevention.",
        hypothesis="Statins are first-line therapy for primary prevention.",
    )
    assert isinstance(res, EntailmentResult)
    assert res.entails
    assert res.p_entailment > 0.5
    assert abs(res.p_entailment + res.p_neutral + res.p_contradiction - 1.0) < 1e-3


def test_zero_overlap_yields_neutral() -> None:
    res = MockNLIVerifier().entails(
        premise="The dog ran across the field.",
        hypothesis="Quantum field theory unifies relativity and electromagnetism.",
    )
    assert not res.entails
    assert res.p_neutral > res.p_entailment


def test_partial_overlap_below_threshold() -> None:
    res = MockNLIVerifier().entails(
        premise="Statins are first-line therapy.",
        hypothesis="Statins are recommended for primary prevention only above 10% risk.",
    )
    assert not res.entails


def test_partial_overlap_above_threshold() -> None:
    res = MockNLIVerifier().entails(
        premise="Statins are first-line therapy for primary prevention above 10% risk.",
        hypothesis="Statins are first-line therapy for primary prevention.",
    )
    assert res.entails


def test_empty_hypothesis_returns_zero_entailment() -> None:
    res = MockNLIVerifier().entails("anything", "")
    assert res.p_entailment == 0.0
    assert not res.entails


def test_batch_returns_one_per_pair() -> None:
    pairs = [("a a a", "a"), ("x", "y")]
    out = MockNLIVerifier().entails_batch(pairs)
    assert len(out) == 2
    assert out[0].entails
    assert not out[1].entails


def test_get_nli_verifier_dispatches_mock() -> None:
    v = get_nli_verifier("mock")
    assert isinstance(v, MockNLIVerifier)


def test_get_nli_verifier_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown nli"):
        get_nli_verifier("transmogrifier")


def test_entailment_result_threshold_property_at_boundary() -> None:
    res = EntailmentResult(p_entailment=0.5, p_neutral=0.3, p_contradiction=0.2)
    assert res.entails


def test_mock_verifier_is_deterministic() -> None:
    v = MockNLIVerifier()
    a = v.entails("alpha beta gamma", "alpha beta")
    b = v.entails("alpha beta gamma", "alpha beta")
    assert a == b
