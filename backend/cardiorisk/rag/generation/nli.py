"""Natural-language-inference (NLI) verifier for citation grounding.

Two implementations behind one Protocol:

- :class:`MockNLIVerifier` — token-overlap heuristic. Emits a 3-class
  probability vector by counting hypothesis tokens that appear in the
  premise. Deterministic, dep-free; used by every unit test and the
  CI smoke step.
- :class:`DeBERTaNLIVerifier` — Hugging Face transformers wrapper
  around ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
  (the strongest open MNLI checkpoint as of mid-2026 and the one
  ADR-017 picked). Lazy import so the test suite doesn't pay the dep
  cost.

Output schema (:class:`EntailmentResult`) reports the full 3-class
probability so downstream callers can apply a tighter threshold than
``P(entailment) >= 0.5`` if needed (e.g. require ``P(contradiction) <
0.1`` as well). The default policy lives in :class:`CitationGenerator`.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

#: Default DeBERTa MNLI checkpoint. Picked by ADR-017 §"NLI verifier".
DEFAULT_DEBERTA_MODEL: Final[str] = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

#: Probability threshold for "the premise entails the hypothesis".
DEFAULT_ENTAILMENT_THRESHOLD: Final[float] = 0.5


@dataclass(frozen=True)
class EntailmentResult:
    """3-class probability over MNLI labels.

    Sum of the three probabilities is 1.0 within float epsilon. The
    ``entails`` convenience property uses
    :data:`DEFAULT_ENTAILMENT_THRESHOLD`; callers can compute their
    own threshold from the raw fields.
    """

    p_entailment: float
    p_neutral: float
    p_contradiction: float

    @property
    def entails(self) -> bool:
        return self.p_entailment >= DEFAULT_ENTAILMENT_THRESHOLD


@runtime_checkable
class BaseNLIVerifier(Protocol):
    """Pluggable NLI verifier.

    Implementations must be deterministic (the eval harness relies on
    byte-stable verdicts).
    """

    name: str

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Return the 3-class probability for one (premise, hypothesis) pair."""
        ...

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        """Batched form; default loops over single calls."""
        ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


class MockNLIVerifier:
    """Token-overlap NLI fallback used by unit tests + CI smoke.

    Algorithm:

    - Compute the set of lowercase alphanumeric tokens in premise and
      hypothesis.
    - ``coverage = |hypothesis ∩ premise| / max(|hypothesis|, 1)``.
    - Map ``coverage`` to a 3-class probability via a fixed lookup
      that biases toward ``neutral`` at low coverage and ``entailment``
      at high coverage. Negation and contradictions are not modelled
      (the production verifier is); the mock exists to exercise the
      generator pipeline, not to do real NLI.

    Calibration: ``coverage >= 0.6`` produces ``p_entailment > 0.5``,
    which means a Mock-based smoke run will mark "the answer
    paraphrased the passage" as entailed and "the answer invented
    text" as neutral. This is enough to catch wiring regressions but
    will NOT catch a real LLM hallucination.
    """

    name: str = "mock-token-overlap"

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        h_tokens = _tokenise(hypothesis)
        if not h_tokens:
            return EntailmentResult(0.0, 1.0, 0.0)
        p_tokens = _tokenise(premise)
        coverage = len(h_tokens & p_tokens) / float(len(h_tokens))
        # Smooth ramp: at coverage=0.0 -> (0.05, 0.9, 0.05); at 1.0 ->
        # (0.92, 0.05, 0.03). Linear in coverage between.
        p_entail = 0.05 + 0.87 * coverage
        p_neutral = 0.9 - 0.85 * coverage
        p_contra = max(0.0, 1.0 - p_entail - p_neutral)
        return EntailmentResult(
            p_entailment=round(float(p_entail), 4),
            p_neutral=round(float(p_neutral), 4),
            p_contradiction=round(float(p_contra), 4),
        )

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        return [self.entails(p, h) for p, h in pairs]


class DeBERTaNLIVerifier:
    """``DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` wrapper.

    Uses :mod:`transformers` ``AutoModelForSequenceClassification`` +
    ``AutoTokenizer`` directly rather than the ``pipeline()`` factory
    so we control batching + ``torch.inference_mode()`` (sentence-
    transformers stack pulls torch unconditionally; reused here).

    The MNLI label order on this checkpoint is
    ``[entailment, neutral, contradiction]``. We re-check by reading
    ``model.config.id2label`` at construction so a future checkpoint
    rename does not silently swap entailment for contradiction.
    """

    name: str = "deberta-v3-large-mnli"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DEBERTA_MODEL,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        with contextlib.suppress(AttributeError):
            self._model.eval()
        # Lock down label order: id2label may be {0:'entailment',
        # 1:'neutral', 2:'contradiction'} on most MNLI heads but we
        # never want to silently misread it.
        id2label: dict[int, str] = {
            int(k): v.lower() for k, v in self._model.config.id2label.items()
        }
        try:
            self._idx_entail = next(i for i, v in id2label.items() if "entail" in v)
            self._idx_neutral = next(i for i, v in id2label.items() if "neutral" in v)
            self._idx_contra = next(i for i, v in id2label.items() if "contra" in v)
        except StopIteration as exc:
            raise RuntimeError(
                f"NLI checkpoint {model_name!r} has unexpected id2label {id2label!r}; "
                "expected entailment / neutral / contradiction"
            ) from exc
        self._batch_size = batch_size
        self._torch = torch

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        return self.entails_batch([(premise, hypothesis)])[0]

    def entails_batch(self, pairs: Sequence[tuple[str, str]]) -> list[EntailmentResult]:
        if not pairs:
            return []
        torch = self._torch
        out: list[EntailmentResult] = []
        for start in range(0, len(pairs), self._batch_size):
            chunk = pairs[start : start + self._batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]
            tok = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits: Any = self._model(**tok).logits
                probs: Any = torch.softmax(logits, dim=-1)
            arr = probs.detach().cpu().numpy()
            for row in arr:
                out.append(
                    EntailmentResult(
                        p_entailment=float(row[self._idx_entail]),
                        p_neutral=float(row[self._idx_neutral]),
                        p_contradiction=float(row[self._idx_contra]),
                    )
                )
        return out


def get_nli_verifier(name: str, **kwargs: Any) -> BaseNLIVerifier:
    """Factory mirroring :func:`embed.get_embedder`.

    Names accepted:

    - ``mock`` — :class:`MockNLIVerifier` (no creds, no weights).
    - ``deberta`` / ``deberta-v3-mnli`` — :class:`DeBERTaNLIVerifier`
      (lazy-loads ~750 MB of weights on first use).
    """
    if name == "mock":
        return MockNLIVerifier(**kwargs)
    if name in ("deberta", "deberta-v3-mnli", "deberta-v3-large-mnli"):
        return DeBERTaNLIVerifier(**kwargs)
    raise ValueError(f"unknown nli verifier {name!r}; known: mock, deberta")
