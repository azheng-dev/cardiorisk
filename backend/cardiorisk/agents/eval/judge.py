"""LLM-as-judge layer for the Phase-6 agent eval (ADR-019).

What it does
============
For every case the eval orchestrator runs, the judge reads the
*generated letter draft* plus the *expected recommendation family*
and answers two questions:

1. ``letter_quality`` (1-5 Likert): is the draft clinically coherent,
   on-topic, and free of contradictory advice?
2. ``recommendation_alignment`` (1-5 Likert): does the draft's
   recommended action match the expected family (statin / lifestyle /
   refer / refuse / etc.)?

Both scores collapse to pass/fail at threshold ``>= 4`` and are
rolled up into ``llm_judge_pass_rate``. The 1-5 Likert gives the
diagnostics (we keep the score distribution + per-tag mean), but the
gate is binary.

Why a separate file and not the scorer
======================================
The judge is the only metric that requires a network round-trip per
case. Keeping it isolated lets us:

- Run a CI eval with a :class:`MockJudge` that scores ``5`` if the
  draft contains the expected family keyword (mirrors the scorer's
  keyword logic) and ``1`` otherwise. Cheap, deterministic.
- Swap to a :class:`GeminiJudge` locally (with ``GEMINI_API_KEY``)
  for the actual eval run, without touching the orchestrator.
- Defer LLM-judge calibration to a future research note: the keyword
  scorer + the LLM judge are two independent signals; agreement
  between them is itself a signal.

The judge interface mirrors :class:`BaseLLMClient` deliberately —
both have a ``name`` + a callable + a :class:`UsageTotals`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from cardiorisk.rag.generation.llm import (
    UsageTotals,
    _approx_tokens,
    estimate_cost_usd,
)

from .scorer import RECOMMENDATION_FAMILY_KEYWORDS

#: A draft is considered "passing" when both Likert scores are at
#: least this value.
JUDGE_PASS_THRESHOLD: Final[int] = 4
#: Default Likert range used by every judge implementation.
LIKERT_MIN: Final[int] = 1
LIKERT_MAX: Final[int] = 5


@dataclass(frozen=True)
class JudgeScore:
    """One judge's verdict on one case."""

    case_id: str
    letter_quality: int  # 1..5
    recommendation_alignment: int  # 1..5
    rationale: str
    passed: bool  # both scores >= JUDGE_PASS_THRESHOLD

    @classmethod
    def from_raw(
        cls,
        *,
        case_id: str,
        letter_quality: int,
        recommendation_alignment: int,
        rationale: str,
    ) -> JudgeScore:
        lq = max(LIKERT_MIN, min(LIKERT_MAX, int(letter_quality)))
        ra = max(LIKERT_MIN, min(LIKERT_MAX, int(recommendation_alignment)))
        return cls(
            case_id=case_id,
            letter_quality=lq,
            recommendation_alignment=ra,
            rationale=rationale,
            passed=lq >= JUDGE_PASS_THRESHOLD and ra >= JUDGE_PASS_THRESHOLD,
        )


@runtime_checkable
class BaseJudge(Protocol):
    """Shape every judge implementation exposes."""

    name: str
    usage: UsageTotals

    def score(
        self,
        *,
        case_id: str,
        letter_draft: str,
        expected_recommendation_family: str,
        tag: str,
    ) -> JudgeScore:
        """Return one :class:`JudgeScore`."""
        ...


# ---------------------------------------------------------------------------
# Mock judge.
# ---------------------------------------------------------------------------
class MockJudge:
    """Deterministic keyword-based judge. Used in every CI step.

    Scoring rule:

    - ``recommendation_alignment``: 5 if any keyword from the expected
      family is in the draft (case-insensitive), 1 otherwise.
    - ``letter_quality``: 5 if the draft is non-empty and at least
      30 words, 3 if non-empty but shorter, 1 if empty.

    This mirrors the scorer's heuristic so the mock judge is a clean
    floor — it doesn't add information beyond what the scorer already
    extracts, but it exercises the entire judge pipeline end-to-end
    (including the per-case JSON serialisation and the aggregate
    pass-rate calc).
    """

    name: str = "mock-judge"

    def __init__(self) -> None:
        self.usage = UsageTotals()

    def score(
        self,
        *,
        case_id: str,
        letter_draft: str,
        expected_recommendation_family: str,
        tag: str,
    ) -> JudgeScore:
        del tag
        draft = letter_draft or ""
        draft_lower = draft.lower()
        keywords = RECOMMENDATION_FAMILY_KEYWORDS.get(expected_recommendation_family, ())
        keyword_hit = any(k.lower() in draft_lower for k in keywords)
        recommendation_alignment = 5 if keyword_hit else 1
        n_words = len(draft.split())
        if n_words == 0:
            letter_quality = 1
        elif n_words < 30:
            letter_quality = 3
        else:
            letter_quality = 5
        rationale = (
            f"keyword_hit={keyword_hit} n_words={n_words} family={expected_recommendation_family}"
        )
        # Token-count + cost = 0 by definition.
        self.usage.add(input_tokens=len(draft.split()), output_tokens=4, cost_usd=0.0)
        return JudgeScore.from_raw(
            case_id=case_id,
            letter_quality=letter_quality,
            recommendation_alignment=recommendation_alignment,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Prompt template + JSON-shaped parser shared by live judges.
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE: Final[str] = """You are a clinical-reasoning grader.

You receive ONE letter draft generated by an AI co-pilot for a primary-care
clinician, plus the expected *recommendation family* the draft should land in.
Your job is to grade the draft on two axes, each on a 1-to-5 Likert scale.

Axes:
1. `letter_quality` (1-5): is the draft clinically coherent, on-topic,
   free of contradictory advice, and written in a register a GP would
   accept? 5 = excellent, 1 = unusable.
2. `recommendation_alignment` (1-5): does the draft's recommended action
   match the expected family below? 5 = clearly matches, 1 = clearly
   does not match (or no recommendation at all when one was expected).

Expected recommendation family: `{family}`
Allowed family vocabulary: lifestyle_only, lifestyle_plus_review,
statin_consider, statin_plus_bp, statin_plus_bp_plus_referral,
specialist_referral_urgent, refusal_no_recommendation.

For the `refusal_no_recommendation` family: an acceptable draft is a
clear refusal ("I do not have the supporting guidance...") and scores
5/5 on both axes if it refuses cleanly without inventing reasoning.

Letter draft:
\"\"\"
{draft}
\"\"\"

Reply with ONE JSON object and nothing else. Use this exact schema:

{{
  "letter_quality": <int 1-5>,
  "recommendation_alignment": <int 1-5>,
  "rationale": "<one short sentence>"
}}
"""


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judge_json(raw: str, *, case_id: str) -> tuple[int, int, str]:
    """Extract the first JSON object from a live judge's response and
    return (letter_quality, recommendation_alignment, rationale).

    Defensive: live LLMs occasionally wrap JSON in prose / code fences.
    We grep out the first ``{...}`` block and parse that. If parsing
    fails or fields are missing, we conservatively return ``(1, 1,
    error)`` so the case scores as a fail. The orchestrator surfaces
    the error in the per-case JSON.
    """
    match = _JSON_OBJECT_RE.search(raw or "")
    if match is None:
        return 1, 1, f"judge_parse_failed:no_json:case={case_id}"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return 1, 1, f"judge_parse_failed:json_decode:{exc}:case={case_id}"
    try:
        lq = int(payload["letter_quality"])
        ra = int(payload["recommendation_alignment"])
        rationale = str(payload.get("rationale", ""))
    except (KeyError, TypeError, ValueError) as exc:
        return 1, 1, f"judge_parse_failed:bad_field:{exc}:case={case_id}"
    return lq, ra, rationale


# ---------------------------------------------------------------------------
# Live judges (lazy-imported SDKs, mirror the LLM client classes).
# ---------------------------------------------------------------------------
class GeminiJudge:
    """Google Gemini judge. Reads ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.

    Default model: ``gemini-2.5-flash`` — same model the generator
    uses, on purpose. The judge is *not* assumed to be more capable
    than the generator; it's a second sample with a different prompt
    that the orchestrator compares against. ADR-019 discusses the
    trade-off.
    """

    name: str = "gemini-judge"

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} (or GOOGLE_API_KEY) not set; "
                "GeminiJudge requires a Google AI Studio key"
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.usage = UsageTotals()

    def score(
        self,
        *,
        case_id: str,
        letter_draft: str,
        expected_recommendation_family: str,
        tag: str,
    ) -> JudgeScore:
        del tag
        from google.genai import types as gtypes

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            family=expected_recommendation_family, draft=letter_draft or ""
        )
        config = gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=512)
        resp: Any = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        text = str(getattr(resp, "text", "") or "")
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            input_tokens = int(getattr(meta, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
        else:
            input_tokens = _approx_tokens(prompt)
            output_tokens = _approx_tokens(text)
        self.usage.add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(self._model, input_tokens, output_tokens),
        )
        lq, ra, rationale = _parse_judge_json(text, case_id=case_id)
        return JudgeScore.from_raw(
            case_id=case_id,
            letter_quality=lq,
            recommendation_alignment=ra,
            rationale=rationale,
        )


class GroqJudge:
    """Groq-hosted Llama-3.3-70B judge. Reads ``GROQ_API_KEY``.

    Opt-in second live judge. Useful for a judge-vs-judge agreement
    check (Gemini-judge agrees with Groq-judge => higher confidence
    that the letter is actually good).
    """

    name: str = "groq-judge"

    def __init__(
        self,
        *,
        model: str = "llama-3.3-70b-versatile",
        api_key_env: str = "GROQ_API_KEY",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {api_key_env!r} not set; GroqJudge requires a Groq API key"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self._model = model
        self.usage = UsageTotals()

    def score(
        self,
        *,
        case_id: str,
        letter_draft: str,
        expected_recommendation_family: str,
        tag: str,
    ) -> JudgeScore:
        del tag
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            family=expected_recommendation_family, draft=letter_draft or ""
        )
        resp: Any = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = str(resp.choices[0].message.content or "")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            input_tokens = _approx_tokens(prompt)
            output_tokens = _approx_tokens(text)
        self.usage.add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost_usd(self._model, input_tokens, output_tokens),
        )
        lq, ra, rationale = _parse_judge_json(text, case_id=case_id)
        return JudgeScore.from_raw(
            case_id=case_id,
            letter_quality=lq,
            recommendation_alignment=ra,
            rationale=rationale,
        )


def get_judge(name: str, **kwargs: Any) -> BaseJudge:
    """Factory mirroring :func:`get_llm_client`.

    Names: ``mock``, ``gemini``, ``groq``.
    """
    if name == "mock":
        return MockJudge()
    if name in ("gemini", "google"):
        return GeminiJudge(**kwargs)
    if name == "groq":
        return GroqJudge(**kwargs)
    raise ValueError(f"unknown judge {name!r}; known: mock, gemini, groq")


# ---------------------------------------------------------------------------
# Aggregate helpers.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JudgeAggregate:
    """Cross-case roll-up of one judge's verdicts."""

    judge_name: str
    n_cases: int
    pass_rate: float
    mean_letter_quality: float
    mean_recommendation_alignment: float
    per_tag: dict[str, dict[str, float | int]]

    @classmethod
    def from_scores(
        cls, judge_name: str, scores: Sequence[JudgeScore], tags: Sequence[str]
    ) -> JudgeAggregate:
        n = len(scores)
        if n == 0:
            return cls(
                judge_name=judge_name,
                n_cases=0,
                pass_rate=0.0,
                mean_letter_quality=0.0,
                mean_recommendation_alignment=0.0,
                per_tag={},
            )
        if len(tags) != n:
            raise ValueError("scores and tags must be the same length")
        per_tag_groups: dict[str, list[JudgeScore]] = {}
        for tag, s in zip(tags, scores, strict=True):
            per_tag_groups.setdefault(tag, []).append(s)
        per_tag = {
            tag: {
                "n": len(group),
                "pass_rate": sum(1 for x in group if x.passed) / len(group),
                "mean_letter_quality": sum(x.letter_quality for x in group) / len(group),
                "mean_recommendation_alignment": sum(x.recommendation_alignment for x in group)
                / len(group),
            }
            for tag, group in sorted(per_tag_groups.items())
        }
        return cls(
            judge_name=judge_name,
            n_cases=n,
            pass_rate=sum(1 for x in scores if x.passed) / n,
            mean_letter_quality=sum(x.letter_quality for x in scores) / n,
            mean_recommendation_alignment=sum(x.recommendation_alignment for x in scores) / n,
            per_tag=per_tag,
        )


__all__ = [
    "JUDGE_PASS_THRESHOLD",
    "BaseJudge",
    "GeminiJudge",
    "GroqJudge",
    "JudgeAggregate",
    "JudgeScore",
    "MockJudge",
    "get_judge",
]
