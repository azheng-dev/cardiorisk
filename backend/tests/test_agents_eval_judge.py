"""Tests for the Phase-6 LLM-judge layer.

Covers:

- MockJudge keyword logic + Likert clamping.
- The base protocol surface.
- Factory dispatch.
- Live judges' API-key guards (no network calls).
- The aggregate roll-up.
- The JSON-extraction parser used by live judges.
"""

from __future__ import annotations

import pytest

from cardiorisk.agents.eval.judge import (
    JUDGE_PASS_THRESHOLD,
    BaseJudge,
    GeminiJudge,
    GroqJudge,
    JudgeAggregate,
    JudgeScore,
    MockJudge,
    _parse_judge_json,
    get_judge,
)
from cardiorisk.agents.eval.scorer import RECOMMENDATION_FAMILY_KEYWORDS


class TestJudgeScore:
    def test_from_raw_clamps_to_likert_range(self) -> None:
        s = JudgeScore.from_raw(
            case_id="a001",
            letter_quality=99,
            recommendation_alignment=-3,
            rationale="r",
        )
        assert s.letter_quality == 5
        assert s.recommendation_alignment == 1

    def test_passed_when_both_axes_meet_threshold(self) -> None:
        ok = JudgeScore.from_raw(
            case_id="a001",
            letter_quality=JUDGE_PASS_THRESHOLD,
            recommendation_alignment=JUDGE_PASS_THRESHOLD,
            rationale="r",
        )
        not_ok = JudgeScore.from_raw(
            case_id="a002",
            letter_quality=JUDGE_PASS_THRESHOLD,
            recommendation_alignment=JUDGE_PASS_THRESHOLD - 1,
            rationale="r",
        )
        assert ok.passed
        assert not not_ok.passed


class TestMockJudge:
    def _long_draft(self, family: str) -> str:
        keyword = RECOMMENDATION_FAMILY_KEYWORDS[family][0]
        # > 30 words triggers the 5/5 letter-quality branch.
        body = " ".join(["word"] * 31)
        return f"{keyword} {body}"

    def test_protocol(self) -> None:
        judge: BaseJudge = MockJudge()
        assert isinstance(judge, BaseJudge)
        assert judge.name == "mock-judge"

    def test_keyword_hit_scores_5(self) -> None:
        judge = MockJudge()
        s = judge.score(
            case_id="a001",
            letter_draft=self._long_draft("statin_plus_bp"),
            expected_recommendation_family="statin_plus_bp",
            tag="high_risk",
        )
        assert s.recommendation_alignment == 5
        assert s.letter_quality == 5
        assert s.passed

    def test_keyword_miss_scores_1(self) -> None:
        judge = MockJudge()
        s = judge.score(
            case_id="a002",
            letter_draft="Please continue current care for many many many words " * 5,
            expected_recommendation_family="statin_plus_bp",
            tag="high_risk",
        )
        assert s.recommendation_alignment == 1
        assert not s.passed

    def test_short_draft_scores_3_letter_quality(self) -> None:
        judge = MockJudge()
        keyword = RECOMMENDATION_FAMILY_KEYWORDS["lifestyle_only"][0]
        # Has the keyword but is shorter than 30 words.
        s = judge.score(
            case_id="a003",
            letter_draft=f"Recommend {keyword}.",
            expected_recommendation_family="lifestyle_only",
            tag="low_risk",
        )
        assert s.recommendation_alignment == 5
        assert s.letter_quality == 3
        assert not s.passed  # 3 < threshold (4)

    def test_empty_draft_scores_1_letter_quality(self) -> None:
        judge = MockJudge()
        s = judge.score(
            case_id="a004",
            letter_draft="",
            expected_recommendation_family="statin_plus_bp",
            tag="high_risk",
        )
        assert s.letter_quality == 1
        assert not s.passed

    def test_tracks_usage(self) -> None:
        judge = MockJudge()
        judge.score(
            case_id="a005",
            letter_draft="hi",
            expected_recommendation_family="lifestyle_only",
            tag="low_risk",
        )
        assert judge.usage.n_calls == 1
        assert judge.usage.cost_usd == 0.0


class TestFactory:
    def test_dispatches_mock(self) -> None:
        assert isinstance(get_judge("mock"), MockJudge)

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            get_judge("transmogrifier")


class TestLiveJudgesRequireKeys:
    def test_gemini_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            GeminiJudge()

    def test_groq_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            GroqJudge()


class TestJSONParser:
    def test_parses_clean_json(self) -> None:
        raw = '{"letter_quality": 4, "recommendation_alignment": 5, "rationale": "ok"}'
        lq, ra, rationale = _parse_judge_json(raw, case_id="a001")
        assert lq == 4
        assert ra == 5
        assert rationale == "ok"

    def test_strips_prose_around_json(self) -> None:
        raw = (
            'Here is my evaluation:\n\n{"letter_quality": 3, "recommendation_alignment": 4, '
            '"rationale": "passable"}\n\nThanks!'
        )
        lq, ra, _ = _parse_judge_json(raw, case_id="a001")
        assert lq == 3
        assert ra == 4

    def test_no_json_returns_fail(self) -> None:
        lq, ra, rationale = _parse_judge_json("no json here", case_id="a001")
        assert lq == 1
        assert ra == 1
        assert "judge_parse_failed" in rationale

    def test_missing_field_returns_fail(self) -> None:
        raw = '{"letter_quality": 4, "rationale": "x"}'
        lq, ra, rationale = _parse_judge_json(raw, case_id="a001")
        assert lq == 1
        assert ra == 1
        assert "judge_parse_failed" in rationale


class TestJudgeAggregate:
    def _score(self, *, passed: bool, lq: int = 4, ra: int = 4) -> JudgeScore:
        return JudgeScore(
            case_id="x",
            letter_quality=lq,
            recommendation_alignment=ra,
            rationale="r",
            passed=passed,
        )

    def test_empty_scores_returns_zeros(self) -> None:
        agg = JudgeAggregate.from_scores("mock-judge", [], [])
        assert agg.n_cases == 0
        assert agg.pass_rate == 0.0
        assert agg.per_tag == {}

    def test_aggregate_rolls_up_correctly(self) -> None:
        scores = [
            self._score(passed=True, lq=5, ra=5),
            self._score(passed=True, lq=4, ra=5),
            self._score(passed=False, lq=2, ra=3),
        ]
        tags = ["high_risk", "high_risk", "low_risk"]
        agg = JudgeAggregate.from_scores("mock-judge", scores, tags)
        assert agg.n_cases == 3
        assert agg.pass_rate == pytest.approx(2 / 3)
        assert agg.mean_letter_quality == pytest.approx((5 + 4 + 2) / 3)
        assert agg.mean_recommendation_alignment == pytest.approx((5 + 5 + 3) / 3)
        assert agg.per_tag["high_risk"]["pass_rate"] == pytest.approx(1.0)
        assert agg.per_tag["low_risk"]["pass_rate"] == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            JudgeAggregate.from_scores("mock", [self._score(passed=True)], ["a", "b"])
