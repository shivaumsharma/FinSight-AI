"""
Unit tests for the self-evaluation scorers (app/evaluation/) -- the
grounding/citation checks that went through a documented v1 -> v2
rewrite because the v1 verbatim-substring approach was structurally
unsatisfiable against a prompt that explicitly asks the model to
paraphrase. These tests pin down v2's actual (word-overlap) behavior.
"""

import pytest

from app.evaluation.citation_evaluator import CitationEvaluator
from app.evaluation.grounding_validator import GroundingValidator
from app.evaluation.report_validator import ReportValidator
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.scorer import ScoreAggregator


# ---------------------------------------------------------------- grounding

def test_grounding_validator_supports_a_paraphrased_sentence():
    research_summary = "Revenue growth accelerated to 25 percent driven by strong enterprise demand."
    report = "The company's revenue growth accelerated significantly due to enterprise demand."
    result = GroundingValidator().validate(research_summary, report)
    assert result.supported_claims == 1
    assert result.unsupported_claims == 0
    assert result.grounding_score == 100.0


def test_grounding_validator_flags_a_fabricated_sentence():
    research_summary = "Revenue growth accelerated to 25 percent driven by strong enterprise demand."
    report = "The company announced a surprise merger with a rival aerospace manufacturer."
    result = GroundingValidator().validate(research_summary, report)
    assert result.unsupported_claims == 1
    assert result.supported_claims == 0


def test_grounding_validator_ignores_short_fragments():
    # Below the 20-character minimum -- too short to meaningfully
    # score either way, so it must not count as unsupported filler.
    result = GroundingValidator().validate("Revenue growth accelerated.", "Yes.")
    assert result.supported_claims == 0
    assert result.unsupported_claims == 0


def test_grounding_validator_ignores_a_bare_section_heading():
    # "Market and Earnings Analysis" (29 chars) is the one required
    # heading long enough to clear the 20-char length filter on its
    # own, so it's the one that actually exercises _is_heading rather
    # than being screened out by length alone first.
    research_summary = "Revenue growth accelerated on strong enterprise demand."
    report = "Revenue grew significantly across all segments this quarter. Market and Earnings Analysis"
    result = GroundingValidator().validate(research_summary, report)
    # Only the first (real) sentence should be scored; the trailing
    # bare heading must be skipped, not counted as unsupported.
    assert result.supported_claims + result.unsupported_claims == 1


def test_grounding_validator_stemmer_collapses_inflection():
    # "operating"/"operations" and "declining"/"decline" should overlap
    # via the crude suffix-stripping stemmer, not be treated as
    # unrelated tokens just because a paraphrasing model changed tense.
    research_summary = "Operating margins are declining due to input cost pressure."
    report = "The company's operations margin decline reflects rising input costs today."
    result = GroundingValidator().validate(research_summary, report)
    assert result.supported_claims == 1


# ---------------------------------------------------------------- citations

def test_citation_evaluator_detects_explicit_evidence_tag():
    citations = [{"text": "some evidence text that is not otherwise quoted"}]
    report = "Enterprise adoption accelerated this quarter [Evidence 1]."
    result = CitationEvaluator().evaluate(citations, report)
    assert result.citations_used == 1
    assert result.citation_coverage == 100.0


def test_citation_evaluator_falls_back_to_paraphrase_overlap():
    citations = [{"text": "management raised full year guidance citing strong enterprise demand"}]
    report = "Management raised its full-year guidance, citing strong enterprise demand across the business."
    result = CitationEvaluator().evaluate(citations, report)
    assert result.citations_used == 1


def test_citation_evaluator_marks_unused_citation_as_unused():
    citations = [{"text": "a completely unrelated statement about litigation risk"}]
    report = "Revenue grew twelve percent this quarter."
    result = CitationEvaluator().evaluate(citations, report)
    assert result.citations_used == 0
    assert result.citation_coverage == 0.0


def test_citation_evaluator_news_uses_news_tag_convention():
    articles = [{"headline": "Company faces antitrust probe", "summary": "Regulators opened an inquiry."}]
    report = "The business faces regulatory scrutiny [News 1]."
    result = CitationEvaluator().evaluate_news(articles, report)
    assert result.citations_used == 1


def test_citation_evaluator_handles_empty_citation_list():
    result = CitationEvaluator().evaluate([], "Any report text.")
    assert result.citations_available == 0
    assert result.citation_coverage == 0


# ---------------------------------------------------------------- completeness
#
# validate() takes the per-section {name: text} dict
# (context.report_data["narrative"]), not the flattened report string
# -- see report_validator.py's own module docstring for why checking
# the flattened string (report_tool.py always writes every section's
# heading regardless of whether the narrative actually generated)
# could never detect a missing section at all.

def test_report_validator_flags_no_missing_sections_when_all_present():
    narrative = {
        "Executive Summary": "Real content.",
        "Business Analysis": "Real content.",
        "Market and Earnings Analysis": "Real content.",
        "Risk Analysis": "Real content.",
        "Investment Thesis": "Real content.",
    }
    result = ReportValidator().validate(narrative)
    assert result.complete is True
    assert result.missing_sections == []
    assert result.completeness_score == 100.0


def test_report_validator_flags_missing_sections():
    # Only 2 of the 5 required sections are present in the dict at all.
    narrative = {"Executive Summary": "Real content.", "Risk Analysis": "Real content."}
    result = ReportValidator().validate(narrative)
    assert result.complete is False
    assert "Business Analysis" in result.missing_sections
    assert len(result.missing_sections) == 3
    assert result.completeness_score == pytest.approx(2 / 5 * 100)


def test_report_validator_flags_sections_present_only_as_the_fallback_placeholder():
    """Regression test for the completeness_score-always-100 bug: every
    key exists (report_tool.py's f-string always writes a heading for
    every NARRATIVE_SECTIONS entry), but the VALUE is narrative_builder's
    own "model failed to produce this section" placeholder, not real
    content. Before the fix, checking the flattened string for the
    section NAME as a substring would find "Executive Summary" (the
    heading) regardless of what followed it, and this would have wrongly
    validated as complete=True, completeness_score=100.0."""
    from app.reporting.narrative_builder import NARRATIVE_SECTION_FALLBACK

    narrative = {
        "Executive Summary": NARRATIVE_SECTION_FALLBACK,
        "Business Analysis": "Real content.",
        "Market and Earnings Analysis": NARRATIVE_SECTION_FALLBACK,
        "Risk Analysis": "Real content.",
        "Investment Thesis": "Real content.",
    }
    result = ReportValidator().validate(narrative)
    assert result.complete is False
    assert result.missing_sections == ["Executive Summary", "Market and Earnings Analysis"]
    assert result.completeness_score == pytest.approx(3 / 5 * 100)


def test_report_validator_flags_completely_failed_generation():
    # Every section is the fallback placeholder -- the exact scenario
    # that used to validate as a perfect 100.0 score.
    from app.reporting.narrative_builder import NARRATIVE_SECTION_FALLBACK, NARRATIVE_SECTIONS

    narrative = {section: NARRATIVE_SECTION_FALLBACK for section in NARRATIVE_SECTIONS}
    result = ReportValidator().validate(narrative)
    assert result.complete is False
    assert result.completeness_score == 0.0
    assert len(result.missing_sections) == 5


def test_report_validator_handles_a_none_or_empty_narrative_without_raising():
    result = ReportValidator().validate(None)
    assert result.complete is False
    assert result.completeness_score == 0.0

    result = ReportValidator().validate({})
    assert result.complete is False
    assert result.completeness_score == 0.0


# ---------------------------------------------------------------- retrieval + aggregate

def test_retrieval_evaluator_scores_full_coverage_at_5_chunks():
    result = RetrievalEvaluator().evaluate(retrieved_chunks=list(range(20)), reranked_chunks=["a"] * 5)
    assert result.retrieval_score == 100.0


def test_retrieval_evaluator_scores_partial_coverage_below_5_chunks():
    result = RetrievalEvaluator().evaluate(retrieved_chunks=list(range(20)), reranked_chunks=["a"] * 2)
    assert result.retrieval_score == pytest.approx(40.0)


def test_score_aggregator_weights_match_documented_split():
    class _Score:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    overall = ScoreAggregator().aggregate(
        grounding=_Score(grounding_score=100.0),
        retrieval=_Score(retrieval_score=0.0),
        citations=_Score(citation_coverage=0.0),
        report=_Score(completeness_score=0.0),
    )
    assert overall == pytest.approx(40.0)  # grounding is weighted 40%
