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

def test_report_validator_flags_no_missing_sections_when_all_present():
    report = "\n".join([
        "# Executive Summary\ntext", "# Business Analysis\ntext",
        "# Market and Earnings Analysis\ntext", "# Risk Analysis\ntext",
        "# Investment Thesis\ntext",
    ])
    result = ReportValidator().validate(report)
    assert result.complete is True
    assert result.missing_sections == []
    assert result.completeness_score == 100.0


def test_report_validator_flags_missing_sections():
    # Only 2 of the 5 required sections are present.
    report = "# Executive Summary\ntext\n# Risk Analysis\ntext"
    result = ReportValidator().validate(report)
    assert result.complete is False
    assert "Business Analysis" in result.missing_sections
    assert len(result.missing_sections) == 3
    assert result.completeness_score == pytest.approx(2 / 5 * 100)


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
