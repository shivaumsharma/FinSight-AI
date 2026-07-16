"""
citation_evaluator.py

Measures evidence utilization.

Version 2
---------
v1 counted a citation as "used" only if the first 40 characters of
its raw chunk text appeared verbatim inside the generated report.
The report-writing prompt tells the model to summarize evidence in
its own words and explicitly forbids inventing/copying quotes, so a
verbatim 40-character match almost never happened -- citation
coverage was close to unsatisfiable regardless of how well-cited the
report actually was.

v2 counts a citation as "used" if either:
  a) the report explicitly references it by its evidence number,
     e.g. "[Evidence 3]" or "Evidence 3" (see narrative_builder.py,
     which asks the model to tag sentences this way), or
  b) a meaningful fraction of the citation's own content words show
     up in the report (paraphrase detection), as a fallback for
     reports that don't use the explicit tag.

v3 adds evaluate_news(), the same mechanism scoped to news articles
and "[News N]" tags instead of SEC evidence -- this is what
news_grounding_rate (evaluation_engine.py) is computed from.
"""

import re
from dataclasses import dataclass



_SUFFIXES = (
    "ational", "ization", "ations", "tions", "ing", "ally", "ment",
    "ness", "tion", "ence", "ance", "ives", "ive", "ers", "er", "ed",
    "es", "ly", "al", "s",
)


def _stem(word: str) -> str:
    """Same crude stemmer as grounding_validator.py -- see there for why."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass
class CitationResult:

    citations_available: int
    citations_used: int
    citation_coverage: float


class CitationEvaluator:

    # Fraction of a citation's own content words that must appear in
    # the report for it to count as "used" via paraphrase detection.
    OVERLAP_THRESHOLD = 0.3

    def evaluate(
        self,
        citations,
        generated_report: str,
    ) -> CitationResult:
        return self._evaluate(citations, generated_report, tag="evidence")

    def evaluate_news(
        self,
        news_articles,
        generated_report: str,
    ) -> CitationResult:
        """
        Same mechanism as evaluate(), scoped to news articles and
        "[News N]" tags (1-based within `news_articles`, matching the
        numbering narrative_builder.py's prompt uses) instead of SEC
        evidence citations.
        """
        citations = [
            {"text": f"{a['headline']} {a.get('summary', '')}"}
            for a in news_articles
        ]
        return self._evaluate(citations, generated_report, tag="news")

    def _evaluate(self, citations, generated_report: str, tag: str) -> CitationResult:

        report_norm = generated_report.lower()

        used = 0

        for i, citation in enumerate(citations, start=1):

            if self._explicitly_referenced(tag, i, report_norm):
                used += 1
                continue

            if self._paraphrase_overlap(citation["text"], report_norm) >= self.OVERLAP_THRESHOLD:
                used += 1

        total = len(citations)

        coverage = (

            used / total * 100

            if total

            else 0

        )

        return CitationResult(

            citations_available=total,

            citations_used=used,

            citation_coverage=round(coverage, 2),

        )

    # -----------------------------

    def _explicitly_referenced(self, tag: str, index: int, report_norm: str) -> bool:

        patterns = [
            f"[{tag} {index}]",
            f"{tag} {index}",
        ]

        return any(pattern in report_norm for pattern in patterns)

    # -----------------------------

    def _paraphrase_overlap(self, citation_text: str, report_norm: str) -> float:

        words = self._content_words(citation_text)

        if not words:
            return 0.0

        report_words = set(self._content_words(report_norm))

        overlap = sum(1 for w in words if w in report_words)

        return overlap / len(words)

    # -----------------------------

    def _content_words(self, text: str) -> list[str]:

        text = text.lower()

        text = re.sub(r"[^\w\s]", " ", text)

        return [_stem(w) for w in text.split() if len(w) > 3]