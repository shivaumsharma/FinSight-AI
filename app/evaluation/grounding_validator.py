"""
grounding_validator.py

Checks whether claims made by the generated report
are supported by the grounded research summary.

Version 2:
- Word-overlap based, not exact-substring based.
- Rule based / fast / deterministic.

Why this changed from v1
-------------------------
v1 marked a sentence "supported" only if it appeared, verbatim
(after lowercasing/punctuation-stripping), as a *substring* inside
the research summary. But the report-writing prompt explicitly
instructs the model to summarize/paraphrase the evidence and never
copy it verbatim ("This is a summarization task, NOT a creative
writing task" / "Never invent quotes" -- summarizing is exactly what
was asked for). A correctly-written, fully accurate, fully grounded
report will almost never contain a sentence that is an exact
substring of the source text, so v1's grounding score was close to
unsatisfiable regardless of report quality -- it wasn't actually
measuring faithfulness, it was measuring "did the model fail to
paraphrase."

v2 instead checks how much of a sentence's *meaningful* vocabulary
(content words, stopwords stripped) also appears somewhere in the
research summary. A sentence that paraphrases the source will share
most of its content words with the source even though the wording
differs; a hallucinated sentence introducing facts not present in
the source will not. This is still a cheap, deterministic,
non-semantic check (no embeddings/NLI), just a better proxy for
"is this actually grounded" than literal string containment.

Future versions:
- NLI
- LLM Judge
"""

import re
from dataclasses import dataclass


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "by", "at", "from",
    "has", "have", "had", "will", "would", "could", "should", "may",
    "might", "also", "their", "which", "who", "than", "then", "there",
    "here", "not", "no", "if", "so", "such", "into", "over", "about",
    "across", "we", "you", "they", "he", "she", "i",
}



_SUFFIXES = (
    "ational", "ization", "ations", "tions", "ing", "ally", "ment",
    "ness", "tion", "ence", "ance", "ives", "ive", "ers", "er", "ed",
    "es", "ly", "al", "s",
)


def _stem(word: str) -> str:
    """
    Very crude suffix-stripping stemmer (no external NLP deps).
    Just enough to collapse common word-form mismatches a small
    paraphrasing model produces -- e.g. "operating"/"operations",
    "profitability"/"profit", "declining"/"decline" -- so the overlap
    check isn't defeated by inflection alone.
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass
class GroundingResult:

    supported_claims: int
    unsupported_claims: int
    grounding_score: float

    supported_sentences: list[str]
    unsupported_sentences: list[str]


class GroundingValidator:

    # Fraction of a sentence's content words that must appear
    # somewhere in the research summary for the sentence to count
    # as "supported." Tune this if the metric feels too strict/loose
    # once you see it running against real reports.
    SUPPORT_THRESHOLD = 0.3

    def validate(
        self,
        research_summary: str,
        generated_report: str,
    ) -> GroundingResult:

        research_words = set(self._content_words(research_summary))

        sentences = self._split_sentences(generated_report)

        supported = []
        unsupported = []

        for sentence in sentences:

            if len(sentence) < 20:
                continue

            if self._is_heading(sentence):
                continue

            words = self._content_words(sentence)

            if not words:
                continue

            overlap = sum(1 for w in words if w in research_words)
            ratio = overlap / len(words)

            if ratio >= self.SUPPORT_THRESHOLD:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        total = len(supported) + len(unsupported)

        score = (
            len(supported) / total * 100
            if total
            else 0.0
        )

        return GroundingResult(
            supported_claims=len(supported),
            unsupported_claims=len(unsupported),
            grounding_score=round(score, 2),
            supported_sentences=supported,
            unsupported_sentences=unsupported,
        )

    # -----------------------------

    def _content_words(self, text: str) -> list[str]:

        text = text.lower()

        text = re.sub(r"[^\w\s]", " ", text)

        return [
            _stem(w) for w in text.split()
            if w not in _STOPWORDS and len(w) > 2
        ]

    # -----------------------------

    def _split_sentences(self, text):

        return re.split(r"(?<=[.!?])\s+", text)

    # -----------------------------

    def _is_heading(self, text):

        headings = {

            "executive summary",

            "business analysis",

            "market and earnings analysis",

            "risk analysis",

            "investment thesis",

            "confidence score",

        }

        return text.lower().strip() in headings