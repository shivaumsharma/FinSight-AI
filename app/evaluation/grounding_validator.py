"""
grounding_validator.py

Checks whether claims made by the generated report
are supported by the grounded research summary.

Version 1:
- Rule based
- Fast
- Deterministic

Future versions:
- NLI
- LLM Judge
"""

import re
from dataclasses import dataclass


@dataclass
class GroundingResult:

    supported_claims: int
    unsupported_claims: int
    grounding_score: float

    supported_sentences: list[str]
    unsupported_sentences: list[str]


class GroundingValidator:

    def validate(
        self,
        research_summary: str,
        generated_report: str,
    ) -> GroundingResult:

        research = self._normalize(research_summary)

        sentences = self._split_sentences(generated_report)

        supported = []
        unsupported = []

        for sentence in sentences:

            if len(sentence) < 20:
                continue

            if self._is_heading(sentence):
                continue

            sentence_norm = self._normalize(sentence)

            if sentence_norm in research:
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

    def _normalize(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        return text

    # -----------------------------

    def _split_sentences(self, text):

        return re.split(r"(?<=[.!?])\s+", text)

    # -----------------------------

    def _is_heading(self, text):

        headings = {

            "executive summary",

            "bull case",

            "bear case",

            "financial outlook",

            "investment recommendation",

            "confidence score",

        }

        return text.lower().strip() in headings