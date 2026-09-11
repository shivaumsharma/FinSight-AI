"""
report_validator.py

Checks whether the generated report's narrative sections actually
contain real content -- not just whether their headings are present in
the final assembled string.

THE BUG THIS REPLACED: validate() used to take the flattened report
string and check whether each REQUIRED section NAME appeared as a
substring anywhere in it. But report_tool.py builds that string as
`f"# {section}\n{narrative[section]}"` for every section
UNCONDITIONALLY -- the heading itself is always present, whether or
not the model actually produced real content for it (a section the
model failed to write gets narrative_builder.py's own
NARRATIVE_SECTION_FALLBACK placeholder instead of vanishing). So
"Executive Summary" was always a substring of the report regardless of
what came after it, `missing` was always empty, and completeness_score
was mathematically always 100.0 -- confirmed empirically: forcing
every section to the fallback placeholder still validated as
complete=True. This is the one metric specifically meant to catch a
completely failed narrative generation, and it could not.

validate() now takes the pre-assembly {section_name: text} dict
(context.report_data["narrative"], populated by
narrative_builder.build_narrative_sections) instead of the flattened
string, and checks each section's actual VALUE against the shared
NARRATIVE_SECTION_FALLBACK constant -- the thing report_tool.py's
string-building step was papering over.
"""

from dataclasses import dataclass
from typing import Dict

from app.reporting.narrative_builder import NARRATIVE_SECTION_FALLBACK


@dataclass
class ReportValidationResult:

    complete: bool

    missing_sections: list[str]

    completeness_score: float


class ReportValidator:

    REQUIRED = [

        "Executive Summary",

        "Business Analysis",

        "Market and Earnings Analysis",

        "Risk Analysis",

        "Investment Thesis",

    ]

    def validate(self, narrative: Dict[str, str]):

        narrative = narrative or {}
        missing = []

        for section in self.REQUIRED:

            text = (narrative.get(section) or "").strip()

            if not text or text == NARRATIVE_SECTION_FALLBACK:

                missing.append(section)

        score = (

            (len(self.REQUIRED) - len(missing))

            / len(self.REQUIRED)

            * 100

        )

        return ReportValidationResult(

            complete=len(missing) == 0,

            missing_sections=missing,

            completeness_score=score,

        )
