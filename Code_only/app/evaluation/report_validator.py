"""
report_validator.py

Checks whether the generated report
contains all required sections.
"""

from dataclasses import dataclass


@dataclass
class ReportValidationResult:

    complete: bool

    missing_sections: list[str]

    completeness_score: float


class ReportValidator:

    REQUIRED = [

        "Executive Summary",

        "Bull Case",

        "Bear Case",

        "Financial Outlook",

        "Investment Recommendation",

    ]

    def validate(self, report: str):

        missing = []

        for section in self.REQUIRED:

            if section.lower() not in report.lower():

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