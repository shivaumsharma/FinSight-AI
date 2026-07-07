"""
benchmark_runner.py

Loads benchmark cases and evaluates reports.
"""

import json

from app.evaluation.scorer import ReportScorer


class BenchmarkRunner:

    def load(self, path):

        with open(path) as f:
            return json.load(f)

    def evaluate(
        self,
        benchmark_path,
        generated_report
    ):

        benchmark = self.load(benchmark_path)

        return ReportScorer().score(
            benchmark,
            generated_report
        )