"""
experiment_runner.py

Runs benchmark experiments and stores results.
"""

from datetime import datetime

from experiments.experiment_logger import ExperimentLogger
from evaluation.benchmark_runner import BenchmarkRunner


class ExperimentRunner:

    def __init__(self):

        self.benchmark = BenchmarkRunner()

        self.logger = ExperimentLogger()

    def run(
        self,
        config,
        generated_report: str,
        benchmark_file: str
    ):

        scores = self.benchmark.evaluate(
            benchmark_file,
            generated_report
        )

        result = {

            "timestamp": datetime.now().isoformat(),

            "model": config["model"],

            "prompt_version": config["prompt_version"],

            "reranker": config["reranker"],

            "financial_summary": config["financial_summary"],

            "valuation_summary": config["valuation_summary"],

            "sentiment_summary": config["sentiment_summary"],

            "scores": scores

        }

        self.logger.save(result)

        return result