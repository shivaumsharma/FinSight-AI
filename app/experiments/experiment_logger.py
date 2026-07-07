"""
experiment_logger.py

Stores experiment results.
"""

import json
from pathlib import Path


class ExperimentLogger:

    def __init__(self):

        self.output_dir = Path("experiments/reports")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, result):

        filename = (
            self.output_dir
            / f"{result['timestamp'].replace(':','-')}.json"
        )

        with open(filename, "w") as f:

            json.dump(
                result,
                f,
                indent=4
            )