"""
logger.py

Logs every completed research session.

Purpose
-------
Persist every ResearchContext to disk so that:

- Runs can be audited
- Responses can be inspected
- Evaluation metrics are stored
- Training datasets can later be generated
"""

import json
import os
from datetime import datetime

from app.core.research_context import ResearchContext


class ResearchLogger:

    def __init__(self, log_directory: str = "logs"):

        self.log_directory = log_directory

        os.makedirs(self.log_directory, exist_ok=True)

    def save(
        self,
        context: ResearchContext
    ) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = (
            f"{context.ticker}_{timestamp}.json"
        )

        filepath = os.path.join(
            self.log_directory,
            filename
        )

        log_data = {

            "timestamp": context.request_time.isoformat(),

            "ticker": context.ticker,

            "question": context.question,

            "company_info": context.company_info,

            "market_cap": context.market_cap,

            "beta": context.beta,

            "sentiment": context.sentiment,

            "retrieved_chunks": context.retrieved_chunks,

            "generated_answer": context.generated_answer,

            "context.valuation_results": self._serialize(
                context.valuation_results
            ),


            "investment_thesis": context.investment_thesis,

            "evaluation": context.generated_answer,

            "metadata": context.metadata

        }

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                log_data,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        return filepath

    def _serialize(self, obj):

        """
        Convert DataFrames and other objects
        into JSON-safe structures.
        """

        try:

            if hasattr(obj, "to_dict"):

                return obj.to_dict()

            elif isinstance(obj, dict):

                return {
                    key: self._serialize(value)
                    for key, value in obj.items()
                }

            elif isinstance(obj, list):

                return [
                    self._serialize(item)
                    for item in obj
                ]

            else:

                return obj

        except Exception:

            return str(obj)