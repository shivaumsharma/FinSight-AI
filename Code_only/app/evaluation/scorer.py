class ScoreAggregator:

    def aggregate(

        self,

        grounding,

        retrieval,

        citations,

        report

    ):

        overall = (

            grounding.grounding_score * 0.40 +

            retrieval.retrieval_score * 0.20 +

            citations.citation_coverage * 0.20 +

            report.completeness_score * 0.20

        )

        return round(overall, 2)