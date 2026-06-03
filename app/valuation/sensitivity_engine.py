import pandas as pd

class SensitivityEngine:

    def __init__(self, base_enterprise_value):
        self.base_ev = base_enterprise_value

    def generate_matrix(self):

        wacc_values = [0.08, 0.09, 0.10, 0.11, 0.12]

        growth_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        data = []

        for growth in growth_values:

            row = []

            for wacc in wacc_values:

                adjustment = (
                    (1 + growth)
                    /
                    (wacc / 0.10)
                )

                row.append(
                    round(
                        self.base_ev * adjustment,
                        0
                    )
                )

            data.append(row)

        return pd.DataFrame(
            data,
            index=[
                f"{g:.0%}"
                for g in growth_values
            ],
            columns=[
                f"{w:.0%}"
                for w in wacc_values
            ]
        )