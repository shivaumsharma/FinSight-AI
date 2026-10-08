import pandas as pd


def calculate_tax_rate(financial_df: pd.DataFrame) -> pd.Series:
    """Per-year effective tax rate (tax_expense / pretax_income),
    clipped to [0, 0.5]. Years with pretax_income <= 0 are excluded --
    a loss year's tax_expense/pretax_income ratio isn't a meaningful
    effective tax rate, and letting it through used to clip to a
    fabricated 50% ceiling, corrupting NOPAT/WACC for that year.

    Shared by WACCEngine.calculate_tax_rate (averages this Series) and
    FCFFEngine.calculate_tax_rate (uses it per-year for NOPAT) -- the
    two used to duplicate this formula, and FCFFEngine's copy was
    missing the pretax_income>0 filter that produced the bug above.
    """
    tax_expense = financial_df["tax_expense"].dropna()
    pretax_income = financial_df["pretax_income"].dropna()
    aligned_df = pd.concat([tax_expense, pretax_income], axis=1).dropna()
    aligned_df.columns = ["tax_expense", "pretax_income"]
    aligned_df = aligned_df[aligned_df["pretax_income"] > 0]

    aligned_df["tax_rate"] = (aligned_df["tax_expense"] / aligned_df["pretax_income"]).clip(0, 0.5)

    return aligned_df["tax_rate"]
