import pandas as pd


class DDMEngine:
    """
    Gordon Growth dividend discount model -- a genuinely independent
    valuation lens from the FCFF-DCF (app/valuation/dcf_engine.py):
    values a share off cash actually returned to shareholders and its
    own growth trend, not modeled free cash flow. Deliberately scoped
    to consistent, established dividend payers only (see
    is_dividend_payer) -- this is a cross-check for exactly the mature,
    slow-growth category (deep-value, mega-cap staples) EVALUATION.md's
    own mega-cap/deep-value skew finding centers on, not a general-
    purpose valuation model meant to cover every company DCF already
    covers.
    """

    MIN_YEARS_PAYING = 2
    # Same instability DCFEngine's own MIN_WACC_TERMINAL_SPREAD guards
    # against (Gordon-growth terminal value blows up as the discount
    # rate approaches the growth rate), same 0.03 value -- applied here
    # to DDM's own cost-of-equity/dividend-growth spread instead of
    # DCF's WACC/terminal-growth spread. FLOORED, not declined: an
    # early version returned None whenever the spread was too tight,
    # which turned out to silently exclude exactly the population this
    # model exists for -- confirmed live against KO/PEP/JNJ, all real,
    # multi-decade dividend growers, where this pipeline's own live-
    # CAPM cost of equity (low beta -> low cost of equity, see
    # valuation_pipeline.py's now-live risk-free rate) lands only
    # fractions of a point above their actual dividend growth rate.
    # DCFEngine's own WACC floor exists for precisely this reason;
    # mirrored here rather than reinvented.
    MIN_COE_GROWTH_SPREAD = 0.03

    def __init__(self, financial_df: pd.DataFrame, cost_of_equity):
        self.financial_df = financial_df
        self.cost_of_equity = cost_of_equity

    def calculate_dps_series(self) -> pd.Series:
        """Dividends paid / shares outstanding, aligned per fiscal
        year -- shares outstanding can move year to year (buybacks),
        so this isn't just a running total divided by the latest
        share count."""
        if "dividends_paid" not in self.financial_df.columns or "shares_outstanding" not in self.financial_df.columns:
            return pd.Series(dtype=float)
        dividends = self.financial_df["dividends_paid"].dropna()
        shares = self.financial_df["shares_outstanding"].dropna()
        aligned = pd.concat([dividends, shares], axis=1).dropna()
        aligned.columns = ["dividends_paid", "shares_outstanding"]
        aligned = aligned[aligned["shares_outstanding"] > 0]
        return aligned["dividends_paid"] / aligned["shares_outstanding"]

    def is_dividend_payer(self) -> bool:
        """True only if the company paid a real, non-zero dividend in
        EVERY available fiscal year -- a single one-off or token
        payout shouldn't qualify a company for a model built on the
        assumption of a durable, growing payout stream. A company that
        just started or just suspended its dividend gets excluded
        here, correctly: neither case has the multi-year track record
        this model's growth-rate extrapolation needs to mean anything.

        Also requires the payout to be economically material (latest-
        year payout ratio >= MIN_PAYOUT_RATIO), not just present --
        confirmed live against NVDA: it has paid a nominal dividend for
        4 straight years (technically clears the years-paying check
        above) at a payout ratio under 10% of net income every year,
        a token capital-return gesture, not a real policy. Without
        this check DDM produced a $1.81 "intrinsic value" against a
        ~$209 price -- a real dividend-policy company would never
        actually be priced this way, the model was just being handed
        an input it was never designed to represent."""
        dps = self.calculate_dps_series()
        if len(dps) < self.MIN_YEARS_PAYING or not bool((dps > 0).all()):
            return False
        return self._latest_payout_ratio_is_material()

    MIN_PAYOUT_RATIO = 0.15

    def _latest_payout_ratio_is_material(self) -> bool:
        if "dividends_paid" not in self.financial_df.columns or "net_income" not in self.financial_df.columns:
            return False
        aligned = pd.concat(
            [self.financial_df["dividends_paid"].dropna(), self.financial_df["net_income"].dropna()],
            axis=1,
        ).dropna()
        aligned.columns = ["dividends_paid", "net_income"]
        if aligned.empty or aligned["net_income"].iloc[-1] <= 0:
            return False
        payout_ratio = aligned["dividends_paid"].iloc[-1] / aligned["net_income"].iloc[-1]
        return payout_ratio >= self.MIN_PAYOUT_RATIO

    # A Gordon Growth model takes whatever growth rate it's given and
    # compounds it FOREVER -- no company can out-grow the wider economy
    # at a fast clip indefinitely, which is exactly why the FCFF-DCF
    # fades toward a modest terminal rate over 10 years (see
    # fcff_engine.py's 3-stage fade) rather than using a raw historical
    # CAGR directly. This model has no fade mechanism (a Gordon Growth
    # DDM structurally can't -- there's no multi-stage forecast to fade
    # across), so the trailing DPS CAGR is capped here instead. Confirmed
    # live: MSFT's raw 3-year DPS CAGR (10.1%, a real but clearly
    # temporary acceleration) fed directly into Gordon Growth priced it
    # at $130 against a $486 actual price -- a distortion from the
    # growth INPUT, not a real read on MSFT being overvalued. Capped
    # slightly above the DCF's own highest quality-tier terminal rate
    # (see fcff_engine.py's TERMINAL_GROWTH_ADJUSTMENT_HIGH_QUALITY,
    # 4%+1%=5%) rather than at that same 4-5% band exactly -- a
    # dividend-paying business earning this test's way in (a real,
    # material, multi-year-consistent payout, see is_dividend_payer)
    # can plausibly sustain modestly faster payout growth than the
    # general DCF population for a while by still raising its payout
    # ratio, just not indefinitely faster.
    MAX_SUSTAINABLE_DPS_GROWTH = 0.06

    def calculate_dps_cagr(self):
        """None (not a crash) on the same degenerate inputs
        calculate_revenue_cagr already guards against -- see that
        method's own docstring in fcff_engine.py. Capped at
        MAX_SUSTAINABLE_DPS_GROWTH -- see that constant's own comment."""
        dps = self.calculate_dps_series()
        if len(dps) < 2:
            return None
        beginning_value = dps.iloc[0]
        ending_value = dps.iloc[-1]
        n = len(dps) - 1
        if beginning_value <= 0:
            return None
        raw_cagr = (ending_value / beginning_value) ** (1 / n) - 1
        return min(raw_cagr, self.MAX_SUSTAINABLE_DPS_GROWTH)

    def _floored_cost_of_equity(self, growth: float) -> float:
        """Same floor-up (not decline-to-compute) behavior as
        DCFEngine.wacc_floor_info -- see MIN_COE_GROWTH_SPREAD's own
        comment for why flooring, not returning None, is the right
        call here."""
        floor = growth + self.MIN_COE_GROWTH_SPREAD
        if self.cost_of_equity - growth < self.MIN_COE_GROWTH_SPREAD:
            return floor
        return self.cost_of_equity

    def calculate_intrinsic_value(self):
        """Gordon Growth DDM: next-year DPS / (cost of equity -
        dividend growth rate). None (not a distorted number) only if
        this company isn't a consistent dividend payer, cost_of_equity
        is unavailable, or the growth rate isn't computable -- a too-
        tight cost-of-equity/growth spread is floored up instead (see
        _floored_cost_of_equity), not treated as a reason to decline."""
        if not self.is_dividend_payer() or self.cost_of_equity is None:
            return None

        growth = self.calculate_dps_cagr()
        if growth is None:
            return None

        discount_rate = self._floored_cost_of_equity(growth)
        latest_dps = self.calculate_dps_series().iloc[-1]
        next_year_dps = latest_dps * (1 + growth)
        return next_year_dps / (discount_rate - growth)
