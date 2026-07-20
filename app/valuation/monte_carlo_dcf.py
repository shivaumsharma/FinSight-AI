"""
monte_carlo_dcf.py

Monte Carlo layer over FinSight's own DCF engines -- the sampling
*technique* was observed while read-only-auditing a separate project
(the DCF Valuation Engine at DCF_Project/), not copied from it. That
project's underlying DCF math was audited and found materially
weaker than FinSight's own (flat beta-only discount rate with no
capital-structure weighting, single-stage growth, no capex/NWC
outlier normalization, a `net_income * 0.7` FCF fallback) -- see the
audit for detail. So only the sampling approach is reused here (draw
growth/WACC/terminal-growth as normals, clip to sane ranges, reject
unstable combinations, run many samples); every sample is priced
through FinSight's real FCFFEngine/DCFEngine, not a re-implementation.

Produces a DISTRIBUTION of intrinsic values instead of one point
estimate -- complements the existing 5x5 deterministic sensitivity
grid (app/valuation/sensitivity_analysis.py, which shows how the
point estimate moves under specific WACC/growth combinations) with
actual uncertainty quantification: mean/median/std/percentiles and a
probability-of-undervaluation.

Display-only: NOT folded into the recommendation composite score
(see report_data_builder.py's DCF_WEIGHT/RELATIVE_WEIGHT) until it
has its own accuracy track record -- same reasoning applied to the
ML classifier in ml_valuation_classifier.py, and consistent with how
this project has treated every new signal introduced this session.
"""

import numpy as np

from app.valuation.dcf_engine import DCFEngine

# Sampling widths, adapted from the audited project's Monte Carlo
# technique. Growth is perturbed proportionally to its own magnitude
# (a company with near-zero growth shouldn't get an artificially wide
# absolute band); WACC and terminal growth get fixed absolute widths
# since they're already narrow, mean-reverting quantities.
GROWTH_RATE_STD_FRACTION = 0.30
WACC_STD = 0.015
TERMINAL_GROWTH_STD = 0.005

GROWTH_RATE_CLIP = (-0.10, 0.30)
WACC_CLIP = (0.05, 0.25)
TERMINAL_GROWTH_CLIP = (0.00, 0.05)

DEFAULT_ITERATIONS = 2000


class MonteCarloDCFEngine:

    def __init__(
        self,
        fcff_engine,
        base_growth_rate: float,
        base_wacc: float,
        base_terminal_growth: float,
        total_debt: float,
        cash: float,
        shares_outstanding: float,
        forecast_years: int = 10,
        iterations: int = DEFAULT_ITERATIONS,
        random_state=None,
    ):
        self.fcff_engine = fcff_engine
        self.base_growth_rate = base_growth_rate
        self.base_wacc = base_wacc
        self.base_terminal_growth = base_terminal_growth
        self.total_debt = total_debt
        self.cash = cash
        self.shares_outstanding = shares_outstanding
        self.forecast_years = forecast_years
        self.iterations = iterations
        self._rng = np.random.default_rng(random_state)

    def run(self) -> np.ndarray:
        """Returns an array of sampled intrinsic-value-per-share
        outcomes. Empty array if the base FCFF isn't usable (e.g. DCF
        already determined unavailable for this company) or every
        sampled combination was rejected for stability."""
        base_fcff = self.fcff_engine.calculate_normalized_base_fcff()
        if base_fcff is None or base_fcff <= 0:
            return np.array([])

        rng = self._rng

        growth_samples = rng.normal(
            loc=self.base_growth_rate,
            scale=max(abs(self.base_growth_rate) * GROWTH_RATE_STD_FRACTION, 0.01),
            size=self.iterations,
        )
        wacc_samples = rng.normal(loc=self.base_wacc, scale=WACC_STD, size=self.iterations)
        terminal_samples = rng.normal(
            loc=self.base_terminal_growth, scale=TERMINAL_GROWTH_STD, size=self.iterations
        )

        growth_samples = np.clip(growth_samples, *GROWTH_RATE_CLIP)
        wacc_samples = np.clip(wacc_samples, *WACC_CLIP)
        terminal_samples = np.clip(terminal_samples, *TERMINAL_GROWTH_CLIP)

        # Reject rather than floor: DCFEngine would silently floor an
        # unstable WACC/terminal-growth pair to a fixed value (see
        # DCFEngine.MIN_WACC_TERMINAL_SPREAD), which would pile many
        # rejected samples onto one identical outcome and understate
        # the true dispersion. Filtering upfront keeps every sample
        # in the output genuinely distinct.
        valid = (wacc_samples - terminal_samples) >= DCFEngine.MIN_WACC_TERMINAL_SPREAD
        growth_samples = growth_samples[valid]
        wacc_samples = wacc_samples[valid]
        terminal_samples = terminal_samples[valid]

        intrinsic_values = np.empty(len(growth_samples))
        for i, (growth, wacc, terminal) in enumerate(zip(growth_samples, wacc_samples, terminal_samples)):
            forecast = self.fcff_engine.forecast_fcff(
                forecast_years=self.forecast_years,
                terminal_growth_rate=terminal,
                base_fcff_override=base_fcff,
                initial_growth_rate_override=growth,
            )
            dcf = DCFEngine(forecast_fcff_df=forecast, discount_rate=wacc, terminal_growth_rate=terminal)
            equity_value = dcf.calculate_equity_value(total_debt=self.total_debt, cash=self.cash)
            intrinsic_values[i] = equity_value / self.shares_outstanding

        return intrinsic_values

    @staticmethod
    def statistics(mc_values: np.ndarray, current_price: float):
        if mc_values is None or len(mc_values) == 0:
            return None
        return {
            "mean": float(np.mean(mc_values)),
            "median": float(np.median(mc_values)),
            "std_dev": float(np.std(mc_values)),
            "p25": float(np.percentile(mc_values, 25)),
            "p75": float(np.percentile(mc_values, 75)),
            "ci_lower": float(np.percentile(mc_values, 5)),
            "ci_upper": float(np.percentile(mc_values, 95)),
            "prob_undervalued": float(np.mean(mc_values > current_price)),
            "n_samples": int(len(mc_values)),
        }
