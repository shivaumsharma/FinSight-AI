// Pure math for the Calculators page (web/src/components/Calculators.tsx)
// -- no backend call, no ticker/live-data dependency, so these live as
// plain functions the component can call synchronously on every input
// change rather than a fetch effect like everywhere else in this app.
//
// Deliberately never hardcodes a "current" government-set rate (PPF's
// real-world rate, e.g., is set quarterly by the RBI and changes) --
// every rate is a required user input with a placeholder nudging them to
// check the current number, matching this app's own "never fabricate a
// number that could silently go stale" posture elsewhere (see e.g.
// app/derivatives/options_pricer.py's degrade-honestly comments).

export interface GrowthResult {
  investedAmount: number;
  totalValue: number;
  estimatedReturns: number;
}

// Future value of a fixed payment made at the START of every period
// (SIP and PPF both work this way in practice -- the SIP debit and the
// PPF deposit both land before that period's interest accrues), n
// periods, ratePerPeriod as a fraction (0.01, not 1). Handles rate=0
// separately since the closed-form annuity formula divides by it.
function annuityDueFutureValue(payment: number, ratePerPeriod: number, periods: number): number {
  if (periods <= 0) return 0;
  if (ratePerPeriod === 0) return payment * periods;
  return payment * ((Math.pow(1 + ratePerPeriod, periods) - 1) / ratePerPeriod) * (1 + ratePerPeriod);
}

function growthResult(investedAmount: number, totalValue: number): GrowthResult {
  return { investedAmount, totalValue, estimatedReturns: totalValue - investedAmount };
}

// Standard SIP formula: monthly investment, compounded monthly at
// annualRatePct/12, for `years` years. Confirmed against a known
// benchmark during development: ₹10,000/month at 12% for 10 years ->
// ~₹23.23 lakh total value, matching widely-published SIP calculator
// results for that exact input.
export function sipFutureValue(monthlyAmount: number, annualRatePct: number, years: number): GrowthResult {
  const months = Math.round(years * 12);
  const totalValue = annuityDueFutureValue(monthlyAmount, annualRatePct / 100 / 12, months);
  return growthResult(monthlyAmount * months, totalValue);
}

// One-time investment, compounded annually.
export function lumpsumFutureValue(principal: number, annualRatePct: number, years: number): GrowthResult {
  const totalValue = principal * Math.pow(1 + annualRatePct / 100, years);
  return growthResult(principal, totalValue);
}

// Fixed Deposit maturity value, compounded `compoundingPerYear` times a
// year (quarterly -- 4 -- is the Indian-bank convention, but left as a
// user choice since not every bank/FD uses it).
export function fdMaturity(
  principal: number,
  annualRatePct: number,
  years: number,
  compoundingPerYear: number
): GrowthResult {
  const r = annualRatePct / 100;
  const totalValue = principal * Math.pow(1 + r / compoundingPerYear, compoundingPerYear * years);
  return growthResult(principal, totalValue);
}

// PPF: a fixed amount deposited once a year, compounded annually --
// same annuity-due shape as SIP, just yearly periods instead of
// monthly. Does NOT enforce PPF's real statutory ₹500-₹1.5L/year
// contribution limits or fixed 15-year lock-in -- this is a generic
// "yearly deposit, annual compounding" calculator, not a PPF-rules
// validator; the page copy should make that framing clear rather than
// implying a limit check that isn't actually happening.
export function ppfMaturity(yearlyAmount: number, annualRatePct: number, years: number): GrowthResult {
  const totalValue = annuityDueFutureValue(yearlyAmount, annualRatePct / 100, Math.round(years));
  return growthResult(yearlyAmount * Math.round(years), totalValue);
}

// CAGR% between two point-in-time values. Returns null for an input
// combination the formula can't handle (a zero/negative value, or
// years <= 0) rather than returning NaN for the UI to accidentally
// render as a literal "NaN%".
export function cagr(initialValue: number, finalValue: number, years: number): number | null {
  if (initialValue <= 0 || finalValue <= 0 || years <= 0) return null;
  return (Math.pow(finalValue / initialValue, 1 / years) - 1) * 100;
}
