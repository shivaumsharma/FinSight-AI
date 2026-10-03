# FinSight Returns Sprint — Tracker

Started 2026-10-03. One-week sprint, then a re-evaluation together.

## End goal

A signal-and-portfolio system that earns **positive, believable, out-of-sample** results versus a no-signal baseline, plus **something new** (a technique or data source) that gives it a reason to exist beyond good documentation. "Believable" is not decoration: it is what separates a result a leaderboard or a reviewer will accept from one they will dismiss.

## Honest expectations

- Most attempts at this fail. A published signal loses on average ~26% of its return out-of-sample and ~58% after publication (McLean & Pontiff). The plain "Lazy Prices" replication on the S&P 100 (2009–2026) found no effect at all (deflated Sharpe 0.13).
- So "a week of work produces returns" is not guaranteed. What the week does guarantee is an answer: which ideas survive a fair test, and which do not.
- A negative result is a result. It is also the only kind this project has produced so far, and it is why the numbers can be trusted.

## Rules (set before looking at results, so we can't fool ourselves)

1. **Holdout:** the most recent 12 months are untouched until Day 7. One look, one number.
2. **Trial counter:** every variant we test is counted in `TRIALS` below. More trials means a higher bar (that is what the deflated Sharpe ratio measures).
3. **Kill gates:** each new signal has a written pass/fail line. If it fails on the development period, it is dropped, not tuned.
4. **Controls first:** every new signal must beat, or add to, the boring proven factors (P3), not merely look good alone.

**TRIALS so far (this sprint):** 10 (7 from P3; 1 from P4 primary; 2 sector-neutral variants of the composite IC)

## SCOPE UPDATE (user, 2026-10-04) — supersedes the decisions below where they conflict

- **Active now: P1–P4 only.** P5–P12 (including Bet B Kronos and Bet C LLM alpha mining) are DEFERRED. **I must bring P5–P12 back up for the user after P1–P4 results are in** — they want to read them first.
- **Research first, in a sandbox.** All work lives in `research/` and touches no production code. Anything is converted into a real project addition only if its results are positive against the pre-registered pass lines. A random search that could harm the project is explicitly not wanted.
- **P3 must prove it helps:** the proven factors only count if they measurably improve the score (added IC beyond the current composite, and better walk-forward results) — not merely exist.
- **Papers:** go one at a time, one per message, while the work runs in the background.
- Everything dated 2026-10-03 below (all three bets approved, etc.) stays on record, but Bets B and C do not start until P1–P4 are reviewed.

### P2 measured result — baseline IC (2026-10-04, existing yfinance data, survivorship-biased)

| Score | Pooled IC | Mean per-period IC | t-stat | Positive periods |
|---|---|---|---|---|
| composite_score | +0.093 | +0.075 | **+0.92** | 8 of 11 |
| dcf_score | +0.095 | +0.075 | +0.94 | 8 of 11 |
| relative_score | +0.041 | +0.049 | +0.90 | 7 of 11 |

Per-period composite IC swings from -0.56 to +0.55. **It does NOT clear the pre-registered P2 pass line (t ≥ 2).** Descriptive size split (not used to select anything): S&P 400 t=+1.43, S&P 500 t=+0.90, S&P 600 t≈0 (only 139 rows).

### P2b RESULT — production model rebuilt on EDGAR history (2026-10-04, development only; sealed holdout untouched)

- **Run:** unchanged production scorer on EDGAR as-filed statements; 984 tickers x 53 quarter-ends (2012-06 to 2025-06); 43,343 ticker-dates run, 26,549 with a composite score (the rest: DCF unavailable or too little history); 359-626 scored names per date (median 498). Code: `research/pit_composite.py` (adapter + run), `research/pit_composite_ic.py` (pre-registered test).
- **Adapter fidelity (checked before trusting any IC):** on 33 names scored by both yfinance and EDGAR inputs at 2025-06-30, composite rank correlation **0.83**, DCF-score 0.80. Remaining gap: yfinance's lease-inclusive debt, restated numbers, and the DCF's extreme sensitivity to small input changes (e.g. EXPE upside +252% vs -65% from modest input differences). Imperfect fidelity adds noise, which pushes IC toward zero rather than inflating it. Data fixes made on the way (all before any IC was seen): added debt, retained-earnings, continuing-ops cash-flow, capex and interest tags (missing CFO 34% -> 0%, interest 17% -> 0%); EBIT = pretax + interest as yfinance defines it; 4-year history window to match yfinance.

| Window | Variant | Mean IC | t | Positive dates | Alive? |
|---|---|---|---|---|---|
| **2012-06 to 2022-12 (decisive, 43 dates)** | H1 raw composite | +0.016 | +0.84 | 21/43 | **No** |
| | H2 sector-neutral | +0.010 | +0.58 | 20/43 | **No** |
| | paired H2 - H1 | -0.006 | -1.36 | | |
| 2023-03 to 2025-06 (10 dates) | H1 raw | -0.014 | -0.40 | 3/10 | No |
| | H2 sector-neutral | -0.000 | -0.01 | 4/10 | No |
| All 53 dates | H1 raw | +0.011 | +0.62 | 24/53 | No |
| | H2 sector-neutral | +0.008 | +0.55 | 24/53 | No |

- Descriptive: out-of-sample raw top-minus-bottom quintile spread +1.1%/quarter, t = +1.16. Yearly mean IC swings from -0.085 (2025) to +0.194 (2020).
- **Verdict: the +0.093 / t = 2.29 hint did NOT replicate out of sample.** Raw IC is slightly positive but statistically indistinguishable from zero; sector-neutral does not help (paired t = -1.36). Reading: the nine-quarter hint was noise/regime. The honest claim is that the composite has no demonstrated stock-selection edge in this data (survivorship-biased, so if anything this flatters the model).
- **TRIALS: still 10** (H1 and H2 were already counted).

### WHY THE COMPOSITE FAILS — diagnosis (2026-10-04; `research/why_composite_fails.py`; diagnostic only, no trials added)

1. **Not an aggregation problem:** DCF-only IC +0.012 (t 0.72), relative-only +0.009 (t 0.56), composite +0.011. Both halves are individually ~zero.
2. **Not a noisy score:** the same stock's rank is highly stable quarter to quarter (rank autocorrelation 0.85). It is a consistent ranking that simply does not predict.
3. **It is a disguised value + anti-momentum bet:** per-date correlation with earnings yield +0.44, FCF yield +0.44, 12-1 momentum -0.29, low-vol/ROA/size ~0. P3 showed those factors themselves had ~zero IC in this universe, so the composite inherits that.
4. **Regime-driven, not stock-selection:** per-date IC std is 0.125 vs 0.045 from sampling noise alone. Yearly mean IC: 2020 +0.19 (value rebound), 2018 -0.04, 2019 -0.05, 2025 -0.08. The IC tracks whether value is working that year.
5. **Flat deciles:** next-quarter return by composite decile 0..9 = 4.8, 4.4, 3.8, 3.9, 4.3, 4.5, 4.5, 4.5, 4.8, 5.6 %/qtr. Only the top decile stands out (+1.1 pts over the middle); the bottom is not worse.
6. **Horizon:** against the next 4 quarters IC rises to +0.024 (t 1.35; overlapping, descriptive). Direction right for a valuation signal, still insignificant.
7. **DCF output is extreme:** median upside +45%, 75th pct +149%, 95th pct +570%; 8% of names saturate the DCF score at >= +90. The model calls most stocks cheap, and the ranking is driven by extrapolating 3-4 years of history (hypothesis, not yet tested). The EDGAR/yfinance fidelity check also showed modest input changes flipping the verdict.
8. **Sectors:** nothing consistent (Industrials +0.032 t 1.7, Energy -0.066, rest ~0).
9. **Survivorship flatters all of this;** the true number is not better.

### P2b PRE-REGISTRATION — production model on EDGAR history (written 2026-10-04, BEFORE any score was computed)

- **What is tested:** the unchanged production scorer (`scripts/phase2_backtest._score_ticker_at_date`: DCF + relative valuation, frozen weights/thresholds) fed with EDGAR as-filed statements (shaped like yfinance's) instead of yfinance's restated 4-5 years. Nothing in `app/` or `scripts/` is modified; the adapter lives in `research/`.
- **Dates / outcome:** last trading day of each calendar quarter, 2012-06 through 2025-06 (same as P3). Outcome = total-return (adjusted close) from that quarter-end to the next. At least 100 scored names per date or the date is dropped. Dates after 2025-06-30 are the sealed holdout.
- **Fidelity rules:** only filings dated on or before the test date; the production 90-day filing-lag rule still applies on top; prices on a split basis consistent with as-reported share counts; dual-class tickers excluded (ambiguous share count); risk-free rate = 10-year Treasury close on the date; beta from trailing S&P 500 returns.
- **Known deviations from the yfinance run (disclosed up front):** total debt = long-term debt + current debt (not yfinance's lease-inclusive Total Debt); shares from the 10-K cover page; no restated numbers; survivorship bias remains (same universe).
- **Hypotheses:** H1 (primary): raw `composite_score` Spearman IC per date. H2 (secondary): sector-neutral = within-(date, sector) percentile rank of `composite_score` vs forward return minus the (date, sector) mean return.
- **Pass lines:** a variant is "alive" if mean IC >= 0.02 and t >= 2 across dates. The decisive evidence is the **out-of-sample window 2012-06 to 2022-12**: the nine yfinance-based periods that produced the +0.093 / sector-neutral hint (2024-06 to 2026-06) never overlap it. H2 "improves the model" only if H2 is alive out-of-sample AND the paired IC difference H2 - H1 has t >= 2 there.
- **TRIALS:** H1 and H2 were already counted (10); nothing new is added unless a variant is tried after seeing results.

### P2 DIAGNOSIS — why the baseline IC looked random (2026-10-04)

- **Root cause 1 (measurement artifact):** the earlier t = 0.92 included three periods with 1, 13 and 17 scored stocks (yfinance's ~5-year fundamentals ceiling meant almost nothing could be scored for late-2023 / early-2024 dates). With 13 and 17 stocks the IC was +0.55 and −0.56 — pure noise that inflated the spread. On the nine periods with 140–184 stocks: **mean IC +0.093, t = +2.29, 7 of 9 positive, leave-one-out t between 1.83 and 3.03.** This exclusion was decided after seeing the data (flagged); it matches the ≥100-names rule pre-registered for P3, but it is NOT a clean pass.
- **Root cause 2 (sample size):** 9 usable periods × ~160 names. Per-period IC standard error from n=160 alone is ~0.08. The data cannot produce a stable t-stat; EDGAR (P1) lifts this to ~50 quarterly periods back to 2012.
- **Root cause 3 (sector bets):** 24.5% of composite_score variance is sector membership. Exploratory sector-neutral variant (rank within sector, sector-demeaned returns): mean IC +0.084, t = +2.71 (another formula, +0.092, t = +3.20); period-IC std falls 0.122 → 0.086; 8 of 9 positive. **Counted as 2 trials (counter now 10); post-hoc on the same data, so a hypothesis to validate out of sample, not a result.**
- Raw top-minus-bottom quintile spread: +4.4%/quarter, t = 1.63 (very noisy: −11.4% to +17.4%).
- **Fix plan:** (1) require ≥100 names per period in every IC report (done in all new code); (2) rebuild the valuation score on EDGAR point-in-time data to get ~50 periods, then re-measure raw vs sector-neutral on the pre-2023 history, which neither variant has seen; (3) keep the sealed 2025-07+ holdout for the final test.

### P4 RESULT (2026-10-04, development only, run exactly as pre-registered)

- 1,832 10-Ks downloaded for 150 S&P 400/600 companies; 117 (6.4%) had no extractable Item 1A and were skipped; 1,548 year-over-year pairs scored, 1,531 with a usable outcome across 144 companies.
- New-sentence fraction: median 0.31 (10th–90th percentile 0.18–0.56), so there is plenty of variation to test.
- **13 filing-year groups (all Q1, because 10-Ks cluster in Feb–Mar): mean IC −0.005, t = −0.22, hit rate 46%, top-minus-bottom abnormal return −0.06% over 63 days. Pooled IC over all filings −0.008.**
- **Verdict: KILL GATE TRIGGERED. P4 = not found.** Embedding-distance and LLM-labelling variants were NOT run (the rule bans a rescue search). This matches the public S&P 100 replication.
- Caveats: survivorship (same universe as P3); a possible effect in micro-caps or in the first-year-of-risk-factors era (2005–2011) is untested because our window starts 2012.
- **TRIALS: still 10** (the primary signal was already counted).

### P3 PRE-REGISTRATION (written 2026-10-04, BEFORE any factor result was computed)

- **Data:** EDGAR as-filed fundamentals, using only filings dated before the test date (point-in-time); yfinance prices. Universe = current FinSight universe (survivorship caveat stays open, see P1).
- **Dates:** last trading day of each calendar quarter, 2012-06 through 2025-06 = DEVELOPMENT. Forward return = total return to the next quarter-end. Test dates from 2025-09-30 onward are the SEALED HOLDOUT: the code refuses to evaluate them until Day 7.
- **Five factors, fixed now:** (1) `mom_12_1`: return from 12 months ago to 1 month ago (skips the latest month); (2) `low_vol`: minus the std-dev of daily returns over 252 trading days; (3) `earnings_yield`: latest known annual net income / market cap; (4) `fcf_yield`: (operating cash flow − capex) / market cap; (5) `roa`: net income / total assets. Market cap = as-traded price × as-reported shares (split-corrected).
- **Two combinations, fixed now:** `valuation_only` = average rank of earnings_yield and fcf_yield. `multi_factor` = average of four family ranks: value (the valuation_only rank), momentum, quality (roa), low_vol. Equal weights, no fitting.
- **Metric:** Spearman rank IC per date; report mean, std, t = mean / (std / √n), hit rate; plus top-minus-bottom quintile return. At least 100 stocks per date.
- **Pass lines:** a factor is "alive" if mean IC ≥ 0.02 and t ≥ 2. **P3 "improves the model" only if** `multi_factor` beats `valuation_only` (paired t ≥ 2 on the IC difference) AND `multi_factor` has mean IC ≥ 0.03 with t ≥ 2. Anything failing goes into a "tested, rejected" list, not into the model.
- **Comparison to the real composite:** on the overlap quarters (2023-10 to 2025-06, only ~8 periods, ~196 tickers) report the added IC of the factors beyond the production composite_score. Descriptive only; too few periods for a pass/fail.
- **TRIALS:** this adds 7 (5 factors + 2 combinations). **Counter: 7.**

### P1 results — EDGAR as-filed facts (2026-10-04)
- **Built:** 3,093,406 fact rows, 989 companies, filing dates 2009-04 to 2026-10, every value stamped with the date it became public. Code: `research/edgar_pit.py` (download), `research/pit_panel.py` (point-in-time view + validation). All in the sandbox; nothing in `app/` changed.
- **Correctness check:** EDGAR's latest-year revenue, net income and total assets match yfinance's on **27 of 27** comparisons within 1% (median difference 0.0%), so the pipeline reads the data correctly.
- **Size of the restated-numbers bias** (first-reported vs latest value differing by more than 1%, annual figures): revenue **7.1%** of company-years, operating cash flow 5.7%, net income 4.0%, equity 3.0%, total assets 2.3%. Real but modest per data point — worth fixing, not likely the main reason our IC is weak.
- **Point-in-time coverage** (companies with a current fiscal year): 749 in mid-2012, 836 in 2016, 916 in 2020, 951 in 2022, 979 in 2025. This goes back well past yfinance's ~5-year ceiling, so 2020 and 2022 are now testable.
- **Open issue — successor registrants:** SEC's ticker list points a few companies at a NEW identifier after a reorganization, orphaning their history (XOM → a 2026 registrant with 55 rows; also BLK, MRVL, PNFP and the 2026 spin-offs). They simply drop out of early-date panels (conservative: fewer names, no wrong data). A small manual successor map is a follow-up.
- **Open issue — survivorship, not yet fixed:** prices come from yfinance, which has none for delisted stocks, and 9 universe tickers (AVB, EA, EQR, BLD, JHG, NSA, TMHC, WBS, AVNS) no longer map at all. Free data cannot remove this; the plan is to MEASURE its size (share of historical S&P 500 members with no price history) rather than hide it.
- **Survivorship MEASURED** (`research/survivorship.py`; point-in-time S&P 500 membership from shardul0701/SP500-Survivorship-bias-data-2004-2026). Share of the index's members on Jan 1 that have no price history in our universe: **2012 37%, 2013 35%, 2016 31%, 2020 19%, 2024 9%, 2025 7%**. The 2025 figure (7%) is the floor from ticker renames/universe construction, so the excess survivorship gap is roughly 30 points in 2012, ~24 in 2016, ~12 in 2020. The missing names are the ones that were acquired, bankrupted or dropped out — the losers — so early-period tests are biased toward winners.
- **Status:** P1 data foundation DONE and survivorship quantified; successor map still open. Prices downloaded (997 tickers, 5 failed).
- **TRIALS:** 0 (nothing here is a candidate strategy).

### P3 RESULT (2026-10-04, development period only: 53 quarterly dates, 2012-06 to 2025-06; sealed holdout untouched)

| Signal | Mean IC | t | Hit rate | Q5−Q1 %/qtr |
|---|---|---|---|---|
| mom_12_1 | −0.005 | −0.23 | 49% | −1.3 |
| low_vol | −0.048 | −1.57 | 43% | −5.8 |
| earnings_yield | −0.010 | −0.62 | 43% | −3.4 |
| fcf_yield | +0.009 | +0.58 | 47% | −0.8 |
| roa | −0.007 | −0.45 | 51% | −3.4 |
| valuation_only | −0.002 | −0.14 | 45% | −2.7 |
| multi_factor | −0.025 | −1.22 | 41% | −4.5 |

Paired multi_factor − valuation_only: −0.022, t = −0.82. **Pre-registered verdict: FAIL.** No factor is alive; multi_factor does not beat valuation_only. All of these go on the tested-and-rejected list **for this data**.
**Confound (do not over-read):** the universe is today's index members. The measured survivorship gap (above) means the early-period losers are missing, and the large negative quintile spreads (low-vol −5.8%/qtr) are the signature of that: stocks that were high-vol/cheap-looking in 2012 and then survived into today's indices are winners by construction. So this is "factors do not help in THIS biased data", not "factors do not work".
Size split (descriptive; the bucket min-names bug was fixed after the first run, definitions unchanged): S&P 400 valuation_only −0.002 / multi_factor −0.025; S&P 500 −0.009 / −0.036; S&P 600 +0.009 / −0.009. Nothing clears noise anywhere.
**Next:** P4 may run on this same universe, but any positive P4 result must be re-checked against the survivorship gap before it counts.

### P4 PRE-REGISTRATION (written 2026-10-04, BEFORE any filing text was downloaded or scored)

- **Question:** does how much a company's 10-K risk-factor section (Item 1A) changed versus last year's predict its next-quarter return? (Lazy Prices idea; the public replication on the S&P 100 found nothing, so the prior is skeptical.)
- **Sample:** 150 tickers drawn with a fixed seed (0) from the S&P 400 and S&P 600 buckets of the FinSight universe (smaller, less-covered names, where the effect should be strongest), single-CIK companies only. All their 10-Ks filed 2012-01 onward and before 2025-07-01 (sealed holdout: filings after that are not scored until Day 7). Every filing paired with the same company's previous 10-K.
- **Signal (primary, fixed):** `new_sentence_fraction` = share of this year's Item 1A sentences (lowercased, whitespace-normalised, at least 8 words) that do NOT appear in last year's Item 1A. Higher = more change. Predicted sign: negative (more change, worse return).
- **Outcome:** return from the close 2 trading days after the filing date to the close 63 trading days later, minus the equal-weight average of all sample stocks over the same window (abnormal return). Uses total-return (adjusted) prices.
- **Metric:** group filings by calendar quarter of filing date; per group, Spearman IC of (−new_sentence_fraction) vs abnormal return; need at least 30 filings per group. Report mean IC, t across groups, hit rate, and top-minus-bottom quintile abnormal return.
- **Pass line (dev):** mean IC ≥ 0.02 and t ≥ 2 on the primary signal. **Kill gate:** mean IC < 0.02 or t < 1.5 → stop P4 and record "not found".
- **Variants (each counts as a trial, only run if the primary survives):** (a) embedding distance (BGE) between the two sections; (b) local-LLM labelling of the largest changes. If the primary fails the kill gate they are NOT run (no rescue search).
- **TRIALS:** primary adds 1. **Counter after registration: 8.**
- **Survivorship caveat applies** (same universe as P3): a positive result must be re-checked before it counts.

## Decisions (user, 2026-10-03)

- **Bets approved:** A (filing change), B (Kronos), C (LLM alpha mining). B and C were parked by default; they are now in scope, with their conditions to be agreed in the brainstorm (see "Brainstorm agenda" below).
- **Success bar approved as written:** beats the equal-weight baseline after costs on the sealed last-12-months holdout, development IC t ≥ 2, deflated Sharpe ≥ 0.95. Otherwise: "not found".
- **Compute:** local only (embeddings + a small local model on the RTX 3060). Hosted APIs are out of scope unless reopened after the Day 4 gate.
- **Build status:** nothing is built yet. Day 1 starts after the brainstorm.

## Brainstorm agenda (to settle together before Day 1)

1. **A — filing change.** Benefit: reuses our filing stack; mid/small caps may still hold the effect; orthogonal to price signals (good for Numerai). Loss/risk: plain version failed in replication; small caps cost more; local-only labelling is weaker. Decide: which sections, what counts as a "changer", which horizons.
2. **B — Kronos.** Benefit: genuinely new technique, feasible on a 6 GB GPU in its small size. Loss/risk: likely trained on our test period, so only a short post-cutoff window can be called clean; zero-shot candlestick models have underperformed simple baselines in published work. Decide: check the training cutoff first; use it as one feature, never as the headline.
3. **C — LLM alpha mining.** Benefit: the most "new research" of the three. Loss/risk: every candidate is a trial; AlphaAgent used strong hosted models, and a small local one will propose weaker factors. Decide: a hard cap on candidates (e.g. 30) and each one counted in TRIALS.
4. **Cross-cutting:** with three bets plus the control, the trial count rises, so the deflated Sharpe bar gets harder to clear. Do we cap total trials up front? Which bet gets the Day 4 kill gate first?

## Where we stand at the start (baseline numbers)

| Measure | Value |
|---|---|
| Signal strength (IC, composite_score vs forward return) | +0.093 pooled; per-period mean +0.075, std 0.260 |
| Walk-forward, top-25 Buy-only | 16.3% CAGR, Sharpe 0.93, max drawdown -22.7% |
| Walk-forward, no-signal equal-weight baseline | ~19.0% CAGR, Sharpe ~1.6, max drawdown -18.8% |
| S&P 500 buy-and-hold | ~21.3% CAGR, Sharpe ~1.0, max drawdown -18.9% |
| Rank-tilt (no hard exclusion) | 12.0% CAGR, Sharpe 0.93, max drawdown -19.5% |
| Per-call accuracy (canonical) | 38.6% vs 58.4% always-Buy |
| ML classifier, leak-fixed | 45.2% held-out; OVERVALUED precision 31.0% |

**Gap to goal:** the strategy loses to the no-signal baseline on every metric. The target is to beat the baseline after costs, on data we have not touched.

---

## The points

### P1. Can we trust the data? (survivorship, restated numbers, 5-year ceiling)
- **Meaning:** *Survivorship bias* = testing on today's index members, so companies that failed or were dropped silently vanish and results look better than reality. *Restated numbers* = yfinance gives the latest corrected financial statements, not what investors saw at the time. The *5-year ceiling* = yfinance only keeps about 5 years of statements, so we cannot test 2020 or 2022.
- **Stand:** all three are present. Known and documented, not fixed.
- **Plan:** (a) rebuild fundamentals from SEC EDGAR XBRL, which carries a `filed` date on every value (tested 2026-10-03: Apple returns 503 concepts, each with filing dates). (b) use point-in-time index membership lists. (c) re-run the existing walk-forward on the new data; the change in numbers is itself the "bias bill".
- **Known wrinkle:** the same line item changes concept names over time (e.g. `SalesRevenueNet` → `Revenues` → `RevenueFromContractWithCustomer...`), so we need a fallback map for ~15 core items.
- **Done when:** baseline re-run on as-filed, survivorship-aware data, with the delta reported.
- **Results:** _(fill)_

### P2. Is the signal strong and stable enough? (IC)
- **Meaning:** *IC (information coefficient)* = rank correlation between today's score and the return that follows. 0 means useless, +0.05 to +0.10 is a weak-but-real equity signal. *Stability* matters as much as size: our per-period IC swings from negative to positive (std 0.26).
- **Stand:** +0.093 pooled, unstable, DCF leg carries nearly all of it.
- **Plan:** every candidate signal is judged by IC on the development period, with a t-statistic across periods.
- **Pass line (pre-registered):** mean IC ≥ 0.03 and t ≥ 2 on development data.
- **Results:** _(fill)_

### P3. The boring control: proven factors (momentum, quality, low volatility)
- **Meaning:** *Factors* are characteristics with decades of evidence: **momentum** (recent winners keep winning), **quality** (profitable, stable firms do better), **low volatility**. Our score has none of them; our ML classifier's own feature importances showed momentum and volatility carrying real weight.
- **Stand:** computed for display (AlphaFactorsEngine), never used in the score.
- **Plan:** build 12-1 momentum, a profitability/Piotroski quality score, and low-vol on the new data; measure IC and a combined score. This is the bar any "new" idea must clear or add to.
- **Done when:** IC table for each factor and for valuation + factors combined.
- **Results:** _(fill)_

### P4. NEW signal A: semantic change in company filings ("Lazy Prices 2.0")
- **Meaning:** Companies quietly change wording in annual reports (new risk language, legal matters, tone) before the market reacts. The original paper measured this with word-overlap; the plain version no longer works on large caps (replication: no effect), and similarity was "saturated" at 0.996 because 10-Ks are mostly boilerplate.
- **Our twist:** (1) section-level, not whole-document: Risk Factors (Item 1A), Legal (Item 3), MD&A (Item 7); (2) *embedding distance* between consecutive years instead of word overlap; (3) an LLM reads only the biggest changes and labels them (risk-increasing / neutral / improving); (4) apply it to S&P 400/600 mid and small caps, where attention is thinner and the effect should survive longer; (5) use the `filed` date so it is point-in-time.
- **Why FinSight:** the filing ingestion, chunking and embedding stack already exists.
- **Kill gate (end Day 4):** if section-change IC < 0.02 or t < 1.5 on development data, drop it.
- **Risk:** small caps cost more to trade (P11).
- **Results:** _(fill)_

### P5. NEW signal B: time-series foundation model (Kronos) — parked by default
- **Meaning:** A model pre-trained on ~12 billion candlesticks from 45 exchanges (Kronos, AAAI 2026) used zero-shot as a forecaster or feature generator.
- **Problem:** its pre-training data likely overlaps our test period, so any backtest inside that period is partly in-sample for the model. Cannot be proved clean without a post-cutoff window, which is short.
- **Plan:** only if you want it, as a stretch. Check the training cutoff first; report only post-cutoff results.
- **Results:** _(fill)_

### P6. NEW signal C: LLM alpha mining (AlphaAgent-style) — parked
- **Meaning:** An LLM proposes formulaic factors, tests them, and keeps the winners.
- **Problem:** it is a machine for generating hundreds of trials, so it is the most exposed to fooling ourselves (see P9). Without a strict trial counter it manufactures false positives.
- **Plan:** park. Revisit only after P9 machinery exists.

### P7. Look-ahead inside the LLM itself
- **Meaning:** A language model trained on text up to 2024 may already "know" how a 2022 filing turned out. A backtest using it on 2022 data is contaminated, however clean the dataset.
- **Rule:** any LLM-derived signal uses either (a) documents filed after that model's training cutoff, (b) a chronologically consistent model (ChronoBERT/ChronoGPT), or (c) non-generative embedding distance only (low contamination risk). Check each model's cutoff and write it down.
- **Results:** _(fill)_

### P8. Combining signals and building the portfolio
- **Meaning:** A signal is not a portfolio. We learned that hard exclusion (Buy-only) throws away winners the model misjudges; tilting is safer. *Volatility management* means holding less when risk is high, which raised Sharpe 50–100% in published tests (Moreira & Muir).
- **Plan:** combine surviving signals with fixed equal weights (no fitting, to avoid overfit), then test: rank-tilt vs top-quintile, with and without volatility scaling, monthly vs quarterly rebalance.
- **Done when:** one chosen construction, picked on development data only.
- **Results:** _(fill)_

### P9. Not fooling ourselves (overfitting control)
- **Meaning:** Try enough variants and one will look great by luck. The *deflated Sharpe ratio* shrinks a result according to how many things you tried and how long the record is; above 0.95 is the usual bar for a real claim.
- **Plan:** the trial counter above, the sealed holdout, and a final DSR computed with the true trial count.
- **Pass line (pre-registered, to approve):** a result is "real" only if on the holdout it beats the equal-weight baseline after costs, development IC has t ≥ 2, and DSR ≥ 0.95. Otherwise we report "not found".
- **Results:** _(fill)_

### P10. Regimes and horizons
- **Meaning:** One 3-year bull market proves little. *Horizon mismatch* = a long-term fair-value signal (DCF) may be judged unfairly on 3-month holds.
- **Plan:** with EDGAR data, test 2012–2026 (including the 2020 crash and the 2022 bear market) and holds of 1, 3, 6, 12 months.
- **Results:** _(fill)_

### P11. Costs and capacity
- **Meaning:** Every trade pays a cost; small stocks cost more and move when you buy.
- **Stand:** flat 10 bps, an assumption we cannot calibrate.
- **Plan:** size-dependent costs (assumed, labelled as assumptions), a sensitivity table, and a turnover cap.
- **Results:** _(fill)_

### P12. A live, external record
- **Meaning:** A backtest is a claim; a live out-of-sample record is evidence. Ours has zero live calls because the server disk resets.
- **Plan:** log daily signals into Snowflake (now connected); submit the surviving signal to Numerai Signals, which scores out-of-sample and values signals that differ from everyone else's (a filing-text signal is a natural fit). Staking is real money: do not stake.
- **Results:** _(fill)_

---

## Day plan

| Day | Work | Points |
|---|---|---|
| 1 | EDGAR as-filed fundamentals pipeline; write the pre-registration | P1, P9 |
| 2 | Finish data (constituents, windows); factor IC on new data | P1, P10, P3 |
| 3 | Factor results; filing-section extraction and embeddings | P3, P4, P7 |
| 4 | Filing-change IC; **kill gate**; stretch signal only if approved | P4, P5 |
| 5 | Combine signals, portfolio tests, costs | P8, P11 |
| 6 | Deflated Sharpe, trial count, live logging setup | P9, P12 |
| 7 | One look at the sealed holdout; write-up; re-evaluate together | all |

## Reading list (for the brainstorm)

- Cohen, Malloy, Nguyen — *Lazy Prices*: https://www.nber.org/system/files/working_papers/w25084/revisions/w25084.rev0.pdf
- Public replication of Lazy Prices (no effect on S&P 100): https://github.com/iqueipopg/lazy-prices
- McLean & Pontiff — *Does Academic Research Destroy Stock Return Predictability?*: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365
- Bailey & López de Prado — *The Deflated Sharpe Ratio*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Gu, Kelly, Xiu — *Empirical Asset Pricing via Machine Learning*: https://academic.oup.com/rfs/article/33/5/2223/5758276
- He, Lv, Manela, Wu — *Chronologically Consistent LLMs*: https://arxiv.org/abs/2502.21206
- Kim, Muhn, Nikolaev — *Financial Statement Analysis with LLMs*: https://arxiv.org/abs/2407.17866
- Shi et al. — *Kronos*: https://arxiv.org/abs/2508.02739
- *AlphaAgent* (KDD 2025): https://arxiv.org/abs/2502.16789
- Moreira & Muir — *Volatility-Managed Portfolios*: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513
- Asness, Frazzini, Pedersen — *Quality Minus Junk*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432
- Asness, Moskowitz, Pedersen — *Value and Momentum Everywhere*: https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021
- Kong et al. — *Evaluating LLMs in Finance Requires Explicit Bias Consideration*: https://arxiv.org/abs/2602.14233
- *TradingAgents* (and its critique: one short backtest window): https://arxiv.org/abs/2412.20138
