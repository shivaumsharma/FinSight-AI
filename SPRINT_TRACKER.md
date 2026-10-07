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

**TRIALS so far (this sprint):** 38 (7 from P3; 1 from P4; 2 sector-neutral; 5 fix candidates; 4 alternatives; 3 DCF-stability measures; 2 round-2 stability tests; 2 horizon-matched ML tests; 2 filing-tone tests; 2 share-issuance tests; 3 earnings-surprise tests; 1 portfolio view of the learned model; 4 insider-purchase tests)

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

### ALTERNATIVES RESULT (2026-10-04; `research/vol_managed.py`, `research/ml_cross_section.py`; development only)

**A1 — learned cross-sectional model (37 out-of-sample test dates, 2016-06 to 2025-06; 47,337 rows; 27 features incl. sector dummies):**

| Model | Mean IC | t | Positive dates | Pass (IC >= 0.03 and t >= 2.5)? |
|---|---|---|---|---|
| HistGradientBoosting | +0.023 | +1.03 | 59% | **No** |
| Ridge | +0.026 | +1.02 | 57% | **No** |
| Production composite, same dates | +0.017 | +0.73 | n/a | n/a |

Best number in the whole sprint, but still within noise. Descriptive: top decile of the GBM's predictions returned 10.4%/qtr vs ~4.3% for the other nine (D9-D0 t = +1.44) — a right-tail, concentrated effect, not a smooth ranking. Ridge leans on small size (log market cap coefficient -0.039) and cheap book-to-market: a small-cap/value tilt, exactly the exposure survivorship bias flatters (small survivors that stayed in today's indices). Yearly GBM IC swings from -0.08 (2021) to +0.12 (2025).

**A2 — volatility-managed / trend-filtered S&P 500 (2007-2025, price basis, 5 bps cost):**

| Strategy | Period | Sharpe (strategy / buy-hold) | Max drawdown (strategy / buy-hold) |
|---|---|---|---|
| Vol-managed (15% target) | 2007-15 | 0.43 / 0.28 | -35% / -57% |
| | 2016-25 | 0.68 / 0.66 | -26% / -34% |
| Trend (200-day) | 2007-15 | 0.62 / 0.27 | -18% / -57% |
| | 2016-25 | 0.58 / 0.66 | -25% / -34% |

**Neither passes the pre-registered rule** (Sharpe +0.10 and drawdown 20% shallower in BOTH halves). Vol-managing cut drawdown 23-39% in both halves but its Sharpe edge vanished in the 2016-25 bull decade (+0.02). The trend filter was excellent in 2007-15 (the 2008 crash) and lagged in 2016-25. Reading: real drawdown protection, no free Sharpe in a bull market — a risk-control feature, not an alpha source.

**Trials: 19.** Nothing built.

### SUE RESULT — earnings-surprise drift (2026-10-04; `research/eps_pull.py`, `sue_ic.py`, `ml_cross_section_4q_sue.py`; 61,842 EPS rows, 1,014 companies)

| Test | Result | Pass? |
|---|---|---|
| S1 standalone, SUE vs next-quarter return (53 dates, median 818 names) | mean IC **-0.0017**, t -0.17, positive 49%, halves -0.007 / +0.004; top-minus-bottom quintile -0.08%/qtr | **No** |
| S2 added to the 4-quarter learned model: HistGradientBoosting | +0.0585 vs +0.0543 without (change +0.004), Newey-West t 1.73 | **No** |
| S2 Ridge | +0.0459 vs +0.0457 (change +0.000), Newey-West t 1.19 | **No** |

Reading: post-earnings drift is absent in this mid/large-cap universe, consistent with the literature that it has largely decayed there (it survives mainly in micro-caps, which this universe excludes). **Rejected.** Share issuance and earnings surprise are now both tested and rejected; the learned model's baseline edge (about +0.054 IC at 4 quarters) is unchanged by either. Untested new-scope signals: insider purchases (Form 4) and 13F holdings changes. Trials: 33.

### SUE PRE-REGISTRATION — earnings-surprise drift (written 2026-10-04, BEFORE any EPS was pulled or scored)

- **Idea:** stocks whose quarterly earnings beat their own seasonal pattern keep drifting up for a few months (post-earnings-announcement drift; Ball & Brown 1968, Bernard & Thomas 1989). Needs no analyst forecasts: surprise = this quarter's EPS minus the same quarter a year ago.
- **Data:** diluted EPS (fallback: basic) for 3-month periods from 10-Qs; Q4 = annual EPS (10-K) minus Q1+Q2+Q3. Each value is stamped with its filing date, so only filings dated before the test date are used.
- **Signal (fixed):** SUE = (EPS_q - EPS_{q-4}) / standard deviation of the last 8 year-over-year changes (need >= 6). At each quarter-end T use the latest SUE whose filing date is before T and within the previous 100 days; otherwise missing.
- **Tests (fixed):**
  - **S1 standalone:** per-quarter Spearman IC of SUE vs next-quarter total return over the 53 development dates (>= 100 names). Pass: mean IC >= 0.02 and t >= 2.5, positive in both halves.
  - **S2 as a feature:** add SUE to the A1b protocol (4-quarter target, HistGradientBoosting and Ridge, fixed settings). Descriptive: change in IC versus A1b on the same data; pass bar = the A1b bar (IC >= 0.03 and Newey-West t >= 2.5).
- **Trials: 3 (S1, S2-hgb, S2-ridge; counter -> 33).** Sealed holdout untouched.

### A1c RESULT — net share issuance as a feature (2026-10-04; `research/ml_cross_section_4q_issuance.py`; 37 test dates, 28 features)

| Model | 4q IC with `share_growth` | Without (A1b, same data) | Change | Newey-West t | Pass? |
|---|---|---|---|---|---|
| HistGradientBoosting | +0.0555 | +0.0543 | +0.0012 | +1.67 | **No** |
| Ridge | +0.0489 | +0.0457 | +0.0032 | +1.31 | **No** |

Net share issuance adds essentially nothing on top of the existing features (the model already captures size/value/quality; and the cover-page share counts are a noisy proxy for true buyback activity). Rejected as a standalone addition. Remaining new-scope signals (insider purchases from Form 4, earnings-surprise drift, 13F holdings changes) are untested; they need new data pulls. Trials: 30.

### A1c PRE-REGISTRATION — net share issuance as a new feature (written 2026-10-04, BEFORE it was run)

- **Idea (first of the "new scope" signals):** companies that shrink their share count (buybacks) tend to outperform ones that issue shares. Data already in hand: cover-page share counts from consecutive annual reports.
- **Feature (fixed):** `share_growth` = latest known annual share count / prior annual share count - 1 (diluted weighted shares if the cover count is missing). Values outside [-40%, +60%] are set missing: they are almost certainly stock splits that the two counts straddle, not real issuance.
- **Test:** the A1b protocol exactly (walk-forward, 4-quarter target, HistGradientBoosting + Ridge, fixed settings, first test 2016-06-30, Newey-West t) with `share_growth` added to the 16 base features. Data: survivors + the 38 former members merged so far (final rerun when the full set arrives).
- **Pass:** the A1b bar (mean IC >= 0.03 and Newey-West t >= 2.5), and, descriptively, the change in IC versus A1b on the same data. Trials: 2 (counter -> 30).

### MIROFISH PILOT — simulated-crowd feedback on FinSight (2026-10-05; sandbox `D:\Coding\Projects\_sandbox`, AGPL, never committed or shipped)

- **Setup:** local fork (Neo4j 5.26 on Java 17 + Ollama qwen2.5:7b + OASIS), all offline; Docker Desktop turned out to be uninstalled on this machine (not by me), so it was run natively. Seed = an honest description of FinSight including its failures. Source scanned first: no external URLs, no telemetry. Setup fixes needed: pip needs UTF-8 mode; `mcp` must be pinned to the lockfile's 1.24.0; the simulation silently downloads a ~440 MB BERT recommender from HuggingFace on first run (this looked like a hang).
- **Result:** 16 agents, 12 simulated hours, 70 posts, 2 likes, 0 comments, ~4 minutes. **Only 4 of the 16 "people" were human personas;** the other 12 were corporate accounts extracted from names in the text (FinSight, SEC Filings, Home Depot, Apple, Microsoft). After the 4 seeded opening posts, agents mostly echoed each other ("Transparency is key..."), with role confusion (a "FinanceStudent" posting in Microsoft's voice, a "HobbyistQuantTrader" speaking as FinSight).
- **What the four seed statements said (the only independent content):** retail investors like the honesty; skeptics and quants say 38.6% directional accuracy undermines usefulness; journalists will focus on credibility.
- **Control (one direct call to the same model):** produced distinct per-audience praise, criticism, adopt/not and what-would-make-them-pay, plus a single most damaging objection (the 27% Sell accuracy and 38.6% directional accuracy), in about a minute. **The swarm added nothing over one prompt.**
- **Verdict:** not usable as an indicator of what real people feel (the opinions are model-generated and unvalidated) and not worth its cost over a single prompt; keep a single-prompt "synthetic audience" only as a brainstorming aid, and get 5-10 real users for actual evidence. Not tested as a stock-sentiment indicator (it cannot be backtested: P7 look-ahead; the only fair test is the forward live record). Services stopped; restart with the paths in the sandbox if wanted.

### SEALED HOLDOUT — RESULT (2026-10-07; `research/holdout_run.py eval`; ONE-SHOT, NOW CONSUMED; frozen-model SHA-256 verified before use)

Four quarters: 2025-09-30, 2025-12-31, 2026-03-31, 2026-06-30 (about 995 stocks scored per date; top decile = 99 names; 25 bps one-way cost).

| HistGradientBoosting (primary) | Gross top-decile | Benchmark (equal weight) | Net excess | Rank IC (1q) |
|---|---|---|---|---|
| 2025-09-30 -> 12-31 | +5.0% | +2.2% | **+2.32 pts** | -0.054 |
| 2025-12-31 -> 03-31 | -1.1% | +1.3% | **-2.53 pts** | -0.183 |
| 2026-03-31 -> 06-30 | +23.7% | +13.5% | **+10.06 pts** | +0.161 |
| 2026-06-30 -> 09-30 | -3.0% | -3.5% | **+0.30 pts** | +0.083 |

- **Pre-registered verdict (primary, HGB): CONFIRMED (directionally, not proven):** mean net excess **+2.54 pts per quarter** (about +10% a year), beats the benchmark in **3 of 4** quarters, total +23.5% net vs +13.4% for the benchmark. **Ridge (secondary): INCONCLUSIVE** (beats in 2 of 4; mean +4.19 pts/qtr, driven by two quarters).
- **Why this is NOT strong evidence (stated plainly, and decisive for how it may be cited):**
  1. **One quarter carries it.** 2026-03-31 -> 06-30 alone contributes +10.06 pts in a quarter when the benchmark itself rose +13.5%. **Without that quarter the mean excess is +0.03 pts per quarter, i.e. zero.**
  2. **The model did not rank stocks better on average:** mean 1-quarter rank IC **+0.0018** (negative in 2 of 4 quarters, -0.18 in one), and the 4-quarter IC on the one date with a complete 4-quarter return is **-0.082** (the top decile still beat the benchmark by +4.0 pts that year). A top-decile portfolio can outperform in a rally through a size/volatility tilt without any ranking skill; that is the most plausible reading given the development-period small-cap tilt.
  3. With four quarters no statistical significance is possible, and the full P9 bar (development t >= 2, deflated Sharpe >= 0.95) was already unmet.
- **What may be claimed:** the development result was not contradicted by fresh data; the model is **not proven** and has **no demonstrated stock-selection skill**. **What may not be claimed:** that the model "works", that +5-7% a year is real, or the +10% annualised holdout figure as a track record.
- **The holdout cannot be reused** (any new model would need its own untouched data). The only remaining clean evidence is the weekly live record; a frozen-model entry in it has not been built.

### SEALED HOLDOUT — PRE-REGISTRATION (user authorised opening it on 2026-10-07; written BEFORE any holdout number was computed; this is a ONE-SHOT test)

- **What is tested:** the frozen learned 4-quarter model (`frozen_model_2025-06-30.joblib`, SHA-256 `ec8b1e25e19ca3d803b2fe157419ff9600a09b1b29ea9c60a1cdb1261507791d`; integrity re-verified before use). No retraining, no tuning, no feature or definition changes. The primary model is the HistGradientBoosting one; Ridge is reported as secondary.
- **Holdout dates:** the four quarter-ends after the development period with a full next-quarter return available: 2025-09-30, 2025-12-31, 2026-03-31, 2026-06-30 (returns to 2025-12-31, 2026-03-31, 2026-06-30, 2026-09-30). Features come from the same code and the same data sources (EDGAR filings dated before each date, production-model scores computed for those dates, prices up to each date); delisted names are included with terminal returns.
- **Primary test (fixed now):** the portfolio check exactly as in development: each date hold the top decile by predicted score (at least 10 names), equal weight, for one quarter; benchmark = equal-weight of every stock with a prediction that date; cost = 25 bps one way on the replaced fraction of the book (turnover measured quarter to quarter).
- **Interpretation rule (fixed now):** with only 4 quarters no result can be statistically significant, so the outcome is classed only as follows. **CONFIRMED (directionally, not proven):** mean net quarterly excess return >= +0.5 percentage points AND the top decile beats the benchmark in at least 3 of 4 quarters. **NOT CONFIRMED:** mean net excess <= 0 OR it beats the benchmark in at most 1 of 4 quarters. **INCONCLUSIVE:** anything else. (Development average was about +1.5 points per quarter gross.)
- **Descriptive only:** rank IC of the predictions vs the next-quarter return on each date; the 4-quarter IC and top-decile 4-quarter excess for the single date (2025-09-30) whose 4-quarter return is complete; Ridge; decile table.
- **Not claimed whatever happens:** the full P9 bar (development t >= 2 and deflated Sharpe >= 0.95) is already unmet (Newey-West t 1.57, DSR 0.78-0.82), so even CONFIRMED means "the development result was not contradicted by fresh data", not "proven edge". The holdout is consumed by this run; no second look, no repeat with changed settings.

### INSIDER RESULT — open-market purchases by officers/directors (2026-10-07; `research/insider_pull.py`, `insider_ic.py`, `ml_cross_section_4q_insider.py`; 342,823 purchases, 8,617 issuers, 2012-01 to 2025-06)

| Test | Result | Pass? |
|---|---|---|
| I1 purchase value / market cap, vs next-quarter return (53 dates; 25% of stock-dates have a purchase in the prior 180 days) | mean IC **+0.0105**, t +1.32, positive 51%, halves +0.011 / +0.010 | **No** |
| I2 number of distinct insider buyers | mean IC **+0.0078**, t +1.04, halves +0.006 / +0.010 | **No** |
| (descriptive) stocks WITH a purchase minus WITHOUT, next quarter | **+0.04%/qtr**, t +0.10 | n/a |
| I1+I2 as features in the learned 4-quarter model, HistGradientBoosting | +0.0472 vs +0.0519 without (worse), Newey-West t 1.36 | **No** |
| Same, Ridge | +0.0461 vs +0.0453 (+0.0008), Newey-West t 1.18 | **No** |

Reading: right-signed but statistically empty in this mid/large-cap universe; the insider effect documented in the literature is concentrated in small/illiquid stocks and in larger "cluster" purchases than a simple 180-day aggregate captures. **Rejected as specified.** 13F holdings changes: dropped (needs a CUSIP-to-ticker map not available free), recorded as not attempted, not as tested. Trials: 38. **All four candidate "new-scope" signals (share issuance, earnings surprise, insider purchases; 13F untested) are now closed: none improves on the learned model's baseline.**

### INSIDER PRE-REGISTRATION — open-market insider purchases (written 2026-10-07, BEFORE any insider data was downloaded)

- **Idea:** executives and directors buying their own company's stock with their own money has been associated with higher later returns for decades (e.g. Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003). Free, date-stamped data: SEC's bulk "Insider Transactions Data Sets" (Forms 3/4/5, 2012Q1-2025Q2 only, about 540 MB; nothing from the sealed holdout period is downloaded).
- **Definition (fixed):** an open-market purchase = transaction code `P`, acquired, on Form 4 (or 4/A), by a reporting owner flagged Officer or Director, price > 0, value = shares x price. At each test date T use filings dated within the 180 days BEFORE T.
- **Signals (fixed):** I1 = total purchase value / market cap at T. I2 = number of distinct insiders who bought (cluster buying). Stocks with no purchase get 0.
- **Tests (fixed):** I1-q and I2-q: per-quarter Spearman IC vs next-quarter return, 53 dates, >= 100 names; pass = mean IC >= 0.02, t >= 2.5, positive in both halves (2012-18, 2019-25). I1-f and I2-f: both added as features to the A1b protocol (HistGradientBoosting and Ridge; 4-quarter target); pass = the A1b bar (IC >= 0.03 and Newey-West t >= 2.5); descriptive: change in IC versus A1b on the same data. Also descriptive: average next-quarter return of stocks with any purchase vs none.
- **Trials: 4 (counter -> 38).** Holdout untouched. 13F holdings changes are NOT attempted: they need a CUSIP-to-ticker map that is not available free, and the signal is weak in this universe; recorded as dropped, not as tested.

### FROZEN MODEL (2026-10-07; `research/freeze_model.py`) — the learned 4-quarter model, locked

- Trained exactly as the last walk-forward retrain: every row whose 4-quarter label was fully realised by 2025-06-30 (**46,323 rows, dates 2012-06-29 to 2024-06-28**), HistGradientBoosting (max_depth 3, learning_rate 0.05, 200 iterations, min_samples_leaf 200, l2 1.0, seed 0) and Ridge (alpha 10); 28 columns (16 rank-normalised base features + sector dummies); target = per-date percentile rank of the next-4-quarter return.
- **File `research/data/frozen_model_2025-06-30.joblib`, SHA-256 `ec8b1e25e19ca3d803b2fe157419ff9600a09b1b29ea9c60a1cdb1261507791d`.** It reproduces the walk-forward predictions for 2025-06-30 with correlation 1.000000 across 1,000 stocks.
- Rule: no retuning, refitting or feature changes before the holdout / live evaluation; any change creates a NEW model with a new hash and a new trial. The sealed holdout (dates after 2025-06-30) is still unopened.

### P8/P9/P11 — PORTFOLIO CHECK of the learned 4-quarter model (2026-10-05; `research/learned_portfolio.py`; DESCRIPTIVE, development data only; trials counted: 34)

Walk-forward predictions (made only with data known at each date) -> hold the top decile of predicted stocks, equal-weight, for one quarter, re-rank every quarter; benchmark = equal-weight of every stock scored that date. 37 quarters, 2016-06 to 2025-06. Cost = one-way bps on the replaced fraction of the book (average turnover 38%).

| Model / cost | Annual return | Benchmark | Excess/yr (t) | Sharpe | Bench Sharpe | Max drawdown | Bench DD | Deflated Sharpe (33 trials) |
|---|---|---|---|---|---|---|---|---|
| HistGradientBoosting, 0 bps | 25.9% | 18.4% | +6.8% (t 2.34) | 1.12 | 0.89 | -24.3% | -28.1% | 0.82 |
| HGB, 10 bps | 25.5% | 18.4% | +6.5% (t 2.24) | 1.11 | 0.89 | -24.5% | -28.1% | 0.81 |
| HGB, 25 bps | 25.0% | 18.4% | +6.0% (t 2.08) | 1.09 | 0.89 | -24.7% | -28.1% | 0.80 |
| HGB, 50 bps | 24.1% | 18.4% | +5.2% (t 1.82) | 1.06 | 0.89 | -25.0% | -28.1% | 0.78 |
| Ridge, 10 bps | 23.5% | 18.4% | +4.6% (t 1.50) | 1.06 | 0.89 | -31.7% | -28.1% | 0.79 |

- HGB beat the equal-weight benchmark in **68% of quarters** and in 8 of 10 calendar years (negative only in 2020 and 2021).
- **This is the strongest evidence in the project, and it is NOT proven.** Deflated Sharpe is 0.78-0.82 against the pre-registered 0.95 bar once the 33 variants tried are accounted for; the development IC t-statistic (Newey-West 1.57) is below 2; and the portfolio result is a descriptive view of the same model, not an independent test.
- **Why to be careful:** (1) the ridge coefficients show a small-company tilt (log market cap -0.036, book-to-market -0.029), the exposure most flattered by remaining survivorship (still ~19-24% of 2012-2016 index members missing) and the most expensive to trade; (2) the top-decile result is a concentrated tail effect; (3) results by year swing (-2.7% to +12.9% excess); (4) the 10-50 bps cost assumptions are not calibrated for small stocks.
- **The sealed holdout (dates after 2025-06-30, now about four quarter-ends of real out-of-sample data) has NOT been opened.** It can be used once. The right next step is to freeze the model exactly as it is and open the holdout for this single test, plus keep the weekly live record running; that decision is the user's.

### G1 FINAL-FINAL (2026-10-05) — with verified manual additions: 108 former members

- 16 hand-supplied SEC IDs were run through the same automatic checks; **10 passed** (Monsanto, Time Warner, Time Warner Cable, GGP, Harman, AGL Resources, Green Mountain/Keurig, Heinz, Level 3, Nielsen), 2 became duplicates of companies already held (Chesapeake/Expand, HCP/Healthpeak), and **4 were rejected by the checks** (FLIR, Frontier, Pall, Walgreens) and stay excluded. Still REVIEW: AGN, ALTR, ARNC, CA, DISCK, DTV, MYL, PGN (reused symbols / successor entities, deliberately skipped). NOMATCH: BMS, DISCA, PX, SIAL, VIAC, WRK.
- **Index coverage (Jan-1 members with prices):** 2012 **76%**, 2016 **81%**, 2020 **88%**, 2024 **93%** (originally 63/69/81/91).
- **Results unchanged within noise:** composite IC +0.0086 (t 0.49) all dates, +0.0148 (t 0.74) out of sample 2012-2022; factors none alive, P3 FAIL; learned 4-quarter model hgb **+0.0519** (Newey-West t 1.57), ridge +0.0453 (t 1.15) versus composite +0.0086 on the same dates; learned 1-quarter hgb +0.0187 / ridge +0.0246.
- **Final reading:** the survivorship correction trimmed the learned model's 4-quarter edge from +0.060 to +0.052 (about 13%) and left every verdict unchanged. This is the version to cite.

### G1 FINAL RESULT — survivorship-corrected rerun with 98 former members (2026-10-05; logs `research/data/final_rerun.log`)

- **Coverage of the index:** matched 98 of 130 downloaded former S&P 500 members (23 REVIEW, 6 NOMATCH, 3 DUP). Share of January-1 index members with price history: **2012 63% -> 76%, 2016 69% -> 79%, 2020 81% -> 87%, 2024 91% -> 93%**. The rest cannot be recovered with free data: names Tiingo does not list (52 of 183), the 75 reused-symbol names, and REVIEW/NOMATCH cases.
- **REVIEW (need a human, never auto-used):** AGN, ALTR, ARNC, CA, DISCK, DTV, FLIR, FTR, GAS, GGP, GMCR, HAR, HCP, HNZ, LVLT, MON, MYL, NLSN, PGN, PLL, TWC, TWX, WBA. **NOMATCH:** BMS, DISCA, PX, SIAL, VIAC, WRK. Many are well-known companies (Monsanto, Time Warner, Walgreens) that can be added with a verified manual CIK in `delisted_overrides.csv`.

| Test | Survivors only | Final (98 former members added) |
|---|---|---|
| Composite IC, all 53 quarters | +0.0106 (t 0.62) | **+0.0086 (t 0.49)** |
| Composite IC, 2012-2022 out of sample | +0.0164 (t 0.84) | +0.0148 (t 0.74) |
| Sector-neutral composite, all dates | +0.0083 (t 0.55) | +0.0079 (t 0.51) |
| P3 factors | none alive | none alive (valuation_only +0.001, multi_factor -0.021, FAIL) |
| Learned model, 4-quarter IC (HistGradientBoosting) | +0.0600 (NW t 1.72) | **+0.0513 (NW t 1.53)** |
| Learned model, ridge, 4q | +0.0478 (NW t 1.20) | +0.0453 (NW t 1.15) |
| Composite, 4q, same dates | +0.0153 | +0.0095 |
| Learned model 1-quarter (hgb / ridge) | +0.0233 / +0.0260 | +0.0156 / +0.0242 |

**Reading:** survivorship bias had been flattering every result a little, as expected: the learned model's 4-quarter edge shrank by about 15% (0.060 -> 0.051) and the composite's IC fell slightly. No verdict changes: the composite and the classic factors still have no demonstrable edge, and the learned 4-quarter model remains the best but unproven candidate (below the t >= 2.5 bar). The earlier conclusions were robust to the correction.

### G1 SURVIVORSHIP FIX — PREVIEW with 38 of ~131 former members added (2026-10-04)

**Pipeline built and verified** (`research/tiingo_prices.py` -> `delisted_match.py` -> `delisted_build.py` -> `pit_composite.py run_new`; shared guard `research/delisting.py`):
- Matching is by company NAME with five automatic checks (name similarity, a 10-K near the end of index membership, plausible market cap at membership-era filings, 10-Ks across >= 70% of member years, first 10-K no later than a year after the first member year); anything else goes to REVIEW, never guessed. Of the 49 names available: 38 AUTO, 3 DUP (share classes already held), 6 REVIEW (AGN, ALTR, ARNC, CA, DISCK, DTV: reused symbols or successor entities, correctly refused), 2 NOMATCH. An earlier version wrongly matched Avon to "TX Rail Products"; fixed by requiring name similarity >= 0.75 and judging the 10-K timing against the end of membership.
- Delisted stocks keep their terminal return (last real price carried forward in the adjusted-close matrix) and are masked from every price-based feature and score after their last real trading day, so the fix cannot leak stale prices.
- Data checks: Aetna's last price $212.70 vs the $207 deal price; Citrix's $18.57 and Altaba's $51.50 special distributions handled continuously; Jan-1 index coverage 63% -> 67% (2012), 69% -> 73% (2016), 81% -> 83% (2020) with only 38 tickers in.

| Test | Survivors only | + 38 former members |
|---|---|---|
| Composite IC, all 53 quarters | +0.0106 (t 0.62) | +0.0101 (t 0.59) |
| Composite IC, 2012-2022 out of sample | +0.0164 (t 0.84) | +0.0161 (t 0.83) |
| P3 factors (momentum, low vol, EY, FCF yield, ROA, combinations) | all ~0 | all ~0 (valuation_only -0.000, multi_factor -0.023) |
| Learned model, 4-quarter IC (hist-gradient-boosting) | +0.0600 (NW t 1.72) | **+0.0543** (NW t 1.59) |
| Learned model, ridge | +0.0478 (NW t 1.20) | +0.0457 (NW t 1.17) |

Reading: the conclusions are robust to the first 38 additions; the learned model loses about a tenth of its edge, the direction survivorship bias predicts. Not final: ~93 more tickers are still downloading (Tiingo free tier ~50 requests/hour).
**To finish when the download completes:** `python research/delisted_match.py && python research/delisted_build.py && python research/pit_composite.py run_new`, merge `p2b_scores_delisted.csv` into `p2b_scores.csv` (original saved as `p2b_scores_survivors.csv`), then rerun `pit_composite_ic.py`, `factor_ic.py`, `ml_cross_section.py`, `ml_cross_section_4q.py`. Originals of every changed data file are kept as `*_pre_delisted.*`.

### G1 DEEP SCAN RESULT (2026-10-04; `research/g1_probe.py`, public Tiingo ticker list)

- 817 tickers were ever in the S&P 500 since 2012; 549 are in our price matrix, 268 are not. yfinance returns >1 year of history for 75 of the 268 (all still trading: renamed/outside our universe) and **nothing for 183 truly delisted names** (TWTR, ATVI, CELG, SIVB, AET, AGN...).
- **Size of the bias:** a survivor-only equal-weight portfolio of Jan-1 members beat the equal-weight S&P 500 ETF (RSP) by **+1.9 pts/year on average** (2012-2024; +1.9 in 2012-16, +1.3 in 2020-24). Missing share by year: 36% (2012) -> 9% (2024); adding the 75 recoverable names would cut it to 28% -> 4%.
- **Sources checked:** Stooq (blocked, returns an HTML gate); yfinance (nothing for delisted); EDGAR (no prices; ticker->CIK lookup is unsafe, e.g. ALTR resolves to a different company); Financial Modeling Prep and EODHD (free-tier limits unverifiable, pages blocked); Nasdaq Data Link WIKI (stale after 2018, needs an account); **Tiingo: its public supported-tickers list covers 131 of the 183 (105 with history before mid-2012); the unlisted ones are mostly renames we hold under a new symbol.**
- **Verdict: NOT exhaustive.** One real route remains: a free Tiingo API key (user must create it) for prices, plus name-based matching of delisted companies to SEC filer IDs with manual verification (symbol reuse makes ticker-based lookup unsafe). Estimated effort: half a day plus about an hour of rescoring. Everything else is exhausted.

### G8 — BUILT: `app/analysis/vol_overlay.py`, `app/reporting/portfolio_risk_overlay.py`, `GET /v1/portfolio/risk-overlay`; 14 offline tests; live check on AAPL/MSFT/JPM/NVDA gave 15.9% realised vol -> 94% suggested exposure; exposure formula matches the research backtest exactly. Drawdown protection only, stated in the response. UI not wired yet.

### G7 RESULT — filing tone change (2026-10-04; `research/tone_change_ic.py`; 1,531 filings, 13 filing-year groups)

| Signal | Mean IC | t | Years positive | Halves | Pass? |
|---|---|---|---|---|---|
| T1 change in LM-negative word share | +0.022 | +1.00 | 69% | +0.048 / +0.001 | **No** |
| T2 change in LM-uncertainty word share | +0.001 | +0.05 | 46% | -0.028 / +0.026 | **No** |

T1 has a faint, right-signed effect that is statistically indistinguishable from noise and is gone in the second half of the sample (the usual post-publication decay pattern). **Rejected; no variants run.** Combined with P4 (sentence-level change) the filing-text idea is exhausted on this data. Trials: 28.

### G7 PRE-REGISTRATION — filing tone change (written 2026-10-04, BEFORE any tone was computed)

- **Source of the word list:** Loughran-McDonald master dictionary (`LM.csv`, 6.7 MB, extracted from the `pysentiment2` 0.1.1 PyPI wheel; 2,355 negative and 297 uncertainty words). Not installed as a dependency; only the data file is used.
- **Data / design:** identical to P4 (same 150-company sample, the same 1,548 year-over-year 10-K Item 1A pairs, same filings <= 2025-06-30, same outcome: abnormal return from 2 trading days after filing to +63 trading days, same grouping by filing quarter, >= 30 filings per group).
- **Signals (fixed):** T1 = change in the share of Item 1A words that are LM-negative versus the company's prior-year filing; score = -T1 (more negative tone predicts lower return). T2 = same with LM-uncertainty words; score = -T2.
- **Pass line (trial-adjusted):** mean IC >= 0.02 and t >= 2.5 across >= 12 filing-year groups, positive in both halves of the sample period. Otherwise the tone idea is rejected; no variants (no level-of-tone, no other categories).
- **Trials: 2 (counter -> 28).**

### G5 RESULT — horizon-matched retest A1b (2026-10-04; `research/ml_cross_section_4q.py`; 37 test dates 2016-06..2025-06, 43k training rows by the end)

| Model | 4-quarter mean IC | naive t | Newey-West t (lag 3) | Pass (IC >= 0.03 and NW t >= 2.5)? |
|---|---|---|---|---|
| HistGradientBoosting | **+0.060** | +2.62 | +1.72 | **No** |
| Ridge | +0.048 | +1.81 | +1.20 | **No** |
| Production composite, same dates | +0.015 | n/a | +0.45 | n/a |

Best result of the sprint: the mean IC is 4x the composite's and clears 0.03, and the learned model beats the hand-built score at the horizon the classifier was labelled on. But the overlapping-window-corrected t-statistic does not reach 2.5, and yearly IC is unstable (2019 +0.21, 2021 -0.17, 2022 +0.22). Survivorship (size/value tilt) still flatters it. **Verdict: not promoted; "display-only" status stands.** It is the right challenger to follow live: freeze this model and feature set, and log its scores alongside the composite in the G6 live record so a clean out-of-sample record accrues. Trials: 26.

### G5 — why the ML classifier is display-only, and the one fair retest (pre-registered 2026-10-04, BEFORE running)

**Why display-only (already measured, EVALUATION.md section 4):** wiring it into the composite as a blended `ML_WEIGHT` moved leave-one-window-out accuracy 44.1% -> 44.8% (flat across weights 0-0.5); gating Sell/Buy calls on it moved accuracy -3.8, -15.8 and -1.4 points; held-out accuracy 45.2% and F1-macro 0.380 on 1,700 rows; its one real edge is OVERVALUED precision 31.0% vs 27.1%. Rule: a signal graduates from display-only only if it clears a pre-registered out-of-sample pass line. It has not. A1 (47k point-in-time rows, 1-quarter horizon) also missed it (IC +0.023, t 1.0).

**Fair retest — horizon-matched (A1b):** the classifier's labels are 12-month outcomes, but A1 predicted 1 quarter. Same A1 features, models and walk-forward protocol (HistGradientBoosting and Ridge, fixed settings), but target and evaluation = next **4 quarters'** total return (cross-sectional rank), first test date 2016-06-30, training only on rows whose 4-quarter label was fully realised before each retrain date. Test dates overlap, so the t-statistic uses a Newey-West correction (lag 3). Pass: mean IC >= 0.03 AND Newey-West t >= 2.5. Trials: 2 (counter -> 26). If it passes, the classifier is re-trained on the EDGAR panel and re-tested in the composite under the same leave-one-window-out protocol before any wiring; if not, the "display-only" status is final.

### G3 ROUND-2 RESULT (2026-10-04; `research/g3_round2.py`) — G3 DECLARED EXHAUSTED

| Test | Result | Pass? |
|---|---|---|
| G3b: Buy&stable minus Buy&unstable | +0.78 pts/quarter, t = +1.60 (53 dates); halves -0.06 / +1.65; mean excess: stable +0.34, unstable -0.43, all Buy +0.10 %/qtr | **No** (t < 2.5, first half negative) |
| G3c: composite IC without the top-|upside| quintile | +0.0113 (t +0.72); halves +0.0070 / +0.0157 | **No** |

Reading: the stable-vs-unstable Buy gap is directionally sensible but not statistically distinguishable from noise and absent in 2012-18; removing the extreme-upside fifth does not rescue the ranking. The DCF's own uncertainty measures carry no reliable information about returns, in either shape of test. Both pre-registered rounds used; no further variants. Existing Monte Carlo / sensitivity flags remain informational (and the MC width adds nothing about realised risk beyond trailing volatility, partial corr +0.007). Trials: 24.

### G3 DIAGNOSIS and ROUND-2 PRE-REGISTRATION (written 2026-10-04, BEFORE round 2 was run; `research/g3_diagnose.py`)

**What went wrong in round 1 (measured):**
1. **Stability is a proxy for extreme upside** (per-date Spearman with |upside|: S1 +0.58, S2 +0.72), and the most extreme-|upside| fifth of calls has IC **-0.019** (t -1.0), the other four fifths +0.007 to +0.020. High "conviction" = extrapolation artefacts, so confidence was measuring the wrong thing.
2. **Wrong test shape:** round 1 measured rank IC *inside* each group. The between-group table shows something it could not see: among names the DCF calls cheap, stable ones returned +0.36% excess per quarter vs -0.55% for unstable (n = 11,574 vs 6,384); among names called rich, no difference (-0.09% vs +0.03%).
3. **The Monte Carlo width is not a risk indicator beyond trailing volatility:** partial correlation with realised |move| after removing 252-day volatility = +0.007 (t 0.9); raw +0.027 against +0.249 for trailing volatility itself. The MC spread is model-internal parameter uncertainty, not information about the future price.

**Round 2 (exactly two tests; if both fail, G3 is declared exhausted):**
- **G3b — stability veto on Buy calls:** Buy = production rule (composite >= BUY_THRESHOLD). Per date, mean next-quarter excess return (vs the date mean) of Buy&stable (S1 >= 0.8) minus Buy&unstable. Pass: mean >= +0.5 pts/quarter, t >= 2.5 over >= 30 dates with >= 20 names in each group, AND positive in both 2012-18 and 2019-25.
- **G3c — drop the extreme-conviction fifth:** composite IC on names outside the top-|upside| quintile of each date. Pass: IC >= 0.03, t >= 2.5, positive in both halves.
- **Trials: 2 (counter -> 24).** Sealed holdout untouched.

### G3 RESULT — DCF stability (2026-10-04; `research/pit_stability.py`, `research/g3_stability_ic.py`; 26,549 scored ticker-dates, 53 dates)

| Measure | Stable group IC (t) | Unstable group IC (t) | Paired stable - unstable (t) | Pass? |
|---|---|---|---|---|
| S1 (Monte Carlo agrees with base verdict) | +0.006 (0.33) | +0.010 (0.56) | -0.004 (-0.19) | **No** |
| S2 (sensitivity-grid agreement) | +0.007 (0.40) | +0.022 (1.08) | -0.018 (-0.84) | **No** |
| S3 (MC prob_undervalued as a score) | +0.015 (0.95) | n/a | n/a | **No** |

73% (S1) / 81% (S2) of calls are "stable". **The DCF's own confidence does not predict returns:** if anything the unstable calls do slightly better. Consequence: the existing confidence flags (Monte Carlo CI, sensitivity grid) stay what they already are — informational model-fragility indicators with no predictive claim; no new flag or abstention rule is justified. Trials: 22.

### GAP-FILLING PROGRESS (2026-10-04)

- **G1 survivorship:** confirmed unfixable with free data (Stooq blocked; yfinance returns 0 rows for TWTR, ATVI, CELG, SIVB). Handled by labelling every single-stock historical result an upper bound (EVALUATION.md) and by G6.
- **G2 EDGAR fundamentals — BUILT:** `app/data/edgar_fundamentals.py` + fallback wiring in `MarketDataLoader` (`FUNDAMENTALS_SOURCE=auto|yfinance|edgar`, default auto = EDGAR only when yfinance's statements fail). 16 offline tests; 57 tests pass across the touched areas. Live check vs yfinance (AAPL, IBM, HD, 4 fiscal years): revenue, net income, operating cash flow and capex identical; shares within 0.2%; total debt gap 8-10% after adding lease liabilities (was 17-24%). Limit: if yfinance is down for company info too, `market_data_tool` still raises before statements are requested, so this covers statement-level failures and the forced-EDGAR mode, not a full yfinance outage.
- **G3 DCF stability — TESTED TWICE, EXHAUSTED:** see G3 RESULT and G3 ROUND-2 RESULT.
- **G4 Sell side — already covered:** reports already carry per-sector, per-rating calibrated accuracy (`calibrated_confidence.py`) and confidence flags (Monte Carlo CI, sensitivity grid, DCF/relative disagreement). No rating-logic change is justified by the evidence.
- **G5 ML classifier — keep display-only:** A1 shows no significant edge even with 47k rows.
- **G6 live record — BUILT:** `scripts/live_record.py` (`snapshot` appends a weekly, append-only score record; `evaluate` joins matured snapshots to realised returns and refuses to print a t-stat before 8 matured dates). 7 offline tests. Not yet scheduled: a GitHub-runner schedule would hit the yfinance datacenter throttle, so run `snapshot` from a machine you control (laptop or the droplet) and commit `live_record/snapshots.csv`.
- **G7, G8 deferred.**

### GAP-FILLING PLAN (user request 2026-10-04: go through every architecture entry, find the gap, fix it)

| # | Component | Gap (measured) | Fix | Status |
|---|---|---|---|---|
| G1 | Data layer | Survivorship: 37% of 2012 S&P 500 members missing; Stooq blocked, yfinance has none of the acquired names (TWTR/ATVI/CELG/SIVB = 0 rows) so no free fix exists | Cannot be removed; bound it. Label every single-stock historical number an upper bound; prefer index-level tests; start the forward record (G6) | doc + G6 |
| G2 | market_data_tool | yfinance: ~5 restated years, throttled, single point of failure | EDGAR as-filed fundamentals provider behind a flag (default off), yfinance fallback, tests | build |
| G3 | valuation_tool | DCF verdict flips on small input changes | Stability measures from the existing Monte Carlo + sensitivity grid; test if stable calls predict better; surface a flag only if they do | test running |
| G4 | Composite / Sell side | Bottom deciles no worse than the middle: Sell carries no information | Confidence labelling; no rating-logic change without evidence | check |
| G5 | ML classifier | 1.5k rows; EDGAR panel gives 47k but A1 shows no significant edge | Keep display-only; document | doc |
| G6 | news / consensus / everything untestable historically | No free historical data to validate; no survivorship-free evidence | Daily point-in-time score snapshots (forward record), immune to look-ahead and survivorship | build |
| G7 | rag / sentiment | Never tested against returns | Tone-change test on the 1,548 downloaded filings; needs a word list download -> ask first | deferred |
| G8 | Portfolio / paper trading | Vol-managed overlay cut drawdown 23-39% but no Sharpe gain | Optional risk-control view | deferred |

### G3 PRE-REGISTRATION — DCF stability (written 2026-10-04, BEFORE the rerun)

- **Run:** the same 984 tickers x 53 dates through the unchanged production scorer, now also capturing `valuation_results` (Monte Carlo statistics and the 5x5 WACC x terminal-growth sensitivity grid).
- **Measures (fixed):** S1 = share of Monte Carlo draws on the same side of the market price as the base DCF verdict. S2 = share of the 25 sensitivity cells on the same side of the price as the base verdict. S3 = Monte Carlo `prob_undervalued` used directly as a score.
- **Hypothesis:** calls the DCF itself is confident about carry more information. Test: split each date's names into stable (S >= 0.8) and unstable (S < 0.8); compare composite IC.
- **Pass line:** stable-group IC >= 0.03, t >= 2.5 (>= 100 names/date and >= 30 dates with a stable group of >= 40 names) AND the paired stable minus unstable IC difference has t >= 2. S3 passes on the standard IC >= 0.03, t >= 2.5. Sealed holdout untouched.
- **Trials:** S1, S2, S3 = 3 (counter -> 22).
- **If it passes:** surface the flag (and abstain/lower confidence on unstable calls) in the product. **If not:** the flag may still ship as a clearly-labelled *uncertainty indicator* (it describes model fragility, not return prediction) but no claim of predictive value is made.

### ALTERNATIVES PRE-REGISTRATION (written 2026-10-04, BEFORE either was run) — from the architecture review

Why these two: the sprint showed the single hand-built score has no edge and that survivorship bias contaminates every single-stock test. A1 attacks the first with 17x more labelled, point-in-time data and a learned (not hand-weighted) combination; A2 sidesteps the second entirely by testing on an index (no delisted constituents), and targets what a "best returns" competition usually scores (risk-adjusted return, drawdown).

**A1 — learned cross-sectional model on the EDGAR panel (Gu-Kelly-Xiu style, small).**
- Rows: ticker x quarter-end 2012-06..2025-06 (sealed holdout untouched). Features (all point-in-time, rank-normalised per date, missing -> median): mom_12_1, 1-month return, 252-day volatility, log market cap, earnings yield, FCF yield, book/market, ROA, gross profitability, accruals ((net income - CFO)/assets), asset growth, revenue growth, leverage, plus the production dcf_score, relative_score and upside_pct, plus sector code.
- Target: per-date percentile rank of next-quarter total return.
- Models (fixed now, no tuning): (i) HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200, l2_regularization=1.0); (ii) Ridge(alpha=10) as the linear baseline.
- Walk-forward: first test date 2016-06-30 (>= 16 prior quarters); retrain every 4 quarters on all dates whose label was fully realised before the retrain date; no peeking, no hyperparameter search.
- Pass line: mean out-of-sample rank IC >= 0.03 and t >= 2.5 (trial-adjusted), >= 100 names per date. Descriptive only: feature importance, comparison with the production composite's IC on the same dates, top-minus-bottom decile spread.
- Trials: 2 (counter -> 17).

**A2 — volatility-managed and trend-filtered S&P 500 exposure (Moreira-Muir; classic trend rule).**
- Data: ^GSPC daily 2007-01 onward, ^IRX for cash return. Price-return basis for both strategy and benchmark (no dividends, so the comparison is like-for-like).
- A2a vol-managed: at each month-end, exposure = min(1.0, 15% / realised annualised vol of the last 21 trading days); remainder in cash; no leverage. A2b trend: exposure 1.0 if close > 200-day average at month-end, else 0 (cash). Cost 5 bps per unit of exposure change. Benchmark: buy-and-hold ^GSPC.
- Pass line (must hold in BOTH 2007-2015 and 2016-2025 and after costs): Sharpe >= benchmark + 0.10 AND max drawdown at least 20% shallower (relative).
- Trials: 2 (counter -> 19).

### FIX-CANDIDATE RESULT (2026-10-04; `research/fix_candidates.py`; development only, sealed holdout untouched)

| Candidate | Pooled IC | t | Segment A (2012-18) | Segment B (2019-25) | 4q IC (desc.) | Pass? |
|---|---|---|---|---|---|---|
| Baseline composite | +0.011 | 0.62 | -0.004 | +0.026 | +0.024 | n/a |
| 1. DCF rank (no saturation) | +0.012 | 0.71 | -0.004 | +0.027 | +0.024 | **No** |
| 2a. composite + momentum | +0.006 | 0.47 | +0.002 | +0.011 | +0.014 | **No** |
| 2b. composite + ROA | +0.001 | 0.09 | -0.014 | +0.017 | +0.003 | **No** |
| 2c. composite + momentum + ROA | +0.002 | 0.12 | -0.005 | +0.008 | +0.001 | **No** |
| 3. regime gate (ON dates) | +0.012 | 0.52 | -0.020 | +0.026 | +0.023 | **No** |

- **Candidate 1:** removing saturation changes nothing (IC +0.0115 vs +0.0106). The DCF's full ordering carries no more information than its clipped score, so the "saturated extremes" explanation is rejected. A full DCF re-run with capped growth is not justified: the proxy says the ordering itself is uninformative.
- **Candidate 2:** adding momentum/quality makes it WORSE (consistent with P3, where multi_factor was -0.025). Fixing "value traps" with the factors we have does not work on this universe.
- **Candidate 3:** the gate is uninformative: the value spread trended up (0.05 in 2014 -> 0.095 in 2022-23), so the signal is ON for 37 of 41 testable dates; spread-vs-IC correlation +0.07; OFF-date IC -0.025 on only 4 dates. Not a falsification of regime timing in principle (a detrended spread would be a new trial), but this definition does not work.
- **Candidate 4:** needs no pass line; it is the only fix that does not depend on the data. Implemented separately (README / EVALUATION.md / report disclosure).
- **Verdict:** none of the three testable fixes works. Trials: 15.

### FIX-CANDIDATE PRE-REGISTRATION (written 2026-10-04, BEFORE any candidate was scored)

Goal: find out which of the four fixes would actually work before building anything. Data: `p2b_scores.csv` (EDGAR-fed production scores, 53 quarterly dates) + P3 factors. Outcome: next-quarter total return. Nothing is re-tuned: every definition below is fixed now; there are no parameters to fit.

- **Candidate 1 (robust DCF, cheap proxy):** `dcf_rank` = per-date percentile rank of raw `upside_pct` (removes the +/-100 saturation and tie block at the top, keeps full ordering). A true re-run of the DCF with capped growth / averaged margins is only worth the ~45-minute run if this proxy shows the DCF ordering carries information.
- **Candidate 2 (quality/trend filter):** 2a = mean of per-date percentile ranks of `composite_score` and `mom_12_1`; 2b = same with `roa`; 2c = mean of all three. Equal weights.
- **Candidate 3 (regime gate):** value spread on date T = mean (earnings_yield + fcf_yield) of the top quintile of `composite_score` minus that of the bottom quintile... see below. ON if the spread is above its expanding median of all PRIOR dates (needs >= 12 prior dates); score the unchanged composite on ON dates only.
  - Spread definition: median `earnings_yield` of the cheapest 20% of stocks (by `valuation_only`) minus median `earnings_yield` of the most expensive 20%.
- **Candidate 4 (change the claim):** not testable with data; it is a product-honesty decision.
- **Trials:** 1 + 3 + 1 = **5 new, counter -> 15.**
- **Evaluation:** Spearman IC vs next-quarter return per date, >= 100 names. Segment A = 2012-06..2018-12, Segment B = 2019-03..2025-06 (sealed holdout untouched).
- **Pass line (raised because 5 candidates are tested at once):** pooled mean IC >= 0.03 AND t >= 2.5 AND positive mean IC in BOTH segments. For Candidate 3: ON-date mean IC >= 0.03, t >= 2.5, positive in both segments, and at least 15 ON dates.
- **Baseline to beat:** composite IC +0.011 (t 0.62). Descriptive only (not eligible to pass): 4-quarter-horizon IC.
- **Rule:** if nothing passes, nothing is built into the scoring; the product change is Candidate 4 (disclosure), which needs no pass line.

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
