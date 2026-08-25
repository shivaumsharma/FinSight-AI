# Evaluation & Results

Every major component in FinSight has a dedicated evaluation script, not just an implementation. This document pulls those results together in one place — including the ones that didn't come out as hoped, because a system that only reports its wins isn't being evaluated honestly.

Numbers below are computed directly from artifacts already checked into `scripts/` and `app/valuation/` (backtest result JSON, the ML classifier's metrics file) or quoted from dated findings recorded in code comments where noted. Reproduction commands are in each section.

---

## 0. The canonical accuracy metric

FinSight reports exactly **one** headline accuracy number — not a table of window-by-window sub-metrics, and not the "Institutional Consensus Score" (analyst-agreement, market-context only, never a predictive-accuracy claim — see `app/reporting/consensus_score.py`). It's computed by `scripts/canonical_accuracy.py` and surfaced on every generated report (`report_data["track_record"]`, rendered in both the Streamlit and web UIs).

**Definition:** of every Buy/Hold/Sell call FinSight's real production decision path (`report_data_builder.derive_recommendation` — the exact function the deployed app calls) makes on the broad, non-cherry-picked 1,002-ticker S&P 500+400+600(partial) universe, what fraction are correct **12 months later** — Buy needs realized return `> +5%`, Sell needs `< -5%`, Hold needs to land between (the same rule Section 1 below validates, unchanged).

**Current number** (pooled across the two non-overlapping 12-month-horizon broad-universe backtests in Section 1 — 12mo→today and 24mo→12mo, N=485+438):

> **36.4% forward-accuracy (N=923, 95% CI 33.4–39.6%) vs. 60.0% Always-Buy baseline (95% CI 56.8–63.1%). The model currently loses to the naive baseline.**

**Sample size caveat, stated plainly:** this run landed at N=923, not the ~1,920 of the prior run, because Yahoo Finance rate-limited over half of both windows' requests (see "Known limitations" for the actual bug this surfaced -- a yfinance parsing crash on a throttled response, not a data problem). The pre-this-run data was already overwritten before that was caught, so this is **not** a clean matched-sample before/after against the prior 38.8%/57.0% figure -- the baseline itself moved 3 points from sample composition alone, which is enough on its own to explain the accuracy shift without attributing it to any code change. Treat 38.8% (prior, N=1,920) and 36.4% (current, N=923) as two independent snapshots, not a validated trend, until a clean full-sample re-run is done. What *is* cleanly measured — via a direct, unconfounded mechanism check rather than the noisy end-to-end backtest — is that the quality-tiered terminal growth fix (see "Known limitations") closed the mega-cap/deep-value valuation skew from ~1.9x to ~1.1x. Whether that translates to a real accuracy gain is the open question this number can't currently answer.

**Why pooled backtest data, not the live call tracker:** FinSight deploys to free-tier hosts with an ephemeral filesystem (see the Deployment section — Cloud Run/Railway both wipe `jobs.db` on redeploy unless a persistent volume is attached). A metric that depends on production calls surviving months of uptime isn't reproducible on this project's actual deploy target today, so it isn't the canonical one. The live tracker (`app/api/db.py`'s `tracked_call_checkpoints`, `window_days=365`) uses the identical scoring rule and is ready to be pooled in via the same `score_rating`/`naive_baseline_accuracy` functions the moment persistent storage exists — it's just not load-bearing yet.

**Scaling this up:** growing the sample and the diversity of market regimes tested (a bear-market window, later vintages, more tickers) means re-running `phase2_backtest.py` against new historical windows and re-running `canonical_accuracy.py` — not waiting on production traffic an ephemeral host can't retain. Real fixes to the underlying valuation logic (the mega-cap/deep-value skew, growth-rate fade) should show up as a re-run improvement in this same number, computed the same way, not a new metric.

**Reproduce:** `python scripts/canonical_accuracy.py` (writes `scripts/canonical_accuracy_result.json`, the artifact the report pipeline reads).

---

## 1. Recommendation engine — point-in-time backtest

**Method** (`scripts/phase2_backtest.py`): simulates running the pipeline as of N months ago and checks the realized return by today, with explicit no-look-ahead controls —
- a fiscal year only counts as "known" once `filing_date + 90 days <= as_of_date` (10-K filing lag),
- beta is computed from trailing price history ending at the as-of date, not today's yfinance beta,
- price/market cap use the as-of-date close, not the current price.

A rating is scored correct if `Buy` → realized return `> +5%`, `Sell` → `< -5%`, `Hold` → within that band.

Numbers below are from the most recent re-run (post-CAPM-fix AND post-quality-tiered-terminal-growth-fix, see "Known limitations"). The broad-universe rows have a smaller n than earlier in this document because of a Yahoo Finance rate-limiting episode during collection — see Section 0's sample-size caveat and "Known limitations" for the actual bug this surfaced; the curated rows are unaffected (a much smaller request volume):

| Window | Universe | n scored | Pipeline accuracy | Always-Buy baseline |
|---|---|---:|---:|---:|
| As-of 12mo ago → today | hand-curated tickers (mega-cap, deep-value, hypergrowth/neg-FCF, financials — the curated set has grown since this table was first written, no longer exactly 79) | 98 | **48.0%** | 57.1% |
| As-of 24mo ago → 12mo ago | same curated set | 95 | **32.6%** | 71.6% |
| As-of 12mo ago → today | 1,002-ticker broad universe (rate-limited collection, n reduced) | 485 | **35.1%** | 61.2% |
| As-of 24mo ago → 12mo ago | 1,002-ticker broad universe (rate-limited collection, n reduced) | 438 | **37.9%** | 58.7% |

### The honest finding

In every one of the four windows tested, a naive **"always predict Buy"** baseline beats the pipeline's raw direction-accuracy. This isn't a bug — all four windows sit inside a broadly rising market (universe average realized return ranged from +10.6% to +35.6%), so "the market went up, therefore call everything Buy" is a genuinely strong baseline in-sample, and a flat accuracy percentage doesn't isolate whether the *valuation logic itself* is adding signal.

Decomposed by what the model actually called, a clearer (and more useful) picture shows up:

| Window | Buy precision | Buy avg. realized return | Sell precision | Sell avg. realized return |
|---|---:|---:|---:|---:|
| 12mo curated | 64.6% (n=48) | +21.7% | 34.1% (n=44) | +2.8% |
| 24mo curated | 63.2% (n=38) | +24.6% | 15.6% (n=45) | +44.5% |
| 12mo broad | 54.4% (n=217) | +20.3% | 21.1% (n=199) | +34.5% |
| 24mo broad | 57.8% (n=187) | +13.6% | 25.4% (n=185) | +17.2% |

**Buy calls carry real signal** — 54–65% precision, consistently above each window's universe-average return. **Sell calls do not work in this sample** — precision as low as 15.6%, and in every window the average stock the model called "Sell" on still had a strongly *positive* realized return. `Hold` is statistically unreadable (small n per window).

The likely explanation, not yet confirmed: the ticker universe deliberately includes hypergrowth/negative-FCF names (PLTR, RIVN, SNOW, etc. — see `phase2_backtest.py`'s `TICKERS` dict) that a DCF structurally flags as overvalued on current fundamentals, and all four windows fall inside a period where exactly those names kept re-rating upward anyway.

### "Don't sell into strength" (momentum-gated Sell calls) — tested, doesn't hold up

Direct hypothesis from the semiconductor case study above: the model's worst Sell calls had strong price momentum the composite score never saw. `scripts/tune_momentum_weight.py` tests gating Sell calls with 6-month price momentum — asymmetric by design (only ever pulls the score up, away from Sell, never down, since Buy precision doesn't need help) — against the Phase 2 data (n=1,699 across 4 windows), percentile-normalized the same way DCF/relative scores already are.

**First look, and why it's a trap:** pooled accuracy rises monotonically with the momentum weight (44.1% at weight=0 → 46.8% at weight=0.5), which looks like a clean win. But Sell precision barely moves (27.5% → 27.9%) even as Sell *volume* nearly halves (596 → 308 calls) — if momentum were genuinely telling good Sells from bad ones, precision on the surviving calls should rise substantially, not sit flat. That combination (accuracy up, precision on the affected class flat, volume of that class down) is the signature of a base-rate effect, not a real one: in a sample where the market went up most of the time, calling Sell *less often* mechanically raises blended accuracy regardless of which specific calls get cut.

**Confirmed directly, not just suspected:** compared the momentum-filtered 308 Sells (27.9% precision) against simply taking the 308 *most confident* Sells by composite score alone, no momentum involved (**29.2% precision — higher**). Momentum-based selection is worse than a naive same-size cutoff. It isn't identifying which Sells are wrong; it's cutting volume in a way roughly uncorrelated with correctness, and the accuracy metric alone couldn't tell the difference until this matched-size comparison was run.

**Not shipped.** Same discipline as the ML-classifier composite test (section 4) — this is the second time this session a change looked good on the headline metric until checked against a same-size random-cutoff baseline, and didn't survive it. The matched-size-baseline check itself is worth keeping as a standard test for any future "add a filter/gate" idea, not just this one.

**Reproduce:** `python scripts/tune_momentum_weight.py`.

### A dividend discount model cross-check — built, tested, shipped display-only (real signal, not enough data to trust yet)

A second, genuinely independent valuation lens: `app/valuation/ddm_engine.py`, a Gordon Growth DDM that values the actual dividend stream (not modeled FCFF), scoped to consistent, *material* dividend payers only. Required adding dividend data to the normalizer (`app/data/financial_normalizer.py`/`metric_mappings.py` — "Cash Dividends Paid" was already in yfinance's cashflow statement, just never extracted).

**Two real distortions caught and fixed before this was trustworthy enough to test:**
- **Token dividends inflate nonsense.** NVDA technically "pays a dividend" every year (clears a naive years-paying check) at under 10% of net income — a symbolic gesture, not a capital-return policy. Without a materiality filter, DDM priced NVDA at $1.81 against a ~$209 actual price. Fixed with a payout-ratio floor (`MIN_PAYOUT_RATIO = 0.15`) — NVDA, AMZN, AAPL (payout ~14%, dominated by buybacks) now correctly return `None` rather than a distorted number.
- **A raw multi-year dividend CAGR isn't a "forever" growth rate.** MSFT's real 3-year DPS CAGR (10.1%) fed directly into Gordon Growth compounds it to infinity — no company outgrows the economy forever at that clip, the same reason the FCFF-DCF fades to a modest terminal rate instead of extrapolating raw CAGR (`fcff_engine.py`'s 3-stage fade). DDM has no multi-stage forecast to fade across, so the growth input itself is capped instead (`MAX_SUSTAINABLE_DPS_GROWTH = 0.06`, set just above the DCF's own highest quality-tier terminal rate of 5%).

**Tested for composite inclusion** (`scripts/tune_ddm_weight.py`, curated-universe backtest re-run with `ddm_value` now captured per row, n=71 dividend-payer observations across 2 windows) — structurally a cleaner test than the momentum one, since blending DDM in never changes *which* tickers get scored (only their score), so the volume-reduction trap above doesn't apply here. Result: accuracy on the DDM-available subset rises from 49.3% (today's DCF/relative-only config) to a peak of 52.1% at a modest blend weight, then falls off at higher weights — a real interior peak, not a monotonic edge-of-range trend (the shape that made the momentum result suspicious). But n=71 is too small to trust a 2-3 point swing (well within noise at that sample size), and the script's own plateau check flags it as a narrow spike, not a robust signal.

**Shipped display-only** (`report_data["valuation_analysis"]["dividend_discount_model"]`, both Streamlit and web UI, clearly labeled "not part of the recommendation") — same "prove it, then wire it in" path the ML classifier took (Section 4), not yet blended into the composite. Unlike the momentum result, this isn't a dead end — it's promising and underpowered, worth revisiting once enough point-in-time dividend-payer data exists (the same kind of Phase-2-style broad-universe, multi-window sweep that grew the ML classifier's training set 35x would do the same here).

**Reproduce:** `python scripts/tune_ddm_weight.py` (needs `backtest_results_curated_asof{12,24}mo*.json` regenerated with `ddm_value` captured — rerun `phase2_backtest.py` first if those predate this feature).

### Testing a bear-market window — attempted, structurally blocked

The natural next step was rerunning against a window that includes a real drawdown (e.g. 2022) to see whether Sell precision recovers outside a one-directional bull market. **This is no longer possible with this methodology, and it's worth understanding why rather than just noting the attempt failed.**

`phase2_backtest.py 56 44` (as-of ≈ 2022-01-15, exit ≈ 2023-01-06 — squarely the 2022 bear market, S&P peak-to-trough) scores **0 of 100 tickers**. yfinance's annual-financials endpoint only exposes a *rolling* trailing window (confirmed live: AAPL currently returns exactly FY2021-FY2025, five years back from today, nothing older) — not a fixed historical archive. As real time advances, fiscal years fall off the far end of what yfinance returns at all, regardless of how the as-of date is chosen. 2022's own fiscal year data has now aged out entirely: every ticker fails with "no fiscal year was filed early enough to be known as of the as-of date" or "insufficient point-in-time financial history." Price history has much longer retention (10y+ fetches work fine) — only the fundamentals are the constraint, but the DCF backtest needs both.

Checked whether a less-extreme drawdown exists anywhere still inside the reachable window (monthly S&P closes, live fetch): the market has been almost uninterruptedly bullish since 2023. The mildest available dip is Dec 2024 (peak 6047) → Apr 2025 (trough 5633, -6.8%) — nowhere near a real correction, and by the 12-month mark the market had still risen further net. Ran that window anyway (`phase2_backtest.py 20 8`, curated universe) as the closest available proxy:

| | n scored | Base rate (Up/Down/Flat) | Model accuracy | Always-Buy | Sell precision | Return spread |
|---|---:|---|---:|---:|---:|---:|
| As-of 2024-12-30 → 2025-12-25 (least-bullish reachable window) | 98 | 67.3% / 22.4% / 10.2% | **40.8%** | 67.3% | 22.9% | **-5.31** (Sell calls outperformed Buy calls) |

Still a 67%-up-rate bull market (not a bear market by any real definition), and the model-vs-baseline gap here (-26.5 points) is the **worst of any window tested so far**, not better — and the return spread is negative here too (yet another sign flip). **Conclusion: no evidence that a less-bullish regime alone would fix Sell precision or the ranking instability; the hypothesis is untested at genuine bear-market severity because that data no longer exists in yfinance's reachable window, and the one weaker-bull proxy available doesn't support it either.** This line of inquiry is closed for now, not because it doesn't matter, but because the data to properly test it isn't obtainable through this pipeline's current data source.

**Reproduce:** `python scripts/phase2_backtest.py 20 8` (least-bullish reachable window); `python scripts/phase2_backtest.py 56 44` (reproduces the 2022-window failure, confirming the data-retention constraint above).

### Cross-sectional ranking (Phase 6) — the sign-flipping return spread, resolved to one specific cause

The return-spread metric above (Buy avg return minus Sell avg return) flips sign across windows — positive in one, negative in four others, including the least-bullish window. That metric only uses tickers that crossed the +/-7.5 Buy/Sell threshold and collapses a continuous composite score into 3 buckets. `scripts/cross_sectional_ranking.py` asks the more standard quant-factor question instead, using data already on disk (`composite_score`/`realized_return_pct`, no new backtest run): **Spearman rank correlation between composite_score and realized return, across the whole scored universe, independent of any threshold.**

| Window | n | Spearman ρ | p-value |
|---|---:|---:|---:|
| 12mo curated | 49 | +0.323 | 0.024 |
| 24mo curated | 43 | +0.230 | 0.138 |
| 20mo curated (least-bullish window) | 46 | +0.218 | 0.145 |
| 12mo broad | 584 | **-0.099** | **0.017** |
| 24mo broad | 534 | +0.119 | 0.006 |

Mixed signs even among the *statistically significant* results (12mo broad negative at p=0.017; 24mo broad positive at p=0.006) — confirming, more rigorously than the return-spread metric could, that there's no stable cross-sectional ranking skill. But the largest-sample result (12mo broad, n=584) being both significant AND negative was worth understanding rather than shrugging off as noise.

**Traced to one cause: Information Technology, not a general failure.** Splitting that window by sector:

| | n | Avg realized return | Spearman ρ | p-value |
|---|---:|---:|---:|---:|
| Information Technology | 84 | **+43.5%** | **-0.321** | **0.003** |
| Everything else | 500 | +14.4% | -0.034 | 0.448 (no signal, not negative) |

Outside IT, the composite score has *no* measurable ranking power in either direction (ρ≈0, not significant) — consistent with everything else this section has found. Inside IT specifically, in this one window, the ranking is actively backwards, strongly and significantly: the model's most bearish reads were disproportionately its best performers. The most bearishly-scored IT names in this window (STX, VIAV, DELL, TER, AMAT, LRCX, AMD, KLAC — semiconductor and semiconductor-equipment makers, composite scores as low as -83, all rated Sell) went on to realize **+92% to +419%** returns — an AI/semiconductor infrastructure demand supercycle a trailing-fundamentals DCF has no mechanism to see coming.

**Checked whether this is a persistent blind spot worth building a targeted fix for. It isn't — the pattern doesn't survive a second window:**

| Window | IT n | IT avg return | IT Spearman ρ | p-value |
|---|---:|---:|---:|---:|
| 12mo broad (most recent year) | 84 | +43.5% | **-0.321** | 0.003 |
| 24mo broad (the year before) | 71 | +9.8% | **+0.217** | 0.069 |

The correlation flips sign between the two windows, and the average return the whole sector realized is wildly different (+43.5% vs +9.8%) — this isn't "the model mishandles semiconductors," it's "the most recent 12 months happened to contain an unprecedented, unforecastable AI infrastructure demand shock," which is a description of one exceptional historical event, not a correctable methodology flaw. A rule that dampens or excludes Sell calls on semiconductor names based on this finding would be overfitting to a single historical quirk — exactly the class of mistake this project has rejected everywhere else (the narrow-spike-rejected composite-weight configs, the reranker, the embedding fine-tune). No backward-looking valuation method, DCF or otherwise, could reasonably have priced this in a year ahead of time. **No fix is proposed here; the finding is documented as a limitation, not chased into a change that wouldn't generalize.**

**Reproduce:** `python scripts/cross_sectional_ranking.py` for the pooled/per-window Spearman results; the IT-vs-rest and window-comparison splits above were one-off follow-up queries against the two `backtest_results_ticker_universe_*.json` files, not yet a standalone script.

### The disagreement-guardrail finding (recomputed, not just quoted)

`report_data_builder.py` used to force a `Hold` whenever the DCF's own directional call disagreed with the relative-valuation signal. Recomputing both counterfactuals directly from the saved backtest rows (`dcf_only_rating` vs. the guardrail's forced outcome), on the exact subset where the two signals disagreed:

| Window | n (disagreement subset) | Composite (current, no forced Hold) | Trusting DCF's own call alone | Forced Hold (old guardrail) |
|---|---:|---:|---:|---:|
| 12mo curated | 25 | 64.0% | 68.0% | 16.0% |
| 24mo curated | 15 | 53.3% | 53.3% | 13.3% |
| 12mo broad | 280 | 45.0% | 48.9% | 16.1% |
| 24mo broad | 275 | 43.3% | 47.3% | 15.3% |

Trusting DCF's own directional call outperforms the composite blend on every window's disagreement subset — directionally consistent with the finding recorded in `report_data_builder.py`'s comments (which cites 15.4% vs. 53.8% on a 13-ticker subset from an earlier snapshot; exact figures drift run to run since `realized_return_pct` is computed against "today," a moving target, but the qualitative result reproduces). This is why the forced-Hold override was removed in favor of a confidence-flag annotation instead.

**Reproduce:** `python scripts/phase2_backtest.py 12 0` (or `24 12`, or add `--universe scripts/ticker_universe.json` for the broad run).

---

## 2. RAG retrieval evaluation

**Method** (`scripts/evaluate_retrieval.py`): runs the real production query path against 7 hand-labeled `(ticker, question)` pairs across 5 companies (`app/evaluation/retrieval_labels.py`), scoring raw embedding retrieval (top-20) and reranked output (top-5) separately against Precision@5, Recall@pool, NDCG@5, and MRR.

**Finding, originally recorded in `app/tools/rag_tool.py`'s module docstring at the time the reranker was disabled, reproduced live for this document (2026-07-26):**

| Metric | Raw retrieval (production) | Reranked (cross-encoder) |
|---|---:|---:|
| Precision@5 | **0.552** | 0.438 |
| Recall@pool | 1.000 | 0.547 |
| NDCG@5 | 0.752 | 0.689 |
| MRR | **0.786** | 0.512 |

The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) makes retrieval *worse* on every metric, live and reproducible, not just a one-time historical result. A separate live A/B recorded in the code at the time this was found: citation-grounding score for a generated NFLX report went from 0.0 (reranker on) to 60.0 (reranker off) — with the reranker active, the report wasn't grounded in any of the evidence it was given. A cross-encoder trained on general web-search relevance (MS MARCO) apparently doesn't transfer to dense SEC filing text. `rag_tool.py` uses raw embedding retrieval directly as a result; the reranker code remains in `app/rag/reranker.py` as a documented, evaluated, and rejected approach rather than being silently deleted.

**Reproduce:** `python scripts/evaluate_retrieval.py`

---

## 3. Embedding fine-tune

**Method** (`scripts/finetune_embeddings.py` + `scripts/evaluate_embedding_finetune.py`): fine-tunes the production embedding model (`BAAI/bge-base-en-v1.5`) with `MultipleNegativesRankingLoss` on 194 synthetic `(query, chunk)` pairs generated from ingested filings, 4 epochs. The evaluation is methodologically careful about isolating the actual variable: the baseline model's own top-20 retrieval defines a *frozen candidate pool* (the same pool the hand labels in `retrieval_labels.py` were written against), and both the baseline and fine-tuned models only **re-rank that same frozen pool** — so a difference in score can only come from the embedding model itself, not from a different model surfacing a different set of candidates from the wider corpus.

**Result, run live for this document (2026-07-26):**

| Metric | Baseline BGE | Fine-tuned BGE |
|---|---:|---:|
| Precision@5 | **0.552** | 0.467 |
| Recall@pool | 1.000 | 1.000 |
| NDCG@5 | **0.752** | 0.517 |
| MRR | **0.786** | 0.417 |

The fine-tune made retrieval worse on every metric — a second, independent instance of the same pattern as the reranker above. The likely cause: 194 pairs over 4 epochs on a base model this size is a small, easy-to-overfit training run, and `MultipleNegativesRankingLoss`'s in-batch-negatives approach needs enough batch diversity to avoid the model collapsing toward trivial shortcuts rather than genuine semantic separation. This wasn't a wasted exercise, though — it's exactly why the fine-tuned model was never wired into `app/rag/chroma_store.py` in the first place (which still points at the untouched `BAAI/bge-base-en-v1.5`); the eval script existed specifically so that decision would be evidence-based rather than assumed. Next step to actually improve on baseline: a meaningfully larger training set and/or explicit hard-negative mining instead of relying on in-batch negatives alone.

**Reproduce:** `python scripts/finetune_embeddings.py && python scripts/evaluate_embedding_finetune.py`

---

## 4. ML valuation classifier

**Method** (`app/valuation/ml_valuation_classifier.py`): Logistic Regression vs. XGBoost (falls back to GradientBoosting without xgboost installed), compared via stratified 5-fold CV *and* a held-out test split, with per-class precision/recall/F1 (not one blended accuracy number) and a confusion matrix. Labels are realized forward-return outcomes from FinSight's own point-in-time backtest (`scripts/build_ml_training_set.py`), not analyst agreement — avoiding the look-ahead bias found in a separate project this design was adapted from (see the model's own docstring).

**Two prerequisite fixes, before any feature work could be trusted:**
- `scripts/build_ml_training_set.py` was calling `phase2_backtest.run_one()` without the `risk_free_rate_override` argument `phase2_backtest.py`'s own `main()` now passes — every training row was silently valuing a company "as of 12 months ago" using *today's* live Treasury yield. Fixed by fetching and passing `tnx_history`, same as the main backtest.
- `AlphaFactorsEngine`'s momentum/relative-strength/sector/rate-sensitivity factors compare a (correctly point-in-time) stock price series against benchmark/sector/rate histories that `get_benchmark_history()` otherwise always fetches through *today* — a second, previously-undiscovered look-ahead leak. Fixed with a `point_in_time_cutoff` on the research context that `ValuationTool.run()` uses to truncate those histories before handing them to `AlphaFactorsEngine`.

**Feature expansion:** the classifier's 11 features were previously 100% valuation-derived (growth, WACC, beta, FCF yield, DCF/price, Monte Carlo stats, relative valuation) despite `AlphaFactorsEngine` already computing momentum, quality (Piotroski/Altman), volatility, sentiment, and macro-sensitivity factors for display only. Added `momentum_6m`, `momentum_12m`, `annualized_volatility` — chosen specifically because they depend only on the company's own price series, unlike relative-strength/sector-performance/rate-sensitivity, which are structurally `None` for every non-US (.NS) ticker (confirmed by a real test failure while adding them) and would have silently zeroed out the classifier for that entire population. Piotroski F-Score, Altman Z-Score, and sentiment are deferred for the same reason (real missingness that would shrink an already-tiny training set under this module's strict all-required-features policy) — see `ml_features.py`'s own comment.

**Phase 2 — growing the training set 35x.** `build_ml_training_set.py` previously accepted no arguments: one hardcoded as-of date (12 months ago), the curated ~100-ticker universe only, and a sequential (non-threaded) loop. Generalized to take the same `as_of_months_ago` / `--universe` / `--workers` arguments `phase2_backtest.py` already had, then run across the full 1,002-ticker broad universe over 6 non-overlapping historical windows (12/18/24/30/36/48 months ago). Yahoo Finance rate-limiting degraded hard over the sweep (582→546→516→55→1→0 usable rows per window, worse each time) — 4 of 6 windows still landed cleanly, combining to **1,700 rows** (`combine_ml_training_sets.py`, de-duplicated on ticker+as-of-date). A genuine bug surfaced by the failure: a fully rate-limited window wrote a 0-byte CSV that crashed the combine step (`pandas.errors.EmptyDataError`) — fixed on both ends (a valid header-only CSV even at zero rows; the combiner skips an unreadable file instead of crashing).

**The trap this caught, and why it's the headline finding of this section:** naively retraining on the larger set looked like a big win — accuracy rose to 61% — until the per-class breakdown showed why: the classifier had learned to predict `UNDERVALUED` almost every time (100% recall on it, ~4% recall on the other two classes), trivially matching accuracy since 59.6% of the 1,700-row label distribution *is* `UNDERVALUED` (the broad universe sampled several bull-market years — the same base-rate phenomenon Section 1's Always-Buy baseline already documents, now showing up inside the classifier instead of in the naive baseline). F1-macro — the metric that actually penalizes ignoring minority classes — *dropped* (0.458 → 0.277) even as raw accuracy rose. Caught before this got anywhere near the recommendation composite.

**Fix:** class-balanced training. `LogisticRegression` gets `class_weight="balanced"` (constructor-time, no extra plumbing). The secondary model (XGBoost/GradientBoosting) has no multiclass `class_weight` equivalent, so it gets an explicit `sample_weight` (`sklearn.utils.class_weight.compute_sample_weight`) at fit time — which meant replacing `sklearn.model_selection.cross_validate()`'s convenience wrapper with a manual `StratifiedKFold` loop, since threading `sample_weight` through `cross_validate` requires its newer metadata-routing API (confirmed this installed sklearn version no longer even accepts the older `fit_params=` kwarg).

**Full before/after** (`app/valuation/ml_classifier_metrics.json`):

| | N | Best model | Held-out F1 (macro) | UNDERVALUED / FAIRLY VALUED / OVERVALUED recall |
|---|---:|---|---:|---|
| Original (11 valuation-only features) | 39 | XGBoost | 0.410 | — |
| + momentum/volatility features | 49 | Logistic Regression | 0.458 | — |
| + Phase 2 data, no class balancing | 1,700 | XGBoost | 0.346 *(looks better, isn't)* | 95.3% / 4.2% / 13.0% |
| + Phase 2 data, class-balanced | **1,700** | **XGBoost** | **0.420** | **56.7% / 43.8% / 34.1%** |

The class-balanced result is the honest one: genuinely balanced discrimination across all three classes (not a majority-class default dressed up as accuracy), on 35x the data, with a stable CV F1-macro (0.415 ± 0.040 across 5 folds — tight relative to the ±0.15-0.18 stds at N=39-49) confirming it isn't a lucky split. `annualized_volatility` remains the single most important feature, ahead of every valuation-derived one.

**Phase 4 — does wiring it into the composite actually move accuracy? Tested, not assumed. Result: no.** `scripts/tune_ml_weight.py` grid-searches an `ML_WEIGHT` (a third term alongside `DCF_WEIGHT`/`RELATIVE_WEIGHT`, `ml_score = (P(UNDERVALUED) - P(OVERVALUED)) * 100`, naturally bounded so it needs no percentile normalization) against the Phase 2 windows.

The first version of this test showed a large, monotonically-increasing benefit all the way to `ML_WEIGHT=1.0` — a classic in-sample-leakage signature: it was scoring the saved model against the same 1,700 rows that model was trained on, silently rewarding memorization. Caught, and fixed with leave-one-window-out evaluation — each of the 4 windows scored by a classifier retrained on the *other* 3 only, genuinely out-of-sample for the window being judged. Same class of mistake as the class-imbalance trap above, different flavor, same lesson: an ML result that looks dramatically good deserves more suspicion, not less.

**The honest, out-of-sample result:** pooled accuracy across all 4 windows barely moves — 44.1% at `ML_WEIGHT=0` (today's config) vs. 44.8% at the best-scoring weight (0.3) — and *every* tested weight from 0.0 to 0.5 lands within 1 point of every other. A flat plateau, not a real effect in either direction.

**Conclusion: the ML classifier is not wired into the composite.** `DCF_WEIGHT`/`RELATIVE_WEIGHT` are unchanged. This is exactly what "only wire it in if it clears a real bar" was supposed to prevent shipping — a plausible-sounding feature addition that doesn't survive honest out-of-sample measurement. The classifier stays display-only. Kept as a reproducible negative result (`tune_ml_weight.py` isn't deleted) rather than silently dropped, so this doesn't get re-litigated from scratch once more Phase 2 data exists.

**Reproduce (data):** `for m in 12 18 24 30 36 48; do python scripts/build_ml_training_set.py $m --universe scripts/ticker_universe.json; done && python scripts/combine_ml_training_sets.py && python -c "from app.valuation.ml_valuation_classifier import train; train('scripts/ml_training_set.csv')"` (space the windows out, or lower `--workers`, if re-running — this sweep's later windows got progressively rate-limited running back-to-back). **Reproduce (composite test):** `python scripts/tune_ml_weight.py`.

---

## 5. Per-report self-evaluation

Every generated report is scored at runtime by `app/evaluation/evaluation_engine.py` (grounding 40%, retrieval 20%, citation coverage 20%, completeness 20% — see `app/evaluation/scorer.py`). Grounding and citation checks went through a documented v1→v2 rewrite (`app/evaluation/grounding_validator.py`, `citation_evaluator.py`): v1 required a *verbatim substring match* against the source evidence, but the report prompt explicitly instructs the model to paraphrase — so a correctly-written, fully-grounded report would almost never satisfy v1's check. v2 uses stemmed content-word overlap instead, which actually rewards grounded paraphrase rather than penalizing it. `app/benchmarks/*.json` + `app/evaluation/benchmark_runner.py` additionally check generated reports for 5 fixed companies against expected sentiment/recommendation/topic coverage, so prompt or retrieval changes can be compared against a fixed baseline instead of eyeballed.

---

## 6. Inference latency

Switching the local LLM backend from a raw Hugging Face `transformers` pipeline to `llama.cpp` (Q8_0-quantized GGUF) cut the narrative-generation call (~3,300 prompt tokens, up to 700 generated) from **~257s to ~65s** on the same machine (`app/rag/report_generator.py`). Q8_0 was kept over the ~30%-faster Q4_K_M after a direct A/B: Q4_K_M produced a 2,200-character Executive Summary that consumed the entire generation budget and silently dropped the other four report sections on a real MSFT prompt — confirmed on the actual failure, not assumed.

---

## 7. LangGraph orchestration vs. the hand-rolled controller

**Method** (`scripts/benchmark_orchestration.py`): `app/agents/langgraph_agent.py` is a LangGraph port of `ResearchAgent`'s plan-execution loop, kept alongside it as a documented alternative rather than a replacement. The benchmark runs the same questions through both and checks two things: does the graph visit the exact same tools in the exact same order (`tool_trace` equality), and how does control-flow latency actually compare, isolated from the real tool work neither orchestrator controls.

**First attempt, and why it was thrown out:** an initial run against real AAPL/MSFT queries (`--lite` mode — real tools, LLM narrative call stubbed) produced a 964s outlier for one `ResearchAgent` run against a 32s `LangGraphResearchAgent` run on the *same* ticker — not a real 30x orchestration difference, but a stalled SEC/yfinance call on that particular run (a transient "possibly delisted, no price data found" warning showed up in the same run). Real network I/O variance is large enough to make a small-sample latency comparison meaningless noise, not signal — reported here instead of the flashier-looking but wrong number.

**`--pure` mode** (all 9 tools stubbed to no-ops on both sides, 20 reps/ticker, median reported) isolates what's actually being asked — StateGraph/Pregel dispatch overhead vs. a plain Python `for` loop, with zero network/model calls in either path:

| Orchestrator | Median latency (control-flow only) |
|---|---:|
| `ResearchAgent` (hand-rolled loop) | ~0.00ms |
| `LangGraphResearchAgent` | ~7–13ms |

`tool_trace` was identical across every real and stubbed run — the two orchestrators agree on which tools to run, in which order, every time. LangGraph adds a small, real, consistently-measurable dispatch overhead (single-digit-to-low-double-digit milliseconds) versus a bare loop — completely negligible against the tens of seconds to minutes a real tool call (network fetch, LLM inference) actually takes. The honest conclusion: at this system's scale, the choice between the two is a maintainability/observability decision (state inspection, checkpointing, a visualizable graph), not a performance one.

**Reproduce:** `python scripts/benchmark_orchestration.py` (default `--pure`; `--lite`/`--full` run real tools and are noisier by design, see the script's own docstring).

---

## 8. Redis caching layer

**Method** (`app/core/cache.py` + `scripts/benchmark_redis_cache.py`): two different caching strategies, deliberately not collapsed into one. Content-addressed caching (the narrative LLM call in `narrative_builder.py`, and `ValuationPipeline.run_valuation()`'s statement-derived output) keys on a hash of every input that actually determines the output, so a cache hit can never serve a result computed from different inputs — TTL there is a storage bound, not the correctness mechanism. TTL-only caching (`MarketDataLoader`'s income/balance/cash-flow statement fetches) is scoped specifically to data that only changes quarterly; `current_price`/`market_cap` are never cached this way, since serving a stale price on a live research tool would be a correctness bug, not a staleness inconvenience.

No real Redis server runs anywhere this project deploys by default (local dev, HF Spaces) — every cache call degrades to a silent no-op if unreachable, and the benchmark below uses `fakeredis` (a standard package implementing the real redis-py wire protocol in-process) so it's exercising the actual `cache_get`/`cache_set` code path, not a mock.

| Operation | Cache miss | Cache hit | Speedup |
|---|---:|---:|---:|
| `ValuationPipeline.run_valuation()` (AAPL — WACC, FCFF forecast, 5x5 sensitivity grid, 2,000-sample Monte Carlo) | 5,278ms | 3.0ms | **1,738x** |
| `MarketDataLoader` statement fetch (MSFT — 3 sequential yfinance calls) | 1,149ms | 2.0ms | **567x** |

Both numbers are from independent tickers specifically to avoid a methodology bug caught during this benchmark's own development: an earlier version reused the same ticker across both sections, so the "miss" for the statement fetch was actually riding on a cache the valuation section had already warmed — a real, if small, reminder that a benchmark's own fairness needs the same scrutiny as the thing it's measuring. Neither of these is the dominant cost in a full report (the ~65s LLM narrative call is, and it's now cached too) — but both are genuine, correctly-isolated speedups on their own terms.

**Reproduce:** `python scripts/benchmark_redis_cache.py`

---

## 9. Known limitations

Documented here rather than left implicit, in the same spirit as the rest of this file.

**DCF still undervalues mega-caps relative to deep-value names, though less than previously measured.** The 0.59 (mega-cap) vs. 1.77 (deep-value) split this section used to cite was stale — `app/valuation/fcff_engine.py`'s `forecast_fcff()` already implements a three-stage, ROE-tiered growth fade (a durable, high-ROE compounder holds its own growth rate for up to 5 years before fading toward terminal growth, instead of every company decaying on the same fixed clock), committed before that number was last measured. Re-running `scripts/wacc_capm_audit.py` today: mega-cap median intrinsic-value/price is **~0.91**, not 0.59 — the fade fix already closed roughly half the original gap. Deep-value median is still **~1.75**, essentially unchanged, so the *categorical* skew persists even though the mega-cap number individually improved. NVDA (10.6x) is a genuine outlier pulling the category's mean, not its median.

**Quality-tiered terminal growth, targeting the remaining mega-cap/deep-value gap directly — and it worked, cleanly measured.** The 3-stage fade above closed half the original gap by giving high-ROE companies more explicit-forecast years at their own growth rate before fading — but every company still faded to the SAME flat terminal rate (4%) regardless of quality, understating a durable compounder's perpetuity value relative to a mature business's. Extended the same ROE tier already driving the fade duration to also shift where it lands: `FCFFEngine.quality_terminal_growth_adjustment()` adds +1.0 point for ROE >=25% (moat-tier), 0 for the default tier, -1.0 point for ROE <15% -- within the +-1%-5% band this codebase's own sensitivity table already treats as reasonable, picked from first principles rather than fitted to this session's numbers (a real grid search here would need a full DCF re-run per candidate, unlike the cheap composite-weight tuning). Applied consistently to `forecast_fcff`'s fade math AND `DCFEngine`'s/`MonteCarloDCFEngine`'s own terminal-value calculation (computed once in `ValuationPipeline`, not independently in each place -- a mismatch there would be a real discontinuity bug).

Measured directly through the real pipeline (not the stale `wacc_capm_audit.py`, which still bypasses it -- see the CAPM paragraph below), a live check against 18 mega-cap and 12 deep-value tickers:

| | Mega-cap median IV/price | Deep-value median IV/price | Gap |
|---|---:|---:|---:|
| Before any fix | 0.59 | 1.77 | ~3x |
| After 3-stage growth fade only | 0.91 | 1.75 | ~1.9x |
| **After quality-tiered terminal growth too** | **1.18** | **1.30** | **~1.1x** |

The gap this whole investigation was chasing is now nearly closed -- mega-caps moved from systematically undervalued to essentially fair (median now *above* 1.0), and deep-value moved down toward fair too. Confirmed live: NVDA/MSFT (high-ROE) now get 5% terminal growth, XOM (deep-value energy) gets 3%, MO (default tier) stays at 4%.

**CAPM inputs are now sourced, not hardcoded guesses** — as of this update: risk-free rate is a **live 10-year Treasury yield (`^TNX`)**, fetched fresh per valuation (`app/valuation/valuation_pipeline.py`'s `_live_us_risk_free_rate()`, falling back to a documented 4% constant only if that fetch fails); equity risk premium is **4.45%**, Aswath Damodaran's published US implied ERP as of his July 2026 data update, replacing the old undocumented 6% guess. Lowering ERP disproportionately raises intrinsic value for higher-beta names (ERP is multiplied by beta in CAPM) — exactly the mega-cap/high-growth category the skew above affects — so this is a real, targeted change, not just a citation. **Caveat:** `scripts/wacc_capm_audit.py` itself still calls `WACCEngine` directly with its own hardcoded 4%/6% defaults rather than going through `ValuationPipeline` (the real pipeline every live report and `phase2_backtest.py` actually use) — its own printed CAPM banner is therefore now stale relative to the deployed pipeline, and needs a follow-up fix to route through `ValuationPipeline` before its intrinsic-value numbers reflect the ERP change too. The mega-cap/deep-value figures above already reflect the growth-fade fix (which lives in `FCFFEngine`, which the audit script does call directly) but **not yet** the ERP fix.

**Point-in-time discipline extended to the risk-free rate.** Since the rate is now live-fetched, `scripts/phase2_backtest.py` would otherwise leak today's Treasury yield into a historical "as of 12 months ago" valuation — a new look-ahead risk the fix itself introduced. Fixed by passing a `risk_free_rate_override` (the `^TNX` close nearest the as-of date) through `ValuationTool` → `ValuationPipeline`, same point-in-time discipline already applied to price and beta.

**No predictive skill is currently demonstrated, as of the last full re-run.** See [Section 0](#0-the-canonical-accuracy-metric) for the current pooled number and baseline. The full broad-universe backtest has been re-run with the CAPM fix above: pooled accuracy moved **36.9% → 38.8%** (+1.9pts) against a baseline that itself moved 56.4% → 57.0% (+0.6pts, expected drift as "today" advances) — a real, measured, modest improvement, not a dramatic one, and the model still loses to the naive baseline. Measured, not assumed: this is the actual before/after, not a projection.

**Whether the terminal-growth fix above also moved accuracy: not cleanly measured, and said so rather than reporting a misleading number.** Re-running the full broad-universe backtest after this fix hit severe Yahoo Finance rate limiting (a real, previously-latent yfinance bug surfaced under load: a throttled response comes back malformed, and yfinance's own parser crashes on it with `'NoneType' object is not subscriptable` rather than a clean retriable error) -- over half of both windows errored out even at reduced concurrency (3 workers), and the pre-fix raw per-ticker data had already been overwritten by an earlier re-run before this was caught, so a same-sample matched comparison isn't possible either. The surviving sample (N=923, down from N=1,920) shows accuracy at 36.4% against a baseline of 60.0% -- but the baseline itself shifted 3 points from sample composition alone, which is enough to explain the accuracy change without attributing it to the fix. **Not reporting this as "the fix hurt accuracy"** -- that comparison is invalid, not just noisy. The direct mechanism check above (mega-cap/deep-value IV/price) is the trustworthy evidence for this specific fix; the accuracy question needs a clean, full-sample re-run once Yahoo's rate limit recovers (this session made several thousand ticker-level requests across multiple full-universe sweeps -- likely needs hours, not minutes).

**Reproduce:** `python scripts/wacc_capm_audit.py 12 0` for the intrinsic-value-to-price distribution (growth-fade effect only, pending the ERP-routing fix above); `python scripts/phase2_backtest.py 12 0` (or `24 12`) for accuracy and return spread; `python scripts/canonical_accuracy.py` for the one canonical number.

---

## Resume-ready framing

Numbers that survive a follow-up question are worth more than a single flattering percentage. Suggested framing, in that spirit:

- Built a point-in-time backtesting harness with explicit no-look-ahead controls (filing-lag gating, trailing-window beta, as-of pricing) across 1,000+ tickers and two independent historical windows; used it to find and remove a recommendation rule that measurably hurt accuracy (15–20 points on the affected subset) rather than assuming it helped.
- Diagnosed that a cross-encoder reranker was silently *degrading* RAG retrieval quality via a hand-labeled precision/NDCG/MRR evaluation, and shipped the fix (raw retrieval) with the disproved approach documented in place, not deleted.
- Fine-tuned a retrieval embedding model with contrastive learning (`MultipleNegativesRankingLoss`) and designed a frozen-candidate-pool evaluation methodology to isolate the embedding model as the only variable under test.
- Built an ML valuation classifier (Logistic Regression vs. XGBoost) with stratified k-fold CV, a held-out test split, and per-class metrics — while being explicit that a 39-row training set doesn't yet clear the bar for a production signal, and gating it out of the system's actual recommendation logic until it does.
- Rewrote a report-faithfulness evaluator after realizing its v1 metric was structurally unsatisfiable (penalizing exactly the paraphrasing behavior the generation prompt asked for) — an example of debugging an eval, not just a model.
- Ported the agent's planner-dispatches-tools control flow onto LangGraph as a StateGraph, kept alongside the original hand-rolled controller, and benchmarked the two — including catching and discarding a misleading first result (network I/O variance masquerading as a 30x orchestration difference) before reporting the real, isolated ~7-13ms dispatch-overhead number.
- Added a Redis caching layer with two deliberately different strategies (content-addressed for correctness-sensitive valuation/narrative output, TTL-only for genuinely time-bound statement data) and measured real 500-1,700x speedups on cache hits — after catching a cross-contamination bug in the benchmark's own methodology first.
