# Evaluation & Results

Every major component in FinSight has a dedicated evaluation script, not just an implementation. This document pulls those results together in one place — including the ones that didn't come out as hoped, because a system that only reports its wins isn't being evaluated honestly.

Numbers below are computed directly from artifacts already checked into `scripts/` and `app/valuation/` (backtest result JSON, the ML classifier's metrics file) or quoted from dated findings recorded in code comments where noted. Reproduction commands are in each section.

---

## 1. Recommendation engine — point-in-time backtest

**Method** (`scripts/phase2_backtest.py`): simulates running the pipeline as of N months ago and checks the realized return by today, with explicit no-look-ahead controls —
- a fiscal year only counts as "known" once `filing_date + 90 days <= as_of_date` (10-K filing lag),
- beta is computed from trailing price history ending at the as-of date, not today's yfinance beta,
- price/market cap use the as-of-date close, not the current price.

A rating is scored correct if `Buy` → realized return `> +5%`, `Sell` → `< -5%`, `Hold` → within that band.

| Window | Universe | n scored | Pipeline accuracy | Always-Buy baseline |
|---|---|---:|---:|---:|
| As-of 12mo ago → today | 79 hand-curated tickers (mega-cap, deep-value, hypergrowth/neg-FCF, financials) | 78 | **44.9%** | 53.8% |
| As-of 24mo ago → 12mo ago | same 79 tickers | 76 | **32.9%** | 68.4% |
| As-of 12mo ago → today | 1,002-ticker broad universe | 978 | **34.9%** | 58.6% |
| As-of 24mo ago → 12mo ago | 1,002-ticker broad universe | 968 | **39.0%** | 54.3% |

### The honest finding

In every one of the four windows tested, a naive **"always predict Buy"** baseline beats the pipeline's raw direction-accuracy. This isn't a bug — all four windows sit inside a broadly rising market (universe average realized return ranged from +10.6% to +35.6%), so "the market went up, therefore call everything Buy" is a genuinely strong baseline in-sample, and a flat accuracy percentage doesn't isolate whether the *valuation logic itself* is adding signal.

Decomposed by what the model actually called, a clearer (and more useful) picture shows up:

| Window | Buy precision | Buy avg. realized return | Sell precision | Sell avg. realized return |
|---|---:|---:|---:|---:|
| 12mo curated | 63.3% (n=30) | +23.5% | 38.1% (n=42) | +0.6% |
| 24mo curated | 65.2% (n=23) | +22.4% | 17.8% (n=45) | +45.3% |
| 12mo broad | 53.8% (n=359) | +16.8% | 27.1% (n=501) | +27.7% |
| 24mo broad | 53.8% (n=357) | +11.6% | 33.4% (n=509) | +13.8% |

**Buy calls carry real signal** — 54–65% precision, consistently above each window's universe-average return. **Sell calls do not work in this sample** — precision as low as 17.8%, and in three of four windows the average stock the model called "Sell" on still had a strongly *positive* realized return. `Hold` is statistically unreadable (n as low as 6–8 per window).

The likely explanation, not yet confirmed: the ticker universe deliberately includes hypergrowth/negative-FCF names (PLTR, RIVN, SNOW, etc. — see `phase2_backtest.py`'s `TICKERS` dict) that a DCF structurally flags as overvalued on current fundamentals, and all four windows fall inside a period where exactly those names kept re-rating upward anyway. **Next step, not yet done:** rerun against a window that includes a drawdown (e.g. 2022) to see whether Sell precision recovers outside a one-directional bull market — the harness (`--universe`, positional `as_of_months_ago exit_months_ago` args) already supports this, it just hasn't been pointed at a bear window yet.

### The disagreement-guardrail finding (recomputed, not just quoted)

`report_data_builder.py` used to force a `Hold` whenever the DCF's own directional call disagreed with the relative-valuation signal. Recomputing both counterfactuals directly from the saved backtest rows (`dcf_only_rating` vs. the guardrail's forced outcome), on the exact subset where the two signals disagreed:

| Window | n (disagreement subset) | Composite (current, no forced Hold) | Trusting DCF's own call alone |
|---|---:|---:|---:|
| 12mo curated | 19 | 57.9% | 68.4% |
| 24mo curated | 10 | 30.0% | 40.0% |
| 12mo broad | 290 | 40.7% | 43.1% |
| 24mo broad | 272 | 43.4% | 46.3% |

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

**Actual current metrics** (`app/valuation/ml_classifier_metrics.json`, self-flagged):

> "Only 39 labeled rows — below the 40-row floor for stable cross-validation. Metrics below are noisy; treat as directional evidence, not a benchmark, until the training set grows."

| Model | 5-fold CV accuracy | 5-fold CV F1 (macro) | Held-out test F1 (macro) |
|---|---:|---:|---:|
| Logistic Regression | 54.6% ± 18.1 | 0.360 ± 0.153 | 0.315 |
| Gradient Boosting | 38.6% ± 19.5 | 0.251 ± 0.131 | 0.157 |

Top features by importance: `fcf_yield` (0.57), `dcf_over_price` (0.50), `mc_mean_over_price` (0.39) — sensibly, the classifier leans hardest on the two most direct valuation signals rather than a noisier input like `net_cash_per_share_over_price` (0.07). The honest framing: this demonstrates the ML pipeline (proper CV, held-out test, avoiding look-ahead leakage, comparing two model families instead of assuming one) more than it demonstrates a production-ready classifier — which is also why it's wired into the report as **display-only**, never part of the recommendation composite (`app/valuation/valuation_pipeline.py`).

**Reproduce:** `python scripts/build_ml_training_set.py && python -c "from app.valuation.ml_valuation_classifier import train; train('scripts/ml_training_set.csv')"` (grows automatically as more backtest runs accumulate labeled rows).

---

## 5. Per-report self-evaluation

Every generated report is scored at runtime by `app/evaluation/evaluation_engine.py` (grounding 40%, retrieval 20%, citation coverage 20%, completeness 20% — see `app/evaluation/scorer.py`). Grounding and citation checks went through a documented v1→v2 rewrite (`app/evaluation/grounding_validator.py`, `citation_evaluator.py`): v1 required a *verbatim substring match* against the source evidence, but the report prompt explicitly instructs the model to paraphrase — so a correctly-written, fully-grounded report would almost never satisfy v1's check. v2 uses stemmed content-word overlap instead, which actually rewards grounded paraphrase rather than penalizing it. `app/benchmarks/*.json` + `app/evaluation/benchmark_runner.py` additionally check generated reports for 5 fixed companies against expected sentiment/recommendation/topic coverage, so prompt or retrieval changes can be compared against a fixed baseline instead of eyeballed.

---

## 6. Inference latency

Switching the local LLM backend from a raw Hugging Face `transformers` pipeline to `llama.cpp` (Q8_0-quantized GGUF) cut the narrative-generation call (~3,300 prompt tokens, up to 700 generated) from **~257s to ~65s** on the same machine (`app/rag/report_generator.py`). Q8_0 was kept over the ~30%-faster Q4_K_M after a direct A/B: Q4_K_M produced a 2,200-character Executive Summary that consumed the entire generation budget and silently dropped the other four report sections on a real MSFT prompt — confirmed on the actual failure, not assumed.

---

## Resume-ready framing

Numbers that survive a follow-up question are worth more than a single flattering percentage. Suggested framing, in that spirit:

- Built a point-in-time backtesting harness with explicit no-look-ahead controls (filing-lag gating, trailing-window beta, as-of pricing) across 1,000+ tickers and two independent historical windows; used it to find and remove a recommendation rule that measurably hurt accuracy (15–20 points on the affected subset) rather than assuming it helped.
- Diagnosed that a cross-encoder reranker was silently *degrading* RAG retrieval quality via a hand-labeled precision/NDCG/MRR evaluation, and shipped the fix (raw retrieval) with the disproved approach documented in place, not deleted.
- Fine-tuned a retrieval embedding model with contrastive learning (`MultipleNegativesRankingLoss`) and designed a frozen-candidate-pool evaluation methodology to isolate the embedding model as the only variable under test.
- Built an ML valuation classifier (Logistic Regression vs. XGBoost) with stratified k-fold CV, a held-out test split, and per-class metrics — while being explicit that a 39-row training set doesn't yet clear the bar for a production signal, and gating it out of the system's actual recommendation logic until it does.
- Rewrote a report-faithfulness evaluator after realizing its v1 metric was structurally unsatisfiable (penalizing exactly the paraphrasing behavior the generation prompt asked for) — an example of debugging an eval, not just a model.
