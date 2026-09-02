# FinSight — Accuracy Problem: Status Brief

Repo: FinSight-AI (DCF + composite scoring / ML classifier Buy/Hold/Sell research tool — the LLM narrates the verdict, it doesn't choose it; the recommendation is 100% deterministic and backtestable).

## Where this stands right now

FinSight has one canonical, reproducible accuracy metric (`scripts/canonical_accuracy.py`): **12-month forward directional accuracy** — of every Buy/Hold/Sell call the real production decision path makes on a broad, non-cherry-picked ~1,000-ticker universe (S&P 500+400+600, partial), what fraction are correct 12 months later. "Correct" = Buy needs realized return `> +5%`, Sell needs `< -5%`, Hold needs to land between.

**Current honest number: roughly 36-39% accuracy vs. a 57-60% "Always Buy" naive baseline, depending on exact sample. The model currently loses to doing nothing.** This is stated as the headline, not buried — a prior "70% accuracy" claim in circulation was actually analyst-agreement (does FinSight's call match Wall Street consensus), a completely different and much less meaningful metric, now retired.

A full session of work (documented in detail below) produced real, validated fixes to the underlying valuation logic, several honest "doesn't work" results that were caught before shipping, and one promising-but-unproven lead. The last *clean, full-sample* accuracy reading was **38.8% (N=1,920) vs. 57.0% baseline**, after a CAPM-sourcing fix. A subsequent, larger valuation-logic fix (below) is proven to work on its own terms but its effect on this headline number is **unconfirmed** — a rate-limiting episode corrupted the follow-up measurement (details below). That's the single biggest loose end.

---

## What's already been fixed and validated (don't re-derive these)

1. **CAPM inputs were hardcoded guesses (4% risk-free rate, 6% equity risk premium), now sourced.** Risk-free rate is a live 10-year Treasury yield; ERP is 4.45%, Aswath Damodaran's actual published implied ERP. Real, measured effect: **36.9% → 38.8% accuracy** (baseline moved 56.4%→57.0% too, expected drift).

2. **The DCF systematically undervalued high-quality/mega-cap companies relative to mature/deep-value ones — this was the biggest, most concrete finding of the session, and it's fixed and cleanly proven.** Root cause: every company's cash-flow forecast faded to the *same* flat terminal growth rate (4%) regardless of business quality, and the explicit forecast window didn't scale with quality either. Two fixes:
   - A 3-stage growth fade (durable, high-ROE companies hold their own growth rate longer before fading) — already existed from earlier work, partially closed the gap.
   - Quality-tiered terminal growth (same ROE tier, +1pt terminal growth for ROE≥25%, -1pt for ROE<15%) — new this session.
   - **Directly measured via the real pipeline** (not a stale bypass script): mega-cap vs. deep-value median intrinsic-value/price went from a **~3x gap (0.59 vs 1.77) → ~1.1x gap (1.18 vs 1.30)**. This is clean, trustworthy evidence — computed live, not confounded.
   - **What's NOT confirmed**: whether this also moved the headline accuracy number. A full backtest re-run to check hit severe Yahoo Finance rate limiting mid-session (a real yfinance library bug surfaced under load — a throttled response comes back malformed and crashes yfinance's own parser). The pre-fix baseline data had already been overwritten before this was caught, so there's no clean matched-sample before/after. The resulting confounded reading was 36.4% (N=923, down from N=1,920) vs. 60.0% baseline — **explicitly not attributed to the fix**, since sample composition alone (baseline moved 3 points) is enough to explain it. **This needs a clean, full-sample re-run once the rate limit recovers** (this session made several thousand ticker-level requests; likely needs hours of cooldown).

3. **The ML classifier (a secondary, currently display-only Buy/Hold/Sell signal) got meaningfully better as a standalone model, but doesn't help the real recommendation when tested honestly.**
   - Grew its training data 35x (49 rows → 1,700 rows) via more historical backtest windows.
   - Caught and fixed two real look-ahead data leaks in the process (live Treasury yield and live benchmark/sector data both leaking into historical point-in-time simulations — same bug class, different features).
   - Caught a class-imbalance trap: bigger data made raw accuracy look better (61%) purely because the model learned to always guess the majority label — the *real*, class-balanced result is 48.7% held-out accuracy, F1 0.42 (up from 0.41 at 39 rows, but on 35x more, statistically stable data).
   - **Tested whether blending it into the actual composite recommendation helps: it doesn't.** First test showed a huge apparent gain (44.1%→46.8pts as blend weight increased) — turned out to be in-sample leakage (the model was being tested on its own training data). Redone properly with leave-one-window-out evaluation: pooled accuracy barely moves (44.1%→44.8%), every tested blend weight lands within 1 point of every other. **Not wired in.**

---

## What's been tried and correctly NOT shipped (useful negative results — don't re-try these without new evidence)

- **A real bear-market test is currently impossible.** yfinance's annual-financials endpoint only exposes a *rolling* ~5-year window from *today*, not a fixed archive — 2022's bear-market fiscal data has aged out entirely (confirmed: every ticker fails when the backtest is pointed at a 2022 as-of date). The mildest available dip in the reachable window is a ~7% pullback (Dec 2024→Apr 2025), not a real correction, and even that window nets positive by the 12-month mark. **No evidence exists either way about how the model performs outside a bull market.**

- **A specific sector failure (semiconductors) was investigated and found to be a one-off, not a fixable pattern.** In the most recent 12-month window, the model's most confident Sell calls were semiconductor/semi-equipment names (STX, DELL, AMD, LRCX, KLAC, etc.) that then gained 90-420% — an AI infrastructure demand supercycle no backward-looking valuation could see coming. Checked the prior year's window: the correlation *reversed sign* (weak positive, not strongly negative). Building a "don't sell semiconductors" rule would be overfitting to one unprecedented historical episode, not fixing a real bug. Not built.

- **"Don't sell into a strong price trend" (momentum-gated Sell calls) — tested, doesn't work.** Blending 6-month price momentum into the composite score (asymmetric: only pulls scores away from Sell, never toward it) showed rising accuracy as the blend weight increased (44.1%→46.8%) — but Sell *precision* barely moved (27.5%→27.9%) even as Sell *volume* nearly halved. Confirmed directly: the momentum-filtered Sells (27.9% precision) scored *worse* than simply taking the same number of most-confident Sells with no momentum involved at all (29.2%). **Momentum isn't identifying which Sells are wrong — it's just calling Sell less often, which mechanically raises accuracy in a bull-dominated sample regardless of correctness. Not shipped.**

- **"Lower confidence right before an earnings report" — tested on 1,100+ real cases, evidence points the opposite way.** Calls made within 30 days of a known earnings date scored *better* (46.6%) than calls made with no report imminent (36.7%) — the reverse of the hypothesis. Not conclusive proof of a reverse effect (confidence intervals still overlap), but zero evidence supports the original idea. Not shipped.

## What's shipped but explicitly unproven (a live, promising lead)

- **A dividend discount model (DDM) — a second, independent valuation method (values the actual dividend cash flow, not modeled free cash flow) — was built, debugged, and tested.** Caught two real bugs before it was trustworthy: (1) companies with a token/symbolic dividend (e.g., NVDA, <10% of net income) were producing nonsense values — fixed with a materiality floor; (2) a raw multi-year dividend growth rate fed directly into a "grows forever" formula produced absurd results (e.g., MSFT) — fixed by capping the growth input, mirroring why the DCF itself fades to a modest terminal rate instead of extrapolating raw history.
  - **Tested for blending into the composite**: showed a real-looking interior peak (accuracy on the DDM-eligible subset rose from 49.3%→52.1% at a moderate blend weight, then fell off at higher weights — a materially different, more trustworthy shape than momentum's monotonic "just keeps climbing" red flag).
  - **But the sample is only 71 point-in-time dividend-payer observations** — far too small to trust a 2-3 point swing. Shipped as **display-only** on every report (clearly labeled, not influencing the rating), pending a larger data-generation pass to actually validate or kill this lead.

---

## Root-cause read on why accuracy is still low

1. **The prediction task itself conflates two hard things**: is the company mispriced (a valuation judgment), AND will the market correct that mispricing within exactly 12 months (a timing/repricing judgment). A DCF can be directionally right about value and still "lose" if the market takes 18 months instead of 12, or never fully agrees.

2. **Every available test window sits inside one continuous, powerful bull market**, and the true bear-market comparison (2022) is now permanently unreachable via this data source. "Always predict Buy" is a very strong baseline almost by construction in this regime, and there's no way, currently, to check whether the story is different in a down market.

3. **Sell calls are the demonstrated, isolated weak point.** Buy precision is consistently 54-65% across every window tested — genuinely fine, in line with published academic/analyst benchmarks. Sell precision is 13-35% and the average "Sell" pick still goes *up* in 4 of 5 tested windows. The failure is concentrated, not diffuse.

4. **The recommendation uses a narrow slice of what the system already computes.** Momentum, sentiment, financial-quality scores (Piotroski F-Score, Altman Z-Score), macro-rate sensitivity — all already computed for on-report *display*, none of it feeds the actual Buy/Hold/Sell decision, which is currently ~80% one DCF snapshot + 20% a relative-valuation cross-check (+ the new, unproven DDM signal, display-only).

5. **Sample sizes for testing new ideas are still fairly small** (tens to low thousands of rows depending on the test), which makes it easy for a genuinely fake effect to look real (this session caught that trap five separate times) and hard to fully validate a genuinely real one (the DDM result).

## Constraints worth respecting when proposing solutions

- **The recommendation engine is deliberately deterministic and backtestable** — the LLM only narrates, never decides. This is *why* everything in this brief could be measured and tested rigorously at all. Any proposal that lets an LLM's free-form judgment influence the actual Buy/Hold/Sell call would break that measurability and needs its own, much harder validation story — flag this explicitly if proposing it, don't fold it in quietly.
- **Data source is yfinance only**, with two now-confirmed real limitations: a rolling ~5-year window on annual financials (no genuine historical archive), and aggressive rate limiting under sustained heavy request volume (a real library bug surfaces as a crash, not a clean retriable error, when this happens).
- **Deploy target has an ephemeral filesystem** (free-tier hosting) — a live, production-call tracking approach that needs data to persist across months of uptime doesn't currently work; a git-committed, periodically-refreshed backtest artifact is the only currently-viable "canonical metric" mechanism.
- Every fix in this project gets shipped only after it clears a real, out-of-sample (or same-size-matched-baseline) test — a change that looks good on a headline number alone is treated as suspect until proven otherwise. Keep that bar.

## Where a fresh pass might productively focus

- Grow the DDM's dividend-payer sample (same kind of broad-universe, multi-window sweep that took the ML classifier from 49→1,700 rows) to actually validate or kill that lead.
- A genuine cross-sectional/rank-based reformulation of scoring (rank stocks against each other rather than an absolute ±5% threshold) — partially investigated (Spearman rank correlation near zero outside one sector anomaly), not fully explored as an alternative scoring *architecture*.
- Multi-method valuation triangulation beyond DCF+relative+DDM (e.g., sum-of-the-parts for conglomerates).
- A genuinely forward-looking, LLM-informed industry/competitive-context signal — the biggest conceptual gap, but the one that most directly threatens the deterministic/backtestable design this project has leaned on throughout. Worth exploring HOW to do this while preserving measurability, not just whether to do it.
