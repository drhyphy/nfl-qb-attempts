# PLAYBOOK — NFL QB Pass Attempts Edge

Read this first. README covers architecture; this covers judgment: what the evidence says, what is
frozen, how to run it, what has broken, what to do next. Built 2026-09-06 (Claude Fable 5.1).

## 1. What the evidence actually supports (as of 2026-09-06)

Source: `output/market_backtest_summary.json`, `output/policy_analysis.json`, `logs/market_backtest.log`.
Archive: BettingPros per-book closing lines for pass attempts, 2022 wk10 – 2025 wk18, 8,114 paired
two-sided quotes on 1,773 QB-games, truth = nflverse official attempts.

**The market is about as accurate as the model.** MAE vs actual on matched QB-games:

| season | n | market center | ridge_eb model |
|---|---:|---:|---:|
| 2022 | 228 | 6.54 | 6.45 |
| 2023 | 519 | 6.63 | 6.66 |
| 2024 | 496 | 6.57 | 6.58 |
| 2025 | 530 | 6.63 | 6.65 |

**But the model carries information the market does not fully price.** Regressing (actual − market)
on (model − market) gives a slope of 0.39 (dev seasons). Blending improves MAE (dev 6.55 vs 6.59;
2025 6.61 vs 6.63) and log loss of P(over) at every book's own quote (2025: book price .6922, market
center .6904, blend λ=0.35 .6887). Small numbers — this is a ~0.5-1% probability edge, which is what a
real edge looks like in a mature market.

**Strategy at closing prices** (bet every real-book quote with blended EV ≥ 3%, ≤ 15%, ≥ 3 books):

| policy | dev 2022-24 | 2025 (untouched) |
|---|---|---|
| consensus shopping only (λ=0) | +4%/+1%/0% by season, n=234 | +16.4%, n=281 |
| ridge_eb λ=0.35, dead zone 1.5 (**frozen**) | +3.3% [−5%, +12%], n=776 | +11.5% [+0.2%, +21.8%], n=668 |
| ridge_eb λ=0.55 | +1.9%, n=1305 | +0.5%, n=1115 |

Closing prices are the harshest benchmark: props open Tue-Wed and the board runs daily, so live bets
are placed at softer earlier prices. A strategy that is +3% at the close is very likely +EV earlier.
Caveats that must stay attached to these numbers:
- 2025 is much stronger than dev, and 2025 also has 2-3 more books in the archive (Caesars, Fanatics,
  theScore). The edge is substantially **line-shopping** (a book 1-2 attempts off the multi-book
  center), amplified by the model. Expect dev-like (+3%) not 2025-like (+11%) going forward.
- λ=0.35 was chosen as the compromise between dev log-loss-best (0.20) and dev MAE-best (0.55) *after*
  seeing that 0.35 was also 2025's log-loss best. Mild peeking; the log-loss curve is flat 0.2–0.4.
- The dead zone (ignore |model − market| < 1.5) came from dev per-bin slopes and held on 2025.
- Aggregator archives can list lines that were not actually bettable at the recorded time. The result
  barely changed when restricted to quotes updated within 6h/24h of kickoff, which is reassuring but
  not proof.
- **Side rule rejected**: overs were the profitable side in dev (+7.7% vs −1.7%), unders in 2025
  (+20% vs −4%). The MLB "fade overs on workload props" pattern did NOT replicate here. Do not add a
  side filter without ≥200 settled ledger bets per side.
- EV buckets were flat (3-5%: +0.8%, 5-8%: −1.2%, 8-12%: +1.5%, 12-20%: +8%). Edge size is not a
  quality signal (same as MLB). Price source (the outlier book) is the signal.

**Model validation** (`models/validation.json`): candidate `ridge_eb` (ridge + per-QB EB residual
correction, K=20) won dev 2022-24 CRPS 4.812 vs ridge 4.815, LGBM 4.881, recent-average 4.974.
2025 holdout: MAE 6.68 vs 7.08 recent baseline (95% CI on the difference [−0.57, −0.20]), CRPS 4.72
vs 4.96, 80% interval coverage 81.6%. Heteroscedastic scale model: no CRPS gain → disabled
(`use_scale_model: false`; scale factor = 1). Astra's ridge reported 6.98 MAE on 544 starts; ours is
not directly comparable because scratches (listed starter, zero snaps) are excluded here.

## 2. Frozen policy (`models/policy.json`) — do not tune on the live board

- Candidate ridge_eb; mean blend `mu_final = mu_market + 0.35·gap` where gap = mu_model − mu_market
  and gap is zeroed when |gap| < 1.5.
- Market center = median over real books of the mean implied by each book's de-vigged price at its
  own line (inverting the model's residual distribution). Needs ≥ 3 real books.
- Bet when EV ≥ 3% on the blended probability, EV ≤ 15%, price ≥ −250, quote updated ≤ 36h ago,
  |gap| ≤ 6 (larger = model or data problem, review by hand), NY-licensed book, one bet per QB (best
  EV), one QB per game.
- Review date: 100 settled decisions or 2026 week 8. Until then: read CLV before W-L.
- Stakes: flat 1u (1% bankroll) until 100 settled bets with non-negative CLV; then quarter-Kelly
  capped at 3%.

## 3. Daily operation

- `run_daily.sh` (launchd 6:30, **not loaded yet** — see README): fetch → settle ledger → Tue/Wed
  rebuild features + `model.py --fit-only` → `edges.py` (records first decisions) → `clv.py grade`.
- Props for the week post Tue-Wed; the Sunday-morning board is the most complete. Re-run
  `python3 src/edges.py` any time (cheap; first-decision rule keeps the ledger honest).
- Read `output/bet_card_latest.txt`. RECOMMENDED = qualified rows; WATCHLIST = best side for every QB
  with the holding reason. `bet_to` = worst price still worth 2% EV; if the price moved past it, pass.
- `clv.py snapshot` near kickoffs (second plist) → `clv.py grade` computes CLV at the last snapshot
  ≤ 3h before kickoff. No snapshot = no CLV (never a prior week's quote).
- Season rollover: `fetch_data.py` derives the season from today's date; features/backtests derive
  splits from the data. First weeks of a season lean on prior-season history (fine — dev per-week
  bins suggested the model is *more* valuable in weeks 1-3, when books lean on stale priors).

## 4. Failure modes seen while building (and fixes)

- **Early exits / scratches poisoned QB averages** (Daniels projected 16.9): QB level features now
  use full games only (`qb_share ≥ 0.85`); listed starters with no snap are dropped (prop voids).
- **BettingPros labels players with their current team** → 35% of archive quotes failed to match
  until matching switched to name + week + event teams. Live board matches on name + nflverse team,
  which is the current team, so it is fine there.
- **BettingPros `updated` after kickoff**: some rows are touched post-game; the backtest keeps
  −15 min ≤ age ≤ 72h.
- **`limit=50` on BettingPros /offers returns 400** — omit it (default page holds 10 offers, enough).
- **Generational suffixes** ('Patrick Mahomes II') — `odds.name_key` strips them.
- **The Odds API key is exhausted** (0/500 this month). Not needed: BettingPros + ScoresAndOdds are
  keyless. Both public keys can rotate — grab the new one from the site's network calls.
- **Week 1 team changers** (Cousins → LV, Rodgers → PIT): model leans on the QB's own history; the
  new-team flag reduces nothing yet. Held only if |gap| > 6. Judgment call: be skeptical of week-1
  new-team recommendations until the ledger says otherwise.

## 4b. Experiments tested and rejected (do not re-run without new evidence)

- **v1.1 QB-conditioned team tendencies** (2026-09-06). Motivation: Daniels projected 22.8 because
  Washington's late-2025 offense (backups, run-heavy) was his "team context". Blended team volume /
  dropback-rate features toward games this QB started (weight n/(n+3)). Result: dev CRPS 4.822 vs
  4.812, dev log loss at quotes .69196 vs .69148 (λ=.35), dev strategy ROI +2.4% vs +3.3%; 2025 marginally
  better on log loss, worse on ROI (+9.8% vs +11.5%). Rejected by the dev-first protocol. Artifacts in
  `models/v1.1_rejected/`. The |gap| > 6 hold covers the Daniels-type case for now; a cleaner fix
  (separate "QB returning after ≥ 4 missed games" flag) is on the roadmap.

## 5. Roadmap (priority order)

1. Ledger + CLV to 100 decisions, then fit λ on live outcomes (logistic on logit p_model, logit
   p_market) and re-check the dead zone.
2. Opening-line study: the archive has `opening_line`/`opening_created` per prop. Measure how far
   closers move from openers and whether the model predicts the move (that is the real early-week edge).
3. Add 2026 archive weeks as they close (`bettingpros_archive.py --seasons 2026`) to keep the
   backtest current; re-run `market_backtest.py` monthly, never to re-tune inside a version.
4. ESPN scoreboard spread/total fallback for games nflverse lacks (currently skipped).
5. Injury/depth-chart gate for skill players (WR1/OL) — not modelled; only QB identity is checked.
6. "QB returning after long absence" flag (Daniels 2026 wk1): shrink λ or hold; test on archive first.

## 6. For future assistant sessions

Memory file: `~/.claude/projects/-Users-samuel-Desktop-Claude-Code/memory/nfl-qb-attempts-edge-project.md`.
Astra's separate attempt lives at `~/Desktop/Codex/nfl-qb-attempts` (untouched). Verify any claim
about what works against `data/ledger/ledger.csv` and `output/market_backtest_summary.json`.
