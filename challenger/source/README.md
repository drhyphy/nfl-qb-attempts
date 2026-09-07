# NFL QB Pass Attempts Edge

Model-vs-market engine for NFL quarterback pass-attempt props. Built 2026-09-06 (Claude Fable 5.1),
from scratch, after auditing the Kalshi / WNBA / MLB / NCAA-baseball / NFL-receptions projects and
Astra's `nfl-qb-attempts` attempt. **Read `PLAYBOOK.md` for judgment; this file covers architecture.**

Location: `~/Projects/nfl-qb-attempts-edge` (real path; `~/Desktop/Claude Code/nfl-qb-attempts-edge`
is a symlink — launchd cannot read Desktop). Python = system 3.13 at
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (same as the MLB/WNBA projects).

## Why this design (the lessons, in one table)

| Lesson (source) | Design consequence here |
|---|---|
| Raw model probabilities vs the market lost (WNBA v1 14-21, NBA frontier −9.6%); market-shrunk blends won (NCAA baseball: Elo+market beat the full ensemble) | The market is the prior. `mu_final = mu_market + λ·(mu_model − mu_market)`; λ is **fitted on a real historical prop archive**, not hand-assigned |
| Every earlier prop project calibrated on synthetic lines because no prop archive existed | BettingPros' public API keeps per-book closing lines + opening lines for pass attempts back to Nov 2022 (`src/bettingpros_archive.py`). λ, thresholds and the strategy ROI are measured on it |
| Price source beats edge size (MLB: FanDuel-main worst, slow books best; edge buckets flipped weekly) | Every book's line is priced separately; the market center is the median *implied mean* across books (distribution inversion), so a book that is 2 attempts off consensus is found even when no other book posts the same line (Astra's exact-line rule discarded this) |
| Trees + shrinkage compress extremes → fake fades of outliers (MLB Misiorowski/Eovaldi, WNBA Shepard) | Ridge is the base; LightGBM only as a monotone-constrained blend candidate; per-QB EB residual correction is a *candidate* that must earn its place; |model−market| > 6 attempts is held for review |
| Distribution shape drives P(over) mechanically (WNBA NB right-skew) | Empirical out-of-fold residual mixture with a heteroscedastic scale model; discrete integer support; pushes on integer lines |
| Early exits (injury/benching) pollute player averages (Daniels 2025 → 16.9 projection before fix) | QB level features use full games only (`qb_share ≥ 0.85`); exit frequency is its own feature; listed starters who never played are dropped (that prop is void) |
| Train/serve mismatch produced one-sided boards (MLB NaN cumsum, stale merge) | One `History.features()` path builds both training and live rows; a test asserts live features stay inside the training support; a leakage test mutates the target game and checks no feature moves |
| CLV against the wrong event / stale quote (MLB clv.py flaw) | CLV needs same player, same week, snapshot < 3h before kickoff; missing stays missing |
| Retrospectively selected rules look great (MLB hybrid unders, WNBA strict profile) | Policy frozen in `models/policy.json` from the dev-season backtest; 2025 archive is the untouched test; the live ledger is first-decision-only |

## Headline numbers (details and caveats in PLAYBOOK §1)
- BettingPros closing archive, 8,114 paired quotes / 1,773 QB-games (2022 wk10 – 2025).
- Market center MAE ≈ 6.6 attempts; model MAE ≈ 6.6; blend slope 0.39 (model adds information).
- Frozen policy (λ=0.35, 1.5-attempt dead zone, EV ≥ 3%) at **closing** prices: dev 2022-24 +3.3%
  (n=776), untouched 2025 +11.5% (n=668). Expect the dev number, not the 2025 number.
- Model vs recent-average baseline on 2025 holdout: MAE 6.68 vs 7.08, CRPS 4.72 vs 4.96.

## Data
- nflverse play-by-play 2017–2025 (`data/pbp/`), weekly player stats, schedules (spread/total, QB starters,
  rest, roof, coaches), depth charts, rosters. `src/fetch_data.py` refreshes the current season.
- BettingPros archive: `data/bettingpros/offers.parquet` (per event/player/side/book line+price+updated),
  `props.parquet` (consensus line, opening line, BP's own actual).
- Live odds (keyless): BettingPros market 333 (DK, Caesars, BetRivers, Fanatics in NY view) +
  ScoresAndOdds comparison JSON (FD, DK, Caesars, Fanatics, bet365, Hard Rock). The Odds API key in
  `~/Desktop/Codex/.env` is exhausted for the month and not used.

## Pipeline
```
src/features.py   pbp + schedule -> team-game table -> QB-game records -> as-of features (data/features.parquet)
src/model.py      expanding-season OOF for 5 candidates; dev 2022-24 CRPS selects; 2025 holdout; artifact models/model.pkl
src/market_backtest.py  archive quotes x OOF predictions -> market MAE, lambda fit, strategy ROI at closing prices
src/odds.py       live quotes (two sources) -> one schema, raw payloads archived
src/edges.py      slate -> starters -> features -> mu_model; quotes -> mu_market; blend -> EV per quote -> bet card + ledger
src/ledger.py     first-decision ledger, settlement vs nflverse, report
src/clv.py        snapshots near kickoff, CLV grading
run_daily.sh      fetch -> settle -> (Tue/Wed) rebuild+refit -> board -> CLV
```

## Run
```bash
python3 src/fetch_data.py                 # refresh nflverse inputs
python3 src/features.py                   # rebuild feature store (4 s)
python3 src/model.py                      # full validation + artifact (≈9 min); --fit-only for the weekly refit
python3 src/bettingpros_archive.py        # (re)download the historical prop archive (≈25 min, cached)
python3 src/market_backtest.py            # market backtest -> output/market_backtest_summary.json
python3 src/edges.py [--no-record]        # live board -> output/bet_card_latest.txt, board_latest.csv
python3 src/ledger.py settle|report
python3 src/clv.py snapshot|grade
python3 -m pytest -q tests
```
Automation: `com.samuel.nfl-qb-attempts-edge.plist` (6:30 daily) and `...clv.plist` (near kickoffs) are
written in the repo root but **not loaded**. Load with
`cp com.samuel.nfl-qb-attempts-edge*.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.samuel.nfl-qb-attempts-edge*.plist`.
