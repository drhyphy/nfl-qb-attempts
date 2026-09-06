# NFL QB Attempts — V2 model card

Version attempts-v2.0 / independent-blend-v2. September 6, 2026. [Frozen protocol](V2_PROTOCOL.md) · [V1 model card](MODEL_CARD_V1.md).

## Lessons and revision

The prior-model audit is preserved in the V1 card. NCAA baseball supported market anchoring and immutable records. Successful MLB unders supported modeling workload and removal mechanisms; weaker NBA/WNBA variants showed that adding rules or reporting large estimated EV does not establish an edge. The MLB hybrid under subset returned +16.45% across 270 retrospectively reconstructed bets; this is promising evidence with selection and timing limitations, not a reason to require every new model to be maximally conservative.

V1 effectively required consensus-only profitability and sharply limited model disagreement. V2 removes that restriction. The attempt forecast is independent of the QB prop line; game spread/total supply explicit information about likely scoring conditions. The new selection policy remains a prospective hypothesis, not a proven improvement in betting returns.

## Data, target, and game state

The target remains official full-game attempts by the scheduled starting QB, including overtime and early exits. Sacks and scrambles are not attempts. Training keeps zero-attempt starters when team data exist. Features only use prior games with a six-hour availability lag and an explicit live decision-time cutoff.

[nflverse](https://github.com/nflverse/nflverse-data) schedules/statistics cover 2017–2025, 4,766 starter-games. V2 has 4,734 eligible records after missing historical game-market/history exclusions. Statistics retain QB recent attempts/variance/carries, team play volume/pass/sack rates, opponent opportunities, rest, venue, week, and coaching/team continuity. Added predictors are expected scoring margin, game total, absolute margin, and margin interactions with team pass rate and QB carries. A positive expected margin means the QB's team is favored; the displayed betting spread has the opposite sign.

Historical schedule spread/total fields are **closing references**, not an archive of 6:30 a.m. prices. This evaluation therefore does not reproduce historical morning information. Live scoring uses freshly retrieved, canonically matched paired sportsbook game spreads/totals from ESPN's public pregame scoreboard (currently DraftKings). It never substitutes historical closing fields for a missing current market.

We also aggregate prior play-by-play state passing tendencies: ahead by more than seven, within seven outside the final two minutes, and behind by more than seven. Eligible plays exclude kneels, spikes, penalties marked no-play, and conversions; dropbacks include sacks and scrambles, then an attempts/dropback conversion distinguishes official attempts. These are causal lagged rates. Upstream xpass is archived only as a diagnostic, excluded because its training vintage is unverified.

The tested script model predicts exposure to leading/close/trailing states, rather than assuming the favorite stays ahead. Its counted-state shares exclude close-state plays in the last 120 seconds, so they are an approximation, not exhaustive whole-game time shares. Multiplying those shares by all projected plays is also approximate. This candidate was not selected. Published state-share diagnostics carry this meaning. The selected regression can assign a small positive favorite effect for some QBs; it does not impose a monotonic trailing-means-more-attempts constraint. Pregame favorite status and realized in-game deficit are different variables. Historical expected margin correlates +0.007 with team attempts and +0.142 with play volume; realized final margin correlates −0.255 with team attempts. The positive partial pregame coefficient persists across development folds, so it is retained without forcing a preferred sign. Scenario means hold other inputs fixed and change the expected pregame margin to −7/0/+7; they are sensitivity checks, not conditional forecasts of a team actually leading at a specific time.

## Fixed selection and measured results

V1 ridge is the frozen comparator. Successor candidates were explicit-market ridge, tendency ridge, shallow tendency boosting, and script opportunity. Selection uses expanding-season 2022–2024 CRPS, with all evaluation-year residual calibration learned from earlier seasons. Extra complexity must improve CRPS at least 0.5% versus explicit-market ridge. The latter was retained: CRPS 5.2672 versus 5.2638 tendency ridge (only 0.064% better), 5.2898 boosting, and 5.3205 script opportunity. V1 CRPS was 5.2872.

**2025 was already examined during V1 development reporting. It is an exposed benchmark, not a freshly untouched holdout.** No selection changed after viewing it.

| Matched 2025 benchmark, 544 starters | V2 explicit-market ridge | Frozen V1 |
|---|---:|---:|
| MAE | 6.9715 | 6.9827 |
| RMSE | 9.0222 | 9.0345 |
| CRPS | 4.9896 | 4.9997 |
| 80% interval coverage | 80.33% | 80.88% |
| Fixed-threshold Brier | 0.16286 | 0.16317 |

Game-block bootstrap 95% intervals for V2 minus V1: MAE −0.0845 to +0.0625; CRPS −0.0513 to +0.0305. **The improvement is small and statistically uncertain.** Fixed thresholds 24.5/29.5/34.5/39.5/44.5 are distribution diagnostics, not historical bets. Week-1 V2 MAE was 6.1773 across only 32 starters.

The discrete distribution remains a smoothed empirical mixture of prior-season out-of-fold residuals, with integer-line push mass. Production refits on available historical data. No historical prop ROI or market-calibration advantage has been established. The future 2026 record is the prospective test.

## Prices and V2 qualification

[ScoresAndOdds](https://www.scoresandodds.com/nfl/props/pass-attempts) supplies paired public full-game prop comparisons; ESPN depth charts independently verify starting roles. Canonical schedule/roster identity, team, kickoff, market scope and timestamps must match. Pseudo-books, live markets and duplicate book votes are excluded. Recommendations use the inherited New York book universe; other real books can inform consensus.

The target book never sets its own benchmark. We take the median no-vig probability from other real books at the exact line, then blend with the independent forecast. Model weight is 50%, reduced to 35% for fewer than twelve NFL starts or a team/coach change. There is **no disagreement-based shrinkage, four-point adjustment cap, 15-point disagreement veto, maximum-EV veto, or consensus-only profitability requirement**.

Predeclared gates:

- Blended EV at least 3%; positive independent-model EV; positive EV after subtracting two percentage points from the conditional blended win probability.
- At least one other book at the exact line. One-peer markets and wide peer ranges are warnings. An anomalous target-book no-vig price more than ten points from peers remains a source-quality gate.
- At least four NFL starts, active roster, independently verified starter and no listed QB injury needing review.
- Current prop and game-market observations at most thirty minutes old. Known prop source updates must be within twenty-four hours. Starting context observed within two hours, source payload within forty-eight hours and correct season.
- One offer per QB, at most one QB per game and five per board.

A two-point stress is sensitivity analysis, not a confidence bound. We archive probabilities and EV under market-only and 35/50/65/100% model weights for every evaluated offer. These weights were not optimized against historical odds or today's recommendations. Bet-to prices target 3% estimated EV at the stored probability; freshness and stress must be re-evaluated before using a later offer.

EV is p(win) × net payout − p(loss). Market probabilities are conditional on no push; model push mass is restored before EV calculation. Observed timestamps reflect feed retrieval, not sportsbook tick time; unavailable book-update times stay missing.

## Record, deployment, and limitations

Every complete run preserves its exact normalized inputs, forecast, policy, warnings, holds, recommendation and model hash. The earliest pregame recommendation per canonical game/player remains the graded decision; later policy changes cannot rewrite it. Exact-event, exact-line last-hour quotes inform CLV; missing closes stay missing. Model/raw artifacts retain ninety days; normalized history stays in Git. GitHub schedules a refresh at 6:30 a.m. Eastern with a DST gate; job delays are possible and the dashboard shows actual time. Source failures publish a non-actionable failure board.

Weather forecasts, coordinator identities, receiver/line injury effects and player-specific early-exit hazards are not quantitatively modeled. Depth charts are not medical clearance. The empirical residual distribution is pooled and may miss matchup-specific uncertainty. Current corrected historical data are not archived as-of publications. Both sophisticated and simple models may fail against prices. Compare this model and Fable at the same pregame deadline, books and prices using paired calibration, exact-line CLV and first-decision ROI with game/week uncertainty, rather than claimed EV alone.
