# NFL QB Attempts — model card

Version: attempts-v1.0 / market-anchor-v1. Built September 6, 2026.

## Lessons from earlier models

An audit of saved code, cards, results and ledgers preceded implementation. These are model records, not personal betting-account fills.

| Evidence | Result | Design consequence |
|---|---|---|
| NCAA baseball saved canonical cards | 518 graded bets, +128.804 units, +24.87% ROI; 115 unresolved game/bet records and one doubleheader ambiguity | Preserve exact decisions and unresolved outcomes |
| NCAA baseball 1,136-game component comparison | Market log loss .65032; full market-shrunk model .64495; unshrunk full model .65677; Elo + market .64280 | Simple market anchoring can outperform a larger ensemble |
| NBA frontier 60 evaluated props | Mean claimed EV 50.5%, realized ROI −9.6%, Brier .329 | Large apparent edges can mean probability error |
| WNBA first-seen bets | 213 bets +9.14% overall; newer recent-form profile −11.98%, stricter profile −10.27% | Do not repair losses with retrospectively selected rules |
| MLB current-policy retrospective ledger | 1,970 bets −4.50%; selected hybrid unders 270 bets +16.45%, retrospectively reconstructed | Promising selected strategies need prospective confirmation |
| MLB recorded CLV | Mean −4.99%; code could select prior-game quotes; recent-only prices still negative | Exact event and quote age mandatory; missing closes stay missing |

These periods and strategies differ. This is a set of design lessons, not a comparable tournament. The detailed source-linked local audits are preserved separately from public source.

## Target, data and timing

The target is official full-game pass attempts by the listed starting QB, including overtime and early exits. Sacks and scrambles are not pass attempts. Training selects the scheduled starter, never the QB with the most realized attempts. Zero-attempt starters are retained when team statistics exist; a feed gap could still look like an early exit. Settlement requires explicit player outcome data and leaves missing rows unresolved.

The initial dataset contains nine regular seasons (2017–2025), 4,766 starter-games, from [nflverse statistics and schedules](https://github.com/nflverse/nflverse-data). Attribution: nflverse, CC BY 4.0, transformed. Availability follows the [nflverse data schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html).

Features include shrunken QB attempts over five/twelve starts, variability and carries; current team attempts, play volume, pass rate and sack rate; opponent attempts and plays allowed; rest, week, home/neutral site, roof, coaching continuity, team changes and experience. A traded QB retains player history but uses the new team's context. A rookie cannot inherit veteran certainty.

Features use prior games only. A six-hour post-kickoff lag excludes same-game and simultaneous results. Live scoring also enforces an explicit decision-time cutoff after all inputs are fetched. Historical data are currently corrected files, not archived as-of publications. The holdout is a pregame statistical prediction test, **not an exact replay of historical 6:30 a.m. information sets**. Historical closing spreads/totals and actual game weather are excluded because historical morning timestamps are unavailable.

## Selection and final test

Four fixed candidates: shrunken recent average, standardized ridge (alpha 150), shallow histogram boosting, and projected plays × projected attempt rate. Hyperparameters were fixed before viewing the final holdout. Candidate selection uses minimum 2022–2024 development CRPS, a proper score of the complete count distribution (lower is better). Expanding-season predictions from 2019–2021 initialize residual calibration. Each evaluation year uses only earlier seasons' residuals.

Ridge won development CRPS: 5.2872 versus 5.3651 recent average, 5.3287 boosting and 5.2888 opportunity model. The entire 2025 season was reserved for final evaluation. There was no subsequent model selection based on 2025 or current odds.

| Untouched 2025 holdout | Ridge | Recent-average baseline |
|---|---:|---:|
| QB starts | 544 | 544 |
| MAE, attempts | 6.9827 | 7.3598 |
| RMSE, attempts | 9.0345 | 9.4717 |
| CRPS, attempts | 4.9997 | 5.2565 |
| 80% interval coverage | 80.88% | — |

A paired game-block bootstrap gives a 95% MAE-difference interval of −0.569 to −0.192 attempts, grouping dependent QBs within games. This demonstrates predictive improvement over recent averages, **not over sportsbook forecasts**. Week 1 has only 32 starters: MAE 6.3255, RMSE 7.8424.

Brier .16317 and log loss .49757 use a prespecified fixed threshold grid: 24.5, 29.5, 34.5, 39.5 and 44.5. These are distribution diagnostics, not archived book lines or thousands of independent bets. The thresholds are correlated within each QB/game.

Probability distributions use a smoothed empirical mixture of prior-season out-of-fold residuals, one-attempt fixed bandwidth, zero censoring and discrete counts. Integer lines preserve push mass. This avoids a Poisson variance assumption and includes early-exit tails. No 2025 labels enter the residual distribution evaluated on 2025. Production refits the selected estimator on available history and uses historical OOF residuals.

## Public prices and launch policy

[ScoresAndOdds](https://www.scoresandodds.com/nfl/props/pass-attempts) provides public full-game comparison prices. Its public schedule JSON-LD provides dates; each offer is independently matched to nflverse event, participant teams, roster and kickoff. [ESPN depth charts](https://www.espn.com/nfl/depth) independently check the starter and injuries.

Only available paired over/under prices at real books count. Pick'em, consensus pseudo-books, live and non-full-game markets are excluded. Duplicate book observations receive one vote. Recommendations use the New York book universe inherited from prior projects; other real books may inform consensus.

The target book is excluded from its own benchmark. The anchor is median no-vig probability from other books at the **exact same line**. Different lines are not silently interchangeable. A thin-market diagnostic can be displayed but never qualify. Model weight is at most 15% early-season, 25% later; sparse experience, team/coaching changes and disagreement reduce it. The final estimate cannot move more than four points from consensus. These launch weights are conservative priors, **not weights fitted to historical NFL prop prices**.

Frozen qualification rules:

- Estimated EV ≥2.5% and positive EV after a 1.5-point adverse probability stress, also removing any beneficial model adjustment.
- At least two other books at the same line; no dispersed consensus or extreme target-book outlier.
- Hold model/market disagreement beyond 15 points and estimated EV above 15%.
- At least four NFL starts, active roster, independently verified starter, no listed QB injury requiring review.
- Retrieved quote ≤30 minutes old; reject future timestamps, mismatched events and known source updates older than 24 hours.
- Verify context observation ≤2 hours, source payload ≤48 hours and correct season.
- One side/line/book per QB, at most one QB per game, at most five recommendations.

The stress test is sensitivity analysis, not a confidence interval. Retrieval timestamps are not sportsbook tick times. Book update timestamps exist only for some prices; missing values remain null. A price can be unchanged for hours, so source-update age is not itself proof it is stale; the 24-hour gate is a conservative exclusion policy. Fresh aggregator quotes still need checking at the sportsbook. The morning card does not update model probabilities continuously.

Unit-stake EV = p(win) × net payout − p(loss). Pushes refund stake. Paired market probabilities are conditional on no push; model push mass is restored for unconditional EV. Bet-to prices target at least 2% estimated EV at the stored probabilities, rounded toward an actually sufficient price.

## Prospective record and comparison

Every complete run has an immutable timestamped file, policy and model hash. Grading freezes the first pregame recommendation per canonical game/player. Later prices and policies cannot replace it or double-count it. Missing/conflicting outcomes remain unresolved. Returns assume flat one-unit stakes at observed offers, not confirmed fills.

The separate near-kickoff collector captures only events starting within one hour. CLV needs paired prices for the same game/player/exact line within that hour. A prior game cannot fill a missing close. This is a near-kickoff reference, not a guaranteed final tick; delayed GitHub jobs may miss windows. Raw artifacts retain 90 days; normalized boards and closing quotes stay in Git history.

For a fair Fable comparison, freeze both versions before outcomes; use the same quote deadlines, books, prices and first-decision rules. Include abstentions. Compare paired Brier/log loss at common offered lines, exact-line CLV and flat-unit ROI with uncertainty grouped by game/week. Do not compare claimed EV or a single week's wins alone. A learned model/market weight needs archived chronological prop prices and a later untouched test that improves on market-only forecasts.

## Material limitations

There is no historical pass-attempts sportsbook archive, proven market edge or verified betting ROI at launch. Regression does not explicitly simulate drives/game scripts; current market consensus supplies much of that information. Weather forecasts, coordinator changes, offensive-line/receiver injury effects and team-specific early exits are not quantitatively modeled. Offensive injuries are retained for review, but only QB injury flags gate bets. Depth charts are not official medical clearance. Week 1 and team changes are especially uncertain. Feed omissions, revisions or failed checks trigger abstention. Improvements must earn their place through independent validation rather than tuning today's board until bets appear.
