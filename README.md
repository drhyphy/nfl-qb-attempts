# NFL QB Attempts

[Daily dashboard](https://drhyphy.github.io/nfl-qb-attempts/) · [Method and validation](research/MODEL_CARD.md)

An experimental NFL quarterback pass-attempts model with public sportsbook price comparisons. The board refresh is scheduled at **6:30 a.m. America/New_York** every morning using GitHub Actions and GitHub Pages. GitHub may delay scheduled jobs; the site displays the actual evaluation time and suspends stale recommendations. No local computer, sportsbook login, paid odds API, or API secret is required.

## Model

V2 adds current game spreads/totals and interactions with passing tendencies and QB carries. Four successor designs were evaluated using 2022–2024 expanding-season CRPS, with frozen V1 as a comparator. A simple explicit-market ridge was retained under the prespecified complexity rule. **2025 is an exposed benchmark**, not a new untouched test: V2 MAE 6.9715 versus V1 6.9827; the paired confidence interval includes no improvement.

The independent count forecast receives 50% weight alongside leave-target-book-out consensus (35% with limited history or team/coach changes). A large model disagreement alone no longer blocks a bet. Qualification requires ≥3% blended EV, positive independent-model EV, and positive EV after a 2-percentage-point miss in the blended probability. Alternative 0/35/50/65/100% model weights are recorded prospectively. These weights are not historically calibrated to prop odds.

At launch, the model has **no verified historical sportsbook ROI or market-calibration advantage**. The site can correctly publish **no qualifying bets**. Estimated EV and the probability stress test are forecasts, not guarantees or statistical confidence bounds.

## Run

### September 14: verified early entry and frozen research

The website's offer cards now read `verified_recommendations`. The existing
`recommendations`, history and performance fields remain the original research
cohort and must not be interpreted as executable prices. `entry_watchlist`
explains current holds while preserving model probabilities and numeric rules.
Expected starters can qualify days before kickoff: **final role confirmation is
not required**. Unknown/questionable roles warn; fresh explicit unavailability
withdraws current suggestions without erasing previous paper entries.

`qb_attempts/quote_verification.py` checks an exact two-sided offer against public
FanDuel New York event markets and book-specific BettingPros New York offers,
retaining identifiers, availability, retrieval timestamps and raw evidence.
BettingPros evidence requires active matching two-sided markets and a fresh
update timestamp for each side; it is labeled timestamped comparison evidence,
distinct from a direct sportsbook observation. Its identifiers are provider IDs.
Current verified offers replace older comparison quotes before either model is
scored, with one main line per book. Model decisions and price collection status
are displayed separately. Missing timestamps remain missing. Cards expire after
30 minutes, even when the rest of the research board remains current.

`data/published/verified_ledger.json` starts a separate one-unit paper cohort,
with immutable first prices, one QB per game per model, and at most five pending
entries per model across refreshes. No wagers are executed. Final ESPN evidence
can reconcile pending results; missing statistics never imply zero attempts.
Evidence-backed DNPs are provisional paper voids, not assumed sportsbook rulings.
Closing CLV for this cohort requires the same verified paired-market evidence;
legacy feed CLV remains labeled as incomplete research evidence.

The morning schedule remains 6:30 Eastern with recovery checks. Intraday runs
at 01:05, 04:05, 14:05, 18:05 and 22:05 UTC update prices, role news and results.
GitHub can delay these jobs. Late refreshes never create another stake for an
existing game entry. Raw verification and settlement responses are retained in
workflow artifacts for 90 days.

The [shadow protocol](research/SHADOW_PROTOCOL.md) freezes a 14.51% independent /
85.49% market probability blend and a conditional residual experiment. Neither
beat market-only in the historical development check. Both remain research-only
and cannot promote themselves. See `data/shadow` for the exact event-join audit,
source hashes and causal fitting/evaluation records. No 2026 outcomes enter fitting.

### Commands

Python 3.11 and Node 22:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_daily.py --refresh-data --train
cd site
npm ci
GITHUB_PAGES=1 npm run build
```

Daily workflow: `.github/workflows/daily.yml`. It refreshes stats, rosters, historical play-by-play aggregates, public prop prices, current ESPN game spreads/totals and depth charts; trains and validates; records the exact decision; grades earlier records; and deploys the site. Both 6:30 Eastern UTC candidates are present, plus recovery checks at :47 UTC during hours 10–12 and :17 during hours 11–13. `scripts/morning_gate.py` skips candidates before 6:30 Eastern and skips later runs only when the **live Pages feed** contains healthy champion and challenger evaluations from this morning. This handles DST, delayed jobs, failed deployments, and failed challenger runs. A valid no-odds day counts as complete. A failed source run publishes an explicit no-bet failure status instead of retaining actionable stale picks.

GitHub schedules are best-effort, so these cloud retries cannot guarantee an exact start time. The existing local **Verify daily sports-model GitHub runs** automation also checks at 6:35 a.m. Eastern and dispatches this workflow if needed, then verifies the published result. This independent backup requires the computer to be on with the app running and GitHub CLI authentication available; cloud scheduling does not. For an idempotent recovery use `gh workflow run daily.yml --repo drhyphy/nfl-qb-attempts -f force=false`; omit that input for an intentional fresh evaluation even after a successful morning.

The separate closing-price collector runs a schedule check at 7, 27 and 47 minutes during potential NFL kickoff hours. It fetches odds only when an official kickoff is within one hour and does not change morning recommendations. The result is a **near-kickoff quote**, not a guaranteed final sportsbook closing tick. Missing capture remains missing CLV.

## Permanent records

- `data/published/history/`: immutable timestamped boards, including holds and policy version.
- `data/published/closing/`: exact-event near-kickoff paired quotes.
- `data/published/performance.json`: first-decision-only flat-unit outcomes and CLV.
- `data/baseline_v1/`: preserved original evaluation; `research/MODEL_CARD_V1.md`: original method.
- `data/model/validation.json`: predictive evaluation and candidate comparison.
- `data/model/oof_predictions.csv`: reproducible historical out-of-fold forecasts.
- GitHub run artifacts: model binary and raw source evidence (90-day retention).

Grading takes the earliest pregame recommendation per game/player. Later runs cannot rewrite a losing decision or double-count it. Unknown results remain unresolved. CLV requires the same canonical game/player/line, paired prices, and a last-hour observation. No prior-event fallback is allowed.

## Scope

Recommendations use the New York book universe carried over from the earlier projects. Other real sportsbooks can inform consensus. No wagers are placed. Context is current public depth-chart information, not an official medical clearance. Read the model card for availability and historical timing limitations.

Data sources: [nflverse](https://github.com/nflverse/nflverse-data) (CC BY 4.0; transformed), [ScoresAndOdds](https://www.scoresandodds.com/nfl/props/pass-attempts), and [ESPN depth charts](https://www.espn.com/nfl/depth).

## Claude Fable challenger

The dashboard now includes Claude Fable's model and a [source-linked comparison](research/CLAUDE_REVIEW.md). Both run on the same current quotes, canonical events, game odds and starting/health checks each morning. Claude's model and native policy are preserved in a portable, provenance-checked snapshot under `challenger/`; see its adapter notes for explicit operational differences.

The head-to-head record begins when both models run together. Separate first-decision ledgers prevent either model's picks from overwriting the other's. Paired forecasts use the first common pregame snapshot and one deterministic common book/line per QB, including held offers. MAE, probability calibration, flat-unit ROI and exact-line CLV are reported separately; past solo champion picks remain in their original history.

- `data/published/challengers/claude/history/`: immutable challenger cards.
- `data/published/comparison/champion/`: champion cards from the common evaluation period.
- `data/published/history/`: combined boards with the frozen paired forecasts.
- `data/published/evaluations/`: every model's evaluated offers and shared source snapshot.
- `research/claude_matched_benchmark.json`: 537 matched 2025 forecasts; exposed benchmark, no established winner.

Challenger update failures do not replace the valid champion card. They show an unavailable challenger and fail the job after the current status is published. The selected base model refits with new completed games; the launch residual pool and saved QB corrections stay frozen to avoid Claude's original incremental-residual bug. No policy retuning occurs automatically.
