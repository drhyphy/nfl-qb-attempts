# NFL QB Attempts

[Daily dashboard](https://drhyphy.github.io/nfl-qb-attempts/) · [Method and validation](research/MODEL_CARD.md)

An experimental NFL quarterback pass-attempts model with public sportsbook price comparisons. The board refresh is scheduled at **6:30 a.m. America/New_York** every morning using GitHub Actions and GitHub Pages. GitHub may delay scheduled jobs; the site displays the actual evaluation time and suspends stale recommendations. No local computer, sportsbook login, paid odds API, or API secret is required.

## Model

V2 adds current game spreads/totals and interactions with passing tendencies and QB carries. Four successor designs were evaluated using 2022–2024 expanding-season CRPS, with frozen V1 as a comparator. A simple explicit-market ridge was retained under the prespecified complexity rule. **2025 is an exposed benchmark**, not a new untouched test: V2 MAE 6.9715 versus V1 6.9827; the paired confidence interval includes no improvement.

The independent count forecast receives 50% weight alongside leave-target-book-out consensus (35% with limited history or team/coach changes). A large model disagreement alone no longer blocks a bet. Qualification requires ≥3% blended EV, positive independent-model EV, and positive EV after a 2-percentage-point miss in the blended probability. Alternative 0/35/50/65/100% model weights are recorded prospectively. These weights are not historically calibrated to prop odds.

At launch, the model has **no verified historical sportsbook ROI or market-calibration advantage**. The site can correctly publish **no qualifying bets**. Estimated EV and the probability stress test are forecasts, not guarantees or statistical confidence bounds.

## Run

Python 3.11 and Node 22:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_daily.py --refresh-data --train
cd site
npm ci
GITHUB_PAGES=1 npm run build
```

Daily workflow: `.github/workflows/daily.yml`. It refreshes stats, rosters, historical play-by-play aggregates, public prop prices, current ESPN game spreads/totals and depth charts; trains and validates; records the exact decision; grades earlier records; and deploys the site. Both UTC trigger candidates are present, with an Eastern DST gate. A failed source run publishes an explicit no-bet failure status instead of retaining actionable stale picks.

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
