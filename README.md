# NFL QB Attempts

[Daily dashboard](https://drhyphy.github.io/nfl-qb-attempts/) · [Method and validation](research/MODEL_CARD.md)

An experimental NFL quarterback pass-attempts model with public sportsbook price comparisons. The board refresh is scheduled at **6:30 a.m. America/New_York** every morning using GitHub Actions and GitHub Pages. GitHub may delay scheduled jobs; the site displays the actual evaluation time and suspends stale recommendations. No local computer, sportsbook login, paid odds API, or API secret is required.

## Model

A regularized regression was selected using 2022–2024 expanding-season distribution scores from four prespecified alternatives. The final 2025 holdout is isolated from model selection. The count distribution uses prior-season out-of-fold residuals, preserves integer-line pushes, and includes early exits. The production estimate makes a small adjustment to median no-vig probabilities from **other sportsbooks at the same line**.

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

Daily workflow: `.github/workflows/daily.yml`. It refreshes stats, rosters, public prices and ESPN depth charts; trains and validates; records the exact decision; grades earlier records; and deploys the site. Both UTC trigger candidates are present, with an Eastern DST gate. A failed source run publishes an explicit no-bet failure status instead of retaining actionable stale picks.

The separate closing-price collector runs a schedule check at 7, 27 and 47 minutes during potential NFL kickoff hours. It fetches odds only when an official kickoff is within one hour and does not change morning recommendations. The result is a **near-kickoff quote**, not a guaranteed final sportsbook closing tick. Missing capture remains missing CLV.

## Permanent records

- `data/published/history/`: immutable timestamped boards, including holds and policy version.
- `data/published/closing/`: exact-event near-kickoff paired quotes.
- `data/published/performance.json`: first-decision-only flat-unit outcomes and CLV.
- `data/model/validation.json`: predictive evaluation and candidate comparison.
- `data/model/oof_predictions.csv`: reproducible historical out-of-fold forecasts.
- GitHub run artifacts: model binary and raw source evidence (90-day retention).

Grading takes the earliest pregame recommendation per game/player. Later runs cannot rewrite a losing decision or double-count it. Unknown results remain unresolved. CLV requires the same canonical game/player/line, paired prices, and a last-hour observation. No prior-event fallback is allowed.

## Scope

Recommendations use the New York book universe carried over from the earlier projects. Other real sportsbooks can inform consensus. No wagers are placed. Context is current public depth-chart information, not an official medical clearance. Read the model card for availability and historical timing limitations.

Data sources: [nflverse](https://github.com/nflverse/nflverse-data) (CC BY 4.0; transformed), [ScoresAndOdds](https://www.scoresandodds.com/nfl/props/pass-attempts), and [ESPN depth charts](https://www.espn.com/nfl/depth).
