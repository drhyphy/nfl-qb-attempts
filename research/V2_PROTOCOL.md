# V2 protocol, specified before fitting the upgrade

September 6, 2026. V1 validation and OOF forecasts are preserved in data/baseline_v1; V1 model.py and scoring.py remain unchanged. Development/selection uses 2022–2024 expanding-season validation. 2025 has been examined during V1 development: it is now an exposed historical benchmark, not a newly untouched test. No historical betting ROI is claimed. Future timestamped decisions provide the next uncontaminated market evaluation.

## Candidate set and selection

Fixed comparisons: frozen V1 ridge as the preserved comparator; four eligible V2 candidates comprising ridge plus pregame expected margin and game total; ridge plus spread/total and lagged state-conditioned pass tendencies; shallow boosting with those features; opportunity model using predicted state shares, lagged pass preferences by state, projected plays and attempt/dropback conversion. The successor selection rule compares the four V2 candidates; V1 remains the benchmark, not an eligible successor.

Historical pregame spread/total values are the nflverse schedule's closing references. Evaluation is explicitly a historical pregame-information proxy, not a replay of 6:30 prices. Live features use freshly observed independent game odds, never QB prop lines to predict attempts. Missing current game odds block publication for that matchup; no invented zero spread or total.

For script exposure, estimate the probability that a representative counted offensive play falls in a substantial lead (>7), close game (within7), or substantial deficit (<−7), conditional on pregame spread/total. Learn this mapping from prior training seasons. Multiply these probabilities by prior team dropback preferences in those situations, projected play volume, and historical attempt/dropback conversion. Integrate over scripts rather than using the realized game state at prediction time. Fit the state model only within each chronological training fold. The three-state approximation is tested against simpler explicit-market models, not assumed superior.

Implementation clarification from review: the neutral tendency excludes close-game plays during the final 120 seconds, while substantial lead/trail counts include all game times. The normalized shares are conditional on that counted subset. Applying them to projected total plays extrapolates those preferences into omitted late close-game situations and is an approximation. These are not exhaustive game-time shares. This documentation correction does not alter features, refit candidates, or retune after observing the comparison.

Tendency features use only earlier games (six-hour statistical availability proxy and live as_of cutoff). nflverse xpass/pass_oe is archived as a diagnostic but excluded from the default predictor set because the upstream fit vintage is not established; state-conditioned rates provide a transparent alternative without that dependency.

Choose lowest 2022–2024 CRPS, applying a 0.5% improvement requirement to replace the simpler explicit-market ridge with additional complexity. Report all candidates, year-specific scores, 2025 MAE/RMSE/CRPS/coverage, paired game-block bootstrap differences, calibration by fixed threshold and spread segment, and spread sensitivity. Candidate hyperparameters fixed in implementation before evaluation. Forecast superiority to a recent-average/V1 baseline does not prove market profitability.

## Production policy hypotheses

Keep identity, full-game market, observed quote validity, active starter and injury/context freshness as hard requirements. Remove the automatic model–market disagreement veto, four-point probability-movement cap, maximum-EV veto and requirement that market-only EV be positive. Use price disagreement as a diagnostic; preserve data anomaly checks.

Until archived historical prop odds support learned blending, use an explicitly provisional 50% model / 50% other-book consensus mix, reduced to 35% model for limited history (<12 starts), a team change or a coaching change. No further reduction merely for disagreement. Require 3% blended EV, positive EV after subtracting 2 percentage points from blended probability (without replacing it with market-only probability), and positive independent-model EV to avoid recommendations supported solely by consensus. At least one other actual sportsbook at the exact line is required; evidence from two or more is preferred and reported. The weight and thresholds are fixed hypotheses, not fitted or claimed optimal.

Evaluate model-only, 35%, 50%, 65%, and market-only probabilities in each timestamped row. Freeze prospective benchmark panels on identical observed offers; do not optimize weights on current recommendations. Disagreement is reported, not automatically fatal. Exposure remains one opinion per QB, maximum one QB per game, five per card; no wagers placed or bankroll promises.
