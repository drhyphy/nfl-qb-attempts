# Frozen probability calibration and conditional distribution experiment

Protocol date: September 14, 2026. These are research forecasts; neither experiment can promote itself or alter Codex/Claude recommendations. No Week 1 results enter fitting. Early bets do not require a confirmed starting lineup.

## What is implemented

`qb_attempts/shadow.py` publishes a market-only no-vig probability, one frozen calibrated blend, and one fixed conditional-distribution experiment. One exact book/line per QB is selected by book/line identity independently of EV, qualification, or eventual outcome. Probabilities in the output condition on no push; separate push masses are preserved. Recommendation selection remains outside this module.

The blend is `market + weight * (independent_model - market)`. Its one parameter is the closed-form Brier-minimizing convex weight, fit on 2022–2023 only. There is no ROI search, per-player tuning, over/under tuning, or 2026 fitting. The weight remains frozen after evaluating 2024. That historical period has been examined in earlier model work, so this is a retrospective development check, not an untouched test. The already exposed 2025 period is reported separately and never used to fit the weight or residual experiment.

The conditional experiment reweights chronological out-of-fold residuals according to expected margin, game total, and predicted attempts (a projected-volume proxy). It uses fixed Gaussian context bandwidths of seven and a global prior equivalent to 200 residual observations. It retains the observed empirical workload tails rather than replacing them with a Poisson variance. Only prior-season residuals enter any historical prediction. Live conditional predictions use the frozen residual archive through 2024. The production independent probability supplied to the blend remains the production model's probability, including its normal historical refreshes.

This is a meaningful small conditional-distribution test, not a full within-game simulator. It does not separately identify the probability of an injury exit, overtime, or actual time leading/trailing. The historical tails contain such outcomes, but interpreting them as calibrated individual exit hazards would be unsupported. A later play/drive model would need its own preregistered protocol and incremental validation before replacing this experiment.

## Archive reconstruction and audit

Original read-only input: `/Users/samuel/Projects/nfl-qb-attempts-edge/data/bettingpros/offers.parquet`.

The original Claude `market_backtest_quotes.parquet` is **not** used for fitting: its first row pairs Geno Smith's January 14, 2023 Seattle–San Francisco playoff quote with his September 13, 2022 Seattle–Denver Week 1 result. The raw provider marks that playoff event as season 2022/week 1, so joining by season/week/player is unsafe.

The new join requires exact UTC scheduled kickoff, the normalized pair of opponent teams, and normalized player identity. Ambiguous identities are excluded. Current provider player-team labels are ignored because they may identify a later team. Each actual sportsbook must have both sides at the same line, both timestamps strictly before kickoff and within 24 hours, with the two updates within 30 minutes of one another. A different real sportsbook at that exact line supplies the market comparison. Consensus and pick'em operators are excluded. The selected book/line does not depend on outcomes.

This rebuild cannot certify that the archived price was executable or available in New York. It is a retrospectively downloaded closing-reference archive. The underlying OOF game spread/total features are also closing references and could postdate an earlier prop quote. Therefore **these metrics are not a backtest of an executable morning strategy, and no historical ROI or validated market edge is claimed**. The daily prospective record is needed to resolve that limitation.

Source SHA-256 hashes and exclusion counts are embedded in `data/shadow/artifact.json`. `joined_archive.json` preserves selected exact-event joins, and `evaluation.json` preserves the causal probability records. These files are distinct from both live model archives.

## Initial locked result

From 29,632 raw side rows, timestamp/book checks retain 8,101; pairing retains 3,916 offers and exact event/forecast joins retain 3,312. Independent peer coverage and one-offer selection yield 911 QB games: 485 fitting observations, 425 in 2024, and only one eligible 2025 observation. Most 2025 provider updates are at/after kickoff; they cannot be rehabilitated as pregame quotes by rounding timestamps.

The fitted model weight is **0.1451258422**, giving the market approximately 85.5% weight. On the 425-row 2024 development check:

| Forecast | Brier | Log loss |
|---|---:|---:|
| Market only | 0.249195 | 0.691535 |
| Frozen calibrated blend | 0.249443 | 0.692037 |
| Independent model | 0.255876 | 0.705578 |
| Conditional residual experiment | 0.256223 | 0.706302 |

**Neither experimental improvement beats the market baseline in this development check.** All remain shadow-only; there is no automatic promotion, parameter retuning after seeing these numbers, or claim that the one-row 2025 result is meaningful evidence.

## Prospective path and operation

Run `../.venv/bin/python scripts/train_shadow.py` from the project root to reproduce the research artifact. Optional input paths and `--as-of` support auditing. Missing source data or insufficient chronological rows writes an explicit insufficient-data artifact. Fitting requires at least 200 non-push training rows and 100 validation rows. Runtime rejects a missing/malformed/future artifact, post-cutoff residual seasons, future quote timestamps, and started games, while leaving live recommendation arrays untouched.

The daily pipeline should store `score_shadow(candidates, now, artifact_path)` with the full snapshot, using all offered candidates. Freeze the first pregame prediction per QB/offer for prospective grading; only trustworthy contemporaneous, paired quotes should enter executable-price comparisons. Research forecasts can still be observed when source availability is uncertain. Do not pool repeated daily offers as independent outcomes.

The next review checkpoint is after at least six complete prospective regular-season weeks **and** 200 settled, source-verified shared-offer QB observations. If coverage is insufficient, continue collecting without moving the checkpoint opportunistically. Compare Brier/log loss and game-block uncertainty against market-only and both preserved models, with actual executable prices and valid closing-price records. ROI is secondary. Any promotion requires a separate documented review; the code always reports `promotion_allowed: false`.
