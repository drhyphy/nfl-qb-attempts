# Claude challenger snapshot and adapter

This directory snapshots the existing Claude project at commit `9763b39ee5a6c3ab4c3a9231e83a308e546379fa` (clean working tree), September 6, 2026. `provenance.json` records SHA-256 hashes and sizes for the original sources and artifacts. The original project is not modified. Model and policy are preserved as a prospective challenger, not promoted over the champion.

- `source/`: verbatim original feature/model/market/live-board and audit-related source, README and playbook.
- `frozen/records.parquet`: original 4,722 historical starter records, 2017–2025.
- `frozen/selected_oof.csv`: original `ridge_eb` rows selected from the five-candidate OOF file; the original full file's hash is recorded.
- `frozen/model.json`: exact numerical parameters of the saved StandardScaler/Ridge model, 3,186 saved residuals and 131 saved QB EB corrections. No cross-version pickle loading or LightGBM dependency is needed in production. Original scaler was disabled; the format supports its saved parameters if present.
- `frozen/policy.json`, `validation.json`, `market_backtest_summary.json`, `policy_analysis.json`: original evidence and policy, including its own disclosure that the market blend was chosen with 2025 visible.
- `frozen/fidelity.json`: eight fixed historical feature rows evaluated with the original saved estimator and original function bodies. Tests compare numerical inference, integer/half-line probabilities, cross-line inversion and final mean to absolute tolerance `1e-12`.
- `engine/`: unchanged original feature builder, verbatim selected model function bodies, and original market math with package-relative imports. Unused candidate-training imports and LightGBM are omitted. `source/` retains the full original implementations.

## What is preserved

The independent model is ridge plus the exact saved per-QB empirical-Bayes corrections. The probability distribution uses the source empirical residual mixture with one-attempt Gaussian smoothing and its original interpolation grid. The frozen selected model has no heteroscedastic scale model.

At each player's market, each paired book/line's no-vig over probability is inverted to an implied attempts center. The median includes the target book, as in the source. All eligible real books at all offered lines contribute; duplicate observations of the same book/line get only their latest observation. Book-count eligibility uses distinct books. As in the source, a book offering several different lines can contribute several implied centers; the number-of-books rule still counts that book only once.

The final mean uses 35% of the model–market difference, unless its absolute size is below 1.5 attempts, where it uses the market center directly. Native eligibility requires 3–15% EV, at least three books, odds at least −250 and a model–market center gap no greater than six attempts. Original change-of-team/coach and fewer-than-four-starts flags remain informational. Recommendations use one QB per game and have no five-bet card cap. EV ranking uses the source's four-decimal EV precision. Ties retain deterministic shared-quote order.

The original `bet_to` helper defaults to 2% EV although the recommendation floor is 3%. That value is preserved and explicitly labeled `bet_to_target: 0.02`; it must not be advertised as a 3%-EV execution threshold.

## Shared deployment adaptations

`qb_attempts/challenger.py` receives the champion's already resolved quotes, schedule, current game odds, starter/injury context and final decision timestamp. It does not obtain a separate advantageous quote snapshot. It also rechecks game identity, paired American prices, known source age, observation freshness and current spread/total availability before any quote influences the market center. Shared source rules exclude known source updates older than 24 hours, which is stricter than Claude's native 36-hour rule.

Recommendations require the same independently verified active starter, injury clearance and context freshness as the champion. These are **adapter rules**, not claims about the original Claude model. Every evaluated offer records `native_reasons`, `shared_safety_reasons` and informational `warnings` separately. Original pure policy eligibility remains in `native_status`.

The portable updater includes completed new seasons instead of retaining the original feature builder's fixed 2017–2025 default. It consumes canonical player statistics and schedules, plus the current public nflverse PBP file (reusing the champion's file if fetched within six hours). Original `team_game_table` and `History.features` produce features with the six-hour as-of lag. Historical snapshot records remain immutable; selected ridge base refits on 2019 and later completed games. Before any new game is available, the exact original saved coefficients are preserved.

**Residuals, scale and saved EB corrections remain frozen at launch.** The adapter does not call the source `fit_only`, append retrospective “OOF” predictions, or claim they are true prospective forecasts. This deliberately differs from its original weekly calibration update. Forward observed predictions and outcomes should support a separately reviewed future calibration change. Runtime model/record hashes must match the completed refresh manifest. If a completed post-snapshot game is missing from processed history, challenger scoring stops with `stale_history` rather than silently using old history.

Runtime files in `runtime/` are reproducible deployment artifacts, separate from this source snapshot. Run `python scripts/refresh_challenger.py --refresh-data` after the common data refresh. No model selection or policy search occurs during the daily job.

Historical nflverse game lines are closing context, not archived 6:30 AM context. Historical `xpass` features retain the source model's upstream training-vintage limitation. The saved historical metrics are reported evidence, not proof of future market profitability. Compare prospective same-offer forecasts and actual recorded selections separately.
