"""V2 checks that distinguish pregame knowledge from realized game script."""
import numpy as np
import pandas as pd
import pytest

from qb_attempts import features, game_features, game_model, model


def record(game="prior", kickoff="2024-09-01T17:00Z", **changes):
    row = dict(game_id=game, season=2024, week=1, kickoff=pd.Timestamp(kickoff),
               player_id="p1", player="QB", team="A", opponent="B", home=1,
               rest=7., coach="A coach", opp_coach="B coach", dome=0,
               attempts=30., team_attempts=35., plays=65., sack_rate=.06,
               qb_carries=3., expected_margin=3., game_total=45., pbp_plays=60.,
               neutral_plays=30., neutral_dropbacks=18., lead_plays=20.,
               lead_dropbacks=8., trail_plays=10., trail_dropbacks=8.,
               dropbacks=34., pass_attempts=30.)
    return {**row, **changes}


def test_added_tendencies_preserve_v1_features_and_exclude_current_future_results():
    prior = record()
    target = record("target", "2024-09-08T17:00Z")
    future = record("future", "2024-09-15T17:00Z", attempts=999.,
                    neutral_dropbacks=999., lead_dropbacks=999., trail_dropbacks=999.)
    history = pd.DataFrame([prior, target, future])
    actual = game_features.game_features(history, [target])
    expected = game_features.game_features(pd.DataFrame([prior]), [target])
    np.testing.assert_array_equal(actual[game_features.TENDENCY_FEATURES],
                                  expected[game_features.TENDENCY_FEATURES])
    old = features.History(history).features([target])
    np.testing.assert_array_equal(actual[features.FEATURES], old[features.FEATURES])
    assert actual.iloc[0].tendency_games == 1
    assert actual.iloc[0].lead_pass12 == pytest.approx((8 + 120 * .5) / 140)


def test_tendencies_use_morning_asof_and_strict_six_hour_availability():
    prior = record()
    target = record("target", "2024-09-02T00:20Z", as_of=pd.Timestamp("2024-09-01T10:30Z"))
    boundary = record("boundary", "2024-09-01T23:00Z")
    later = record("later", "2024-09-01T23:00:01Z")
    frame = game_features.game_features(pd.DataFrame([prior]), [later, target, boundary]).set_index("game_id")
    assert frame.loc["target", "tendency_games"] == 0
    assert frame.loc["boundary", "tendency_games"] == 0
    assert frame.loc["later", "tendency_games"] == 1


def test_opponent_tendency_is_allowed_pass_preference_and_ignores_missing_pbp():
    against_b = record("against", team="C", player_id="c", neutral_plays=100., neutral_dropbacks=90.)
    b_offense = record("offense", team="B", opponent="D", player_id="b", neutral_plays=100., neutral_dropbacks=10.)
    unavailable = record("missing", pbp_plays=np.nan, neutral_dropbacks=10000.)
    target = record("target", "2024-09-08T17:00Z")
    row = game_features.game_features(pd.DataFrame([against_b, b_offense, unavailable]), [target]).iloc[0]
    assert row.opponent_neutral_pass12 == pytest.approx((90 + 72) / 220)
    assert row.tendency_games == 0


def test_spread_is_expected_team_margin_not_american_handicap(monkeypatch):
    records = pd.DataFrame([record(team="A"), record(team="B", player_id="p2")])
    monkeypatch.setattr(game_features, "game_records", lambda stats, games: records.copy())
    games = pd.DataFrame([dict(game_id="prior", home_team="A", spread_line=6.5, total_line=47.)])
    result = game_features.records_with_context(None, games, None).set_index("team")
    assert result.loc["A", "expected_margin"] == 6.5
    assert result.loc["B", "expected_margin"] == -6.5
    assert (result.game_total == 47.).all()


def fit_frame():
    rng = np.random.default_rng(730)
    rows = []
    for index in range(90):
        row = record(game=str(index), expected_margin=float(index % 3 - 1) * 10.)
        row.update({name: float(rng.normal()) for name in features.FEATURES})
        row.update(home=index % 2, game_total=45. + index % 5,
                   attempts=32. - .3 * row["expected_margin"] + rng.normal(),
                   neutral_pass12=.6, lead_pass12=.45, trail_pass12=.75,
                   attempts_per_dropback12=.9, opponent_neutral_pass12=.6,
                   tendency_games=12., margin_abs=abs(row["expected_margin"]),
                   margin_x_team_rate=row["expected_margin"] * (row["team_rate5"] - .5),
                   margin_x_qb_carries=row["expected_margin"] * row["qb_carries5"],
                   state_pass_spread_interaction=.3 * row["expected_margin"])
        # Known synthetic monotone exposure, with nonzero probability for every state.
        row.update(lead_plays=45. if row["expected_margin"] > 0 else 5.,
                   neutral_plays=30., trail_plays=45. if row["expected_margin"] < 0 else 5.)
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_v1_fit_and_forecast_are_identical():
    frame = fit_frame()
    original = model.fit("ridge", frame)
    candidate = game_model.fit_candidate("v1_ridge", frame)
    np.testing.assert_array_equal(candidate.predict(frame), model.predict("ridge", original, frame))


def test_script_shares_are_simplex_directional_and_ignore_realized_target_state():
    frame = fit_frame()
    exposure = game_model.ScriptExposure().fit(frame)
    target = pd.concat([frame.iloc[[0]]] * 3, ignore_index=True)
    target["expected_margin"] = [-10., 0., 10.]
    shares = exposure.predict(target)
    np.testing.assert_allclose(shares.sum(axis=1), 1.)
    assert (shares >= 0).all() and (shares <= 1).all()
    assert shares[0, 0] < shares[1, 0] < shares[2, 0]
    assert shares[0, 2] > shares[1, 2] > shares[2, 2]
    mutated = target.assign(lead_plays=1e6, neutral_plays=0., trail_plays=0., attempts=999.)
    np.testing.assert_array_equal(shares, exposure.predict(mutated))


@pytest.mark.parametrize("name", ["market_ridge", "tendency_ridge", "script_opportunity"])
def test_forecast_does_not_read_realized_game_labels_or_xpass(name):
    frame = fit_frame()
    candidate = game_model.fit_candidate(name, frame)
    target = frame.iloc[:3].copy()
    expected = candidate.predict(target)
    target["attempts"] = 999.
    target["plays"] = 999.
    target["lead_plays"] = 999.
    target["neutral_plays"] = 0.
    target["trail_plays"] = 0.
    target["xpass"] = 999.
    target["dropback_oe"] = 999.
    np.testing.assert_array_equal(expected, candidate.predict(target))
    assert not {"xpass", "dropback_oe", "pass_oe"}.intersection(candidate.columns)


def test_missing_game_market_cannot_silently_become_pickem():
    frame = fit_frame()
    candidate = game_model.fit_candidate("market_ridge", frame)
    with pytest.raises((ValueError, KeyError)):
        candidate.predict(frame.drop(columns="expected_margin"))
    with pytest.raises(ValueError):
        candidate.predict(frame.assign(expected_margin=np.nan))


def test_complexity_requires_predeclared_crps_improvement():
    scores = {name: {"crps": 5.} for name in game_model.CANDIDATES}
    scores["script_opportunity"]["crps"] = 4.98
    assert game_model.choose_candidate(scores) == "market_ridge"
    scores["script_opportunity"]["crps"] = 4.97
    assert game_model.choose_candidate(scores) == "script_opportunity"


class ConstantCandidate:
    columns = game_features.MARKET_FEATURES

    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.repeat(self.value, len(frame))


def test_selection_and_distribution_calibration_exclude_2025_labels(monkeypatch, tmp_path):
    rows = []
    for year in range(2018, 2026):
        for idx in range(2):
            rows.append(record(game=f"{year}-{idx}", season=year, player_id=f"p{idx}",
                               attempts=float(20 + year - 2018), tendency_games=2.))
    frame = pd.DataFrame(rows)
    fits, evaluations = [], []
    original_scores = game_model.distribution_scores

    def fit(name, data):
        fits.append((name, tuple(sorted(data.season.unique()))))
        return ConstantCandidate(25. if name == "market_ridge" else 35.)

    def scores(data, residuals):
        evaluations.append((tuple(sorted(data.season.unique())), data.prediction.iloc[0], residuals.copy()))
        return original_scores(data, residuals)

    monkeypatch.setattr(game_model, "fit_candidate", fit)
    monkeypatch.setattr(game_model, "distribution_scores", scores)
    artifact = game_model.train_validate_v2(frame, tmp_path / "first")
    assert artifact["metrics"]["candidates"]["market_ridge"]["development_2022_2024"]["rmse"] == pytest.approx(np.sqrt(2 / 3))
    for candidate_index, name in enumerate(game_model.CANDIDATES):
        for offset, year in enumerate(range(2019, 2026)):
            assert fits[candidate_index * 7 + offset] == (name, tuple(range(2018, year)))
    for seasons, prediction, residuals in evaluations:
        assert len(seasons) == 1
        year = seasons[0]
        expected = np.repeat([20 + prior - 2018 - prediction for prior in range(2019, year)], 2)
        np.testing.assert_array_equal(residuals, expected)
    assert fits[-1][1] == tuple(range(2018, 2026))
    changed = frame.copy()
    changed.loc[changed.season == 2025, "attempts"] = 75.
    other = game_model.train_validate_v2(changed, tmp_path / "changed")
    assert artifact["name"] == other["name"]
    assert artifact["metrics"]["candidates"] == other["metrics"]["candidates"]
    assert artifact["metrics"]["holdout_2025"]["mae"] != other["metrics"]["holdout_2025"]["mae"]
    assert artifact["metrics"]["market_edge_validated"] is False
    assert artifact["metrics"]["historical_market_roi"] is None


def test_vectorized_distribution_scores_match_existing_discrete_crps():
    frame = pd.DataFrame(dict(attempts=[0., 30., 45.], prediction=[15., 30., 38.]))
    residuals = np.array([-40., -8., 0., 3., 15.])
    summary, scores = game_model.distribution_scores(frame, residuals)
    expected = [model.crps(row.prediction, residuals, row.attempts) for row in frame.itertuples()]
    np.testing.assert_allclose(scores, expected)
    assert summary["crps"] == pytest.approx(np.mean(expected))
