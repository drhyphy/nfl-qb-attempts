"""Behavioral checks for the causal forecast and its evaluation boundary."""
import numpy as np
import pandas as pd
import pytest

from qb_attempts import features, model


def record(game="g1", player="p1", team="A", opponent="B", kickoff="2024-09-01T17:00Z", **changes):
    value = dict(game_id=game, season=2024, week=1, kickoff=pd.Timestamp(kickoff),
                 player_id=player, player=player, team=team, opponent=opponent,
                 home=1, rest=7., coach="coach_" + team, opp_coach="coach_" + opponent,
                 dome=0, attempts=30., team_attempts=35., plays=65., sack_rate=.06,
                 qb_carries=3.)
    return {**value, **changes}


def test_own_outcome_and_simultaneous_outcomes_cannot_change_features():
    previous = record(game="prior", kickoff="2024-08-25T17:00Z")
    target = record(game="target", kickoff="2024-09-01T17:00Z", attempts=1000.)
    concurrent = record(game="other", player="p2", team="C", opponent="B",
                        kickoff="2024-09-01T17:00Z", attempts=2000., team_attempts=2000.)
    future = record(game="future", kickoff="2024-09-08T17:00Z", attempts=3000.)
    expected = features.History(pd.DataFrame([previous])).features([target])
    actual = features.History(pd.DataFrame([future, concurrent, target, previous])).features([target])
    np.testing.assert_array_equal(actual[features.FEATURES], expected[features.FEATURES])


def test_results_become_available_only_after_conservative_six_hour_lag():
    earlier = record(attempts=5., team_attempts=5.)
    boundary = record(game="next", kickoff="2024-09-01T23:00Z")
    later = {**boundary, "kickoff": boundary["kickoff"] + pd.Timedelta(seconds=1)}
    out = features.History(pd.DataFrame([earlier])).features([boundary, later])
    assert out.iloc[0].qb_games == 0
    assert out.iloc[1].qb_games == 1


def test_explicit_morning_asof_excludes_results_before_kickoff_but_after_decision():
    prior = record(game="prior", kickoff="2024-08-25T17:00Z", attempts=20.)
    afternoon = record(game="afternoon", kickoff="2024-09-01T17:00Z", attempts=60.)
    target = record(game="night", kickoff="2024-09-02T00:20Z",
                    as_of=pd.Timestamp("2024-09-01T10:30Z"))
    actual = features.History(pd.DataFrame([prior, afternoon])).features([target]).iloc[0]
    assert actual.qb_games == 1
    # Sorting must follow information cutoff even when requests have reverse kickoff order.
    earlier_kick = record(game="earlier", kickoff="2024-09-01T23:30Z",
                         as_of=pd.Timestamp("2024-09-01T23:20Z"))
    combined = features.History(pd.DataFrame([prior, afternoon])).features([earlier_kick, target]).set_index("game_id")
    assert combined.loc["night", "qb_games"] == 1
    assert combined.loc["earlier", "qb_games"] == 2


def test_opponent_features_measure_allowed_volume_not_opponents_offense():
    against_b = record(game="against_b", team="C", player="c", opponent="B", team_attempts=48., plays=75.)
    b_offense = record(game="b_offense", team="B", player="b", opponent="D", team_attempts=12., plays=51.)
    target = record(game="target", kickoff="2024-09-08T17:00Z")
    out = features.History(pd.DataFrame([against_b, b_offense])).features([target]).iloc[0]
    assert out.opp_attempts5 == 48.
    assert out.opp_plays5 == 75.
    assert out.team_attempts5 == 33.  # A has no prior observed team game.


def test_traded_qb_keeps_player_history_and_uses_current_team_context():
    qb_old = record(team="OLD", attempts=20., team_attempts=25.)
    new_team = record(game="new_prior", team="NEW", player="new_old_qb", team_attempts=45., coach="new_coach")
    target = record(game="target", team="NEW", kickoff="2025-09-07T17:00Z", season=2025, coach="next_coach")
    out = features.History(pd.DataFrame([qb_old, new_team])).features([target]).iloc[0]
    assert out.qb_games == 1
    assert out.qb_team_change == 1
    assert out.team_attempts5 == 45
    assert out.coach_change == 1
    assert out.season_games == 0
    assert out.team_coach_games == 0


def test_game_records_preserve_listed_starter_with_early_exit_and_missing_stat_row():
    games = pd.DataFrame([dict(game_id="g", season=2024, week=1,
        kickoff=pd.Timestamp("2024-09-01T17:00Z"), home_team="A", away_team="B",
        home_score=20, away_score=10, home_qb_id="starter", away_qb_id="no_stats",
        home_qb_name="Starter", away_qb_name="Early Exit", home_rest=7, away_rest=7,
        home_coach="A", away_coach="B", location="Home", roof="outdoors")])
    stats = pd.DataFrame([
        dict(game_id="g", team="A", player_id="starter", attempts=1, carries=0, sacks_suffered=1),
        dict(game_id="g", team="A", player_id="backup", attempts=39, carries=3, sacks_suffered=2),
        dict(game_id="g", team="B", player_id="other", attempts=25, carries=20, sacks_suffered=4),
    ])
    rows = features.game_records(stats, games).set_index("team")
    assert rows.loc["A", "player_id"] == "starter"
    assert rows.loc["A", "attempts"] == 1
    assert rows.loc["A", "team_attempts"] == 40
    assert rows.loc["A", "plays"] == 46  # Attempts + carries + sacks, not dropbacks alone.
    assert rows.loc["B", "attempts"] == 0
    assert len(rows) == 2


def test_integer_push_probability_matches_adjacent_half_lines():
    residual = np.array([-30., -8., -2., 0., 3., 9.])
    o, u, push = model.probabilities(30, residual, 30)
    below = model.probabilities(30, residual, 29.5)
    above = model.probabilities(30, residual, 30.5)
    assert u == pytest.approx(below[1])
    assert o == pytest.approx(above[0])
    assert push == pytest.approx(above[1] - below[1])
    assert push > 0
    assert o + u + push == pytest.approx(1.)
    assert below[2] == pytest.approx(0.)
    assert above[2] == pytest.approx(0.)


def test_probability_mass_is_nonnegative_and_monotone_including_early_exit_tail():
    residual = np.array([-80., -30., -5., 0., 6.])
    results = [model.probabilities(30, residual, line) for line in (-.5, 0., .5, 20., 30., 40., 100.)]
    for result in results:
        assert all(0 <= x <= 1 for x in result)
        assert sum(result) == pytest.approx(1.)
    assert np.all(np.diff([r[0] for r in results]) <= 0)
    assert np.all(np.diff([r[1] for r in results]) >= 0)
    assert results[0] == pytest.approx((1., 0., 0.))
    assert results[1][2] > .3  # All negative latent support is censored to zero attempts.


def test_season_fits_and_residual_calibration_never_use_evaluation_labels(monkeypatch, tmp_path):
    # Tiny deterministic candidates isolate the evaluation protocol from estimator fit quality.
    rows = []
    for season in range(2018, 2026):
        for idx in range(2):
            row = record(game=f"{season}-{idx}", player=f"p{idx}", season=season,
                         attempts=float(20 + season - 2018), week=idx + 1)
            row.update({feature: 1. for feature in features.FEATURES})
            row.update(week=idx + 1, qb_mean5=24.)
            rows.append(row)
    df = pd.DataFrame(rows)
    fits = []
    seen_crps = []
    seen_probs = []
    original_crps = model.crps
    original_probs = model.probabilities
    monkeypatch.setattr(model, "CANDIDATES", ("recent_mean", "ridge"))
    monkeypatch.setattr(model, "fit", lambda name, data: fits.append((name, tuple(sorted(data.season.unique())))))
    monkeypatch.setattr(model, "predict", lambda name, obj, data: np.repeat(24. if name == "recent_mean" else 35., len(data)))
    def capture_crps(mu, residuals, y):
        seen_crps.append((mu, np.asarray(residuals).copy(), y))
        return original_crps(mu, residuals, y)
    def capture_probs(mu, residuals, line):
        seen_probs.append(np.asarray(residuals).copy())
        return original_probs(mu, residuals, line)
    monkeypatch.setattr(model, "crps", capture_crps)
    monkeypatch.setattr(model, "probabilities", capture_probs)
    artifact = model.train_validate(df, tmp_path / "base")
    for candidate_index, name in enumerate(model.CANDIDATES):
        for offset, test_season in enumerate(range(2019, 2026)):
            assert fits[candidate_index * 7 + offset] == (name, tuple(range(2018, test_season)))
        for offset, test_season in enumerate(range(2022, 2025)):
            for item in seen_crps[candidate_index * 6 + offset * 2:candidate_index * 6 + offset * 2 + 2]:
                prediction = 24. if name == "recent_mean" else 35.
                expected = np.repeat([20 + season - 2018 - prediction for season in range(2019, test_season)], 2)
                np.testing.assert_array_equal(item[1], expected)
    selected_prediction = 24. if artifact["name"] == "recent_mean" else 35.
    expected_holdout_residuals = np.repeat([20 + season - 2018 - selected_prediction for season in range(2019, 2025)], 2)
    for residuals in seen_probs:
        np.testing.assert_array_equal(residuals, expected_holdout_residuals)
    expected_live_residuals = np.repeat([20 + season - 2018 - selected_prediction for season in range(2019, 2026)], 2)
    np.testing.assert_array_equal(artifact["residuals"], expected_live_residuals)
    assert fits[-1][1] == tuple(range(2018, 2026))
    modified = df.copy()
    modified.loc[modified.season == 2025, "attempts"] = 75.
    other = model.train_validate(modified, tmp_path / "changed_holdout")
    assert other["name"] == artifact["name"]
    assert other["metrics"]["candidates"] == artifact["metrics"]["candidates"]
    assert other["metrics"]["holdout_2025"]["mae"] != artifact["metrics"]["holdout_2025"]["mae"]
    assert artifact["metrics"]["market_edge_validated"] is False
    assert artifact["metrics"]["historical_market_roi"] is None
