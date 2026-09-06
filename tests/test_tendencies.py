import numpy as np
import pandas as pd
import pytest

from qb_attempts.tendencies import aggregate_team_games


def play(**overrides):
    row = dict(game_id="2025_01_BUF_NYJ", season=2025, week=1, posteam="BUF", defteam="NYJ",
               game_date="2025-09-07", play_type="pass", qb_dropback=1, pass_attempt=1,
               rush_attempt=0, sack=0, qb_scramble=0, qb_kneel=0, qb_spike=0,
               two_point_attempt=0, score_differential=0, game_seconds_remaining=3000,
               xpass=0.6, season_type="REG")
    row.update(overrides)
    return row


def test_sack_and_scramble_are_dropbacks_not_attempts():
    pbp = pd.DataFrame([play(), play(sack=1), play(play_type="run", qb_scramble=1, rush_attempt=1),
                        play(play_type="run", qb_dropback=0, pass_attempt=0, rush_attempt=1)])
    row = aggregate_team_games(pbp).iloc[0]
    assert row.plays == 4
    assert row.dropbacks == 3
    assert row.pass_attempts == 1
    assert row.dropback_rate == 0.75
    assert row.attempts_per_dropback == pytest.approx(1 / 3)
    assert row.dropback_oe == pytest.approx(0.15)


def test_special_plays_and_preseason_do_not_distort_preference():
    pbp = pd.DataFrame([play(), play(qb_kneel=1), play(qb_spike=1), play(two_point_attempt=1),
                        play(play_type="no_play"), play(season_type="PRE"), play(play_type="punt")])
    assert aggregate_team_games(pbp).iloc[0].plays == 1


def test_situation_boundaries_are_preplay_and_missing_is_not_neutral():
    pbp = pd.DataFrame([play(score_differential=7), play(score_differential=-7),
                        play(score_differential=8), play(score_differential=-8),
                        play(game_seconds_remaining=120), play(game_seconds_remaining=121),
                        play(score_differential=np.nan)])
    row = aggregate_team_games(pbp).iloc[0]
    assert row.neutral_plays == 3
    assert row.lead_plays == 1
    assert row.trail_plays == 1
    assert row.plays == 7


def test_missing_xpass_preserves_empirical_neutral_tendency():
    row = aggregate_team_games(pd.DataFrame([play()]).drop(columns="xpass")).iloc[0]
    assert np.isnan(row.dropback_oe)
    assert row.xpass_plays == 0
    assert row.neutral_dropback_rate == 1
    assert np.isnan(row.lead_dropback_rate)


def test_bad_xpass_is_excluded_from_denominator():
    row = aggregate_team_games(pd.DataFrame([play(xpass=0.5), play(xpass=3), play(xpass=np.nan)])).iloc[0]
    assert row.xpass_plays == 1
    assert row.dropback_oe == 0.5


def test_team_separation_and_historical_aliases():
    data = pd.DataFrame([play(posteam="OAK", defteam="SD"), play(posteam="SD", defteam="OAK")])
    rows = aggregate_team_games(data)
    assert set(rows.team) == {"LV", "LAC"}
    assert len(rows) == 2


def test_duplicate_plays_fail_loudly():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_team_games(pd.DataFrame([play(play_id=1), play(play_id=1)]))


def test_missing_required_columns_fail_loudly():
    with pytest.raises(ValueError, match="missing required"):
        aggregate_team_games(pd.DataFrame([play()]).drop(columns="qb_scramble"))


def test_empty_offensive_sample_preserves_schema():
    rows = aggregate_team_games(pd.DataFrame([play(qb_spike=1)]))
    assert rows.empty
    assert "neutral_dropback_rate" in rows
