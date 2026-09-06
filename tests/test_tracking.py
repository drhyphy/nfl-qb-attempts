import json

import pandas as pd
import pytest

from qb_attempts.tracking import grade_history


KICKOFF = "2026-09-13T17:00:00Z"


def rec(**changes):
    return {"game_id": "2026_01_A_B", "player_id": "qb1", "player": "Quarterback",
            "side": "over", "line": 30.5, "odds": 110, "book": "a", "kickoff": KICKOFF,
            "p_final": .56, "p_push": 0., "ev": .176, **changes}


def board(tmp_path, name, at="2026-09-13T10:30:00Z", recommendations=None, version="v1"):
    (tmp_path / name).write_text(json.dumps({"generated_at": at, "model_version": version,
                                            "recommendations": [rec()] if recommendations is None else recommendations}))


def games(**changes):
    return pd.DataFrame([{"game_id": "2026_01_A_B", "kickoff": KICKOFF,
                          "home_score": 20, "away_score": 17, **changes}])


def stats(**changes):
    return pd.DataFrame([{"game_id": "2026_01_A_B", "player_id": "qb1", "attempts": 34, **changes}])


def quote(**changes):
    return {"game_id": "2026_01_A_B", "player_id": "qb1", "line": 30.5, "book": "a",
            "over_odds": -110, "under_odds": -110, "observed_at": "2026-09-13T16:30:00Z", **changes}


def test_first_chronological_publication_is_frozen_and_empty_board_cannot_erase(tmp_path):
    board(tmp_path, "z-first.json")
    board(tmp_path, "a-later.json", at="2026-09-13T11:00Z", recommendations=[rec(side="under", odds=140)], version="v2")
    board(tmp_path, "empty.json", at="2026-09-13T12:00Z", recommendations=[])
    result = grade_history(tmp_path, stats(), games())
    assert (result["bets"], result["wins"], result["settled"]) == (1, 1, 1)
    assert result["units"] == pytest.approx(1.1)
    assert result["results"][0]["snapshot"] == "z-first.json"
    assert result["results"][0]["side"] == "over"
    assert list(result["by_model_version"]) == ["v1"]


def test_at_kickoff_late_or_falsely_future_kickoff_are_excluded(tmp_path):
    board(tmp_path, "late.json", at=KICKOFF)
    board(tmp_path, "later.json", at="2026-09-13T18:00Z", recommendations=[rec(kickoff="2026-09-14T17:00Z")])
    result = grade_history(tmp_path, stats(), games())
    assert result["bets"] == result["settled"] == 0
    assert result["roi"] is None and result["mean_clv"] is None


@pytest.mark.parametrize("actual,side,line,odds,status,profit", [
    (30, "over", 30., -150, "push", 0),
    (29, "under", 30.5, -150, "win", 2/3),
    (31, "under", 30.5, 120, "loss", -1),
    (0, "under", 30.5, -110, "win", 100/110),
])
def test_discrete_settlement_and_actual_zero(tmp_path, actual, side, line, odds, status, profit):
    board(tmp_path, "a.json", recommendations=[rec(side=side, line=line, odds=odds)])
    result = grade_history(tmp_path, stats(attempts=actual), games())
    assert result["results"][0]["result"] == status
    assert result["units"] == pytest.approx(profit)
    assert result["roi"] == pytest.approx(profit)


@pytest.mark.parametrize("missing", ["stats", "game", "home", "away", "conflict"])
def test_unresolved_does_not_become_zero_or_a_loss(tmp_path, missing):
    board(tmp_path, "a.json")
    s, g = stats(), games()
    if missing == "stats": s = stats(player_id="someone_else")
    if missing == "game": g = games(game_id="another_game")
    if missing == "home": g = games(home_score=None)
    if missing == "away": g = games(away_score=None)
    if missing == "conflict": s = pd.concat([s, stats(attempts=35)])
    result = grade_history(tmp_path, s, g)
    assert result["bets"] == result["unresolved"] == 1
    assert result["settled"] == result["units"] == 0
    assert result["roi"] is None
    assert result["results"][0]["actual"] is None


def test_clv_selects_latest_per_book_then_median_and_uses_bet_price(tmp_path):
    board(tmp_path, "a.json")
    quotes = [quote(over_odds=-200, under_odds=150, observed_at="2026-09-13T16:10Z"),
              quote(), quote(book="b", over_odds=-150, under_odds=130)]
    result = grade_history(tmp_path, stats(), games(), quotes)
    fair_b = (1 / (1 + 100/150)) / (1 / (1 + 100/150) + 1/2.3)
    p = (.5 + fair_b) / 2
    assert result["results"][0]["closing_probability"] == pytest.approx(p)
    assert result["mean_clv"] == pytest.approx(p * 2.1 - 1)
    assert result["results"][0]["closing_books"] == 2


@pytest.mark.parametrize("changes", [
    {"game_id": "prior_game"}, {"player_id": "other_qb"}, {"line": 31.5},
    {"observed_at": "2026-09-13T10:30Z"}, {"observed_at": "2026-09-13T17:00:01Z"},
    {"observed_at": "2026-09-13T16:30:00"}, {"under_odds": None},
    {"source_updated_at": "2026-09-12T16:30Z"}, {"is_live": True},
])
def test_clv_missing_for_wrong_event_stale_unpaired_or_live_quote(tmp_path, changes):
    board(tmp_path, "a.json")
    result = grade_history(tmp_path, stats(), games(), [quote(**changes)])
    assert result["clv_bets"] == 0 and result["mean_clv"] is None


def test_clv_inclusive_last_hour_boundaries_and_under_direction(tmp_path):
    board(tmp_path, "a.json", recommendations=[rec(side="under")])
    result = grade_history(tmp_path, stats(), games(), [quote(observed_at="2026-09-13T16:00Z"), quote(book="b", observed_at=KICKOFF)])
    assert result["results"][0]["closing_books"] == 2
    assert result["mean_clv"] == pytest.approx(.05)


def test_malformed_snapshot_diagnostic_does_not_mutate_history(tmp_path):
    board(tmp_path, "good.json")
    (tmp_path / "bad.json").write_text("broken")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = grade_history(tmp_path, stats(), games())
    assert len(result["diagnostics"]) == 1 and result["bets"] == 1
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_separate_game_same_player_counts_as_new_bet(tmp_path):
    board(tmp_path, "a.json", recommendations=[rec(), rec(game_id="next_game")])
    result = grade_history(tmp_path, stats(), games())
    assert result["bets"] == 2 and result["settled"] == 1
