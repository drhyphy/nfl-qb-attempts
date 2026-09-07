import json

import pandas as pd
import pytest

from qb_attempts.comparison import grade_comparison, paired_forecasts

KICKOFF = "2026-09-01T17:00:00Z"


def candidate(**changes):
    return {"game_id": "2026_01_A_B", "player_id": "qb1", "player": "Quarterback",
            "team": "A", "side": "Over", "line": 30.5, "book": "a", "kickoff": KICKOFF,
            "p_final": .6, "p_push": 0., "mean": 33., "ev": .2, "status": "held", **changes}


def pair(**changes):
    return paired_forecasts([candidate()], [candidate(p_final=.4, mean=29.)])[0] | changes


def board(tmp_path, name, at="2026-09-01T10:30:00Z", pairs=None):
    (tmp_path / name).write_text(json.dumps({"generated_at": at,
        "comparison": {"paired_forecasts": [pair()] if pairs is None else pairs}}))


def games(**changes):
    return pd.DataFrame([{"game_id": "2026_01_A_B", "kickoff": KICKOFF,
                         "home_score": 20, "away_score": 17, **changes}])


def stats(**changes):
    return pd.DataFrame([{"game_id": "2026_01_A_B", "player_id": "qb1", "attempts": 34, **changes}])


def test_first_shared_snapshot_is_frozen_without_older_champion_only_advantage(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"generated_at": "2026-08-31T10:30Z",
                                                "recommendations": [candidate()]}))
    board(tmp_path, "z-first.json")
    board(tmp_path, "a-later.json", at="2026-09-01T11:00Z", pairs=[pair(
        challenger={"mean": 34., "p_over": .99, "p_push": 0.})])
    board(tmp_path, "empty.json", at="2026-09-01T12:00Z", pairs=[])
    result = grade_comparison(tmp_path, stats(), games())
    assert result["n_total"] == result["n_settled"] == result["n_probability_scored"] == 1
    assert result["results"][0]["snapshot"] == "z-first.json"
    assert result["champion"]["mae"] == 1
    assert result["challenger"]["mae"] == 5
    assert result["champion"]["brier"] == pytest.approx(.16)
    assert result["challenger"]["brier"] == pytest.approx(.36)
    assert result["differences"]["mae"] == 4


@pytest.mark.parametrize("changes", [{"book": "b"}, {"line": 31.5}, {"player_id": "qb2"},
                                     {"game_id": "other"}, {"kickoff": "2026-09-02T17:00Z"}])
def test_nonidentical_offer_is_not_paired(changes):
    assert paired_forecasts([candidate()], [candidate(**changes)]) == []


def test_selection_uses_shared_book_line_not_ev_status_or_input_order():
    candidates = [candidate(book="z", ev=50, status="qualified"),
                  candidate(book="a", line=31.5, ev=5, status="qualified"),
                  candidate(book="a", line=30.5, ev=-5, status="held")]
    result = paired_forecasts(candidates, list(reversed(candidates)))
    assert len(result) == 1
    assert (result[0]["book"], result[0]["line"]) == ("a", 30.5)


def test_complementary_sides_convert_unconditional_mass_and_deduplicate():
    candidates = [candidate(p_final=.54, p_push=.1),
                  candidate(side="Under", p_final=.36, p_push=.1)]
    result = paired_forecasts(candidates, list(reversed(candidates)))
    assert len(result) == 1
    assert result[0]["champion"]["p_over"] == pytest.approx(.6)
    assert result[0]["challenger"]["p_over"] == pytest.approx(.6)


def test_conflicting_distribution_duplicates_are_not_selected():
    assert paired_forecasts([candidate(), candidate(p_final=.9)], [candidate()]) == []


@pytest.mark.parametrize("challenger", [[], [candidate(p_final=None)], [candidate(mean=float("nan"))],
                                        [candidate(p_push=1)], [candidate(side="bad")]])
def test_partial_failure_does_not_create_pair(challenger):
    assert paired_forecasts([candidate()], challenger) == []


@pytest.mark.parametrize("missing", ["stats", "game", "home", "away", "conflict", "duplicate_game", "negative", "fraction"])
def test_missing_or_conflicting_official_data_remains_unresolved(tmp_path, missing):
    board(tmp_path, "a.json")
    s, g = stats(), games()
    if missing == "stats": s = stats(player_id="other")
    if missing == "game": g = games(game_id="other")
    if missing == "home": g = games(home_score=None)
    if missing == "away": g = games().drop(columns="away_score")
    if missing == "conflict": s = pd.concat([s, stats(attempts=35)])
    if missing == "duplicate_game": g = pd.concat([g, g])
    if missing == "negative": s = stats(attempts=-1)
    if missing == "fraction": s = stats(attempts=30.5)
    result = grade_comparison(tmp_path, s, g)
    assert result["n_total"] == 1
    assert result["n_settled"] == result["n_probability_scored"] == 0
    assert result["champion"]["mae"] is None
    assert result["results"][0]["actual"] is None


def test_integer_push_still_scores_mae_but_neither_binary_score(tmp_path):
    board(tmp_path, "a.json", pairs=[pair(line=34)])
    result = grade_comparison(tmp_path, stats(), games())
    assert result["n_total"] == result["n_settled"] == 1
    assert result["n_probability_scored"] == 0
    assert result["champion"]["mae"] == 1
    assert result["champion"]["brier"] is result["challenger"]["log_loss"] is None


def test_actual_zero_is_valid_and_identical_stat_duplicates_are_consistent(tmp_path):
    board(tmp_path, "a.json")
    result = grade_comparison(tmp_path, pd.concat([stats(attempts=0), stats(attempts=0)]), games())
    assert result["n_settled"] == 1
    assert result["results"][0]["actual"] == 0
    assert result["champion"]["brier"] == pytest.approx(.36)


@pytest.mark.parametrize("at,changes", [(KICKOFF, {}), ("2026-09-01T18:00Z", {"kickoff": "2026-09-02T17:00Z"}),
    ("2099-09-01T10:30Z", {"kickoff": "2099-09-02T17:00Z"}),
    ("2026-09-01T10:30Z", {"kickoff": "bad"}), ("2026-09-01T10:30", {})])
def test_postgame_future_or_invalid_timestamps_cannot_enter_record(tmp_path, at, changes):
    board(tmp_path, "a.json", at=at, pairs=[pair(**changes)])
    assert grade_comparison(tmp_path, stats(), games())["n_total"] == 0


def test_partial_pair_in_history_is_skipped_without_blocking_first_valid_pair(tmp_path):
    malformed = pair()
    del malformed["challenger"]
    board(tmp_path, "early.json", pairs=[malformed])
    board(tmp_path, "later.json", at="2026-09-01T11:00Z")
    result = grade_comparison(tmp_path, stats(), games())
    assert result["n_total"] == 1
    assert result["results"][0]["snapshot"] == "later.json"


def test_malformed_file_is_audited_and_no_confidence_claims_are_reported(tmp_path):
    (tmp_path / "bad.json").write_text("broken")
    board(tmp_path, "good.json")
    result = grade_comparison(tmp_path, stats(), games())
    assert len(result["diagnostics"]) == 1
    assert "correlated" in result["uncertainty_note"]
    assert "confidence_interval" not in result


def test_future_game_cannot_settle_even_if_scores_are_present(tmp_path):
    board(tmp_path, "a.json", pairs=[pair(kickoff="2099-09-01T17:00Z")])
    result = grade_comparison(tmp_path, stats(), games(kickoff="2099-09-01T17:00Z"))
    assert result["n_total"] == 1
    assert result["n_settled"] == 0


def test_numeric_strings_are_normalized_before_scoring(tmp_path):
    board(tmp_path, "a.json", pairs=[pair(champion={"mean": "33", "p_over": ".6", "p_push": "0"})])
    result = grade_comparison(tmp_path, stats(), games())
    assert result["champion"]["mae"] == 1
    assert result["champion"]["brier"] == pytest.approx(.16)
