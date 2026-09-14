import copy
import json

import numpy as np
import pandas as pd

from qb_attempts.shadow import VERSION, conditional_probabilities, fit_weight, prepare_archive, score_shadow, train_shadow


def history():
    return pd.DataFrame({"season": [2021]*250, "candidate": ["market_ridge"]*250,
        "residual": np.linspace(-20, 20, 250), "prediction": [32.]*250,
        "expected_margin": [0.]*250, "game_total": [44.]*250})


def artifact(tmp_path):
    value = {"version": VERSION, "generated_at": "2026-09-14T00:00:00Z", "status": "ok", "weight": .4,
             "residual_history": history().drop(columns="candidate").to_dict("records"), "metrics": {}}
    path = tmp_path/"artifact.json"
    path.write_text(json.dumps(value))
    return path, value


def candidate(**updates):
    return {"game_id": "g", "player_id": "q", "player": "QB", "book": "a", "line": 32.5,
        "kickoff": "2026-09-15T00:00:00Z", "mean": 32., "expected_margin": 0., "game_total": 44.,
        "side": "Over", "p_push": 0., "p_model": .6, "p_market": .5, "ev": .3, **updates}


def test_shadow_is_independent_of_ev_and_does_not_mutate_candidates(tmp_path):
    path, _ = artifact(tmp_path)
    rows = [candidate(book="z", ev=1.), candidate(book="a", ev=-1.)]
    original = copy.deepcopy(rows)
    result = score_shadow(rows, "2026-09-14T12:00:00Z", path)
    assert result["status"] == "ok"
    assert result["promotion_allowed"] is False
    assert result["predictions"][0]["book"] == "a"
    assert np.isclose(result["predictions"][0]["calibrated_p_over"], .54)
    assert rows == original


def test_missing_future_and_contaminated_artifacts_fail_closed(tmp_path):
    path, value = artifact(tmp_path)
    assert score_shadow([], "2026-09-13T00:00:00Z", path)["status"] == "artifact_unavailable"
    value["residual_history"][0]["season"] = 2026
    path.write_text(json.dumps(value))
    assert score_shadow([], "2026-09-14T12:00:00Z", path)["status"] == "artifact_unavailable"
    assert score_shadow([], "2026-09-14T12:00:00Z", tmp_path/"missing")["status"] == "artifact_unavailable"


def test_started_games_excluded_without_role_confirmation_requirement(tmp_path):
    path, _ = artifact(tmp_path)
    rows = [candidate(kickoff="2026-09-14T10:00:00Z"), candidate(player_id="future")]
    result = score_shadow(rows, "2026-09-14T12:00:00Z", path)
    assert [r["player_id"] for r in result["predictions"]] == ["future"]


def test_closed_form_weight_and_conditional_distribution():
    frame = pd.DataFrame({"p_model": [.9, .1], "p_market": [.5, .5], "target": [1, 0]})
    assert fit_weight(frame) == 1.
    po, pu, pp = conditional_probabilities(32, 32., 0, 44, history())
    assert all(0 <= p <= 1 for p in [po, pu, pp])
    assert np.isclose(po+pu+pp, 1.) and pp > 0


def test_archive_rejects_playoff_week_collision_and_postkickoff_quotes():
    features = pd.DataFrame([{"kickoff": "2022-09-13T00:15:00Z", "player": "Geno Smith", "team": "SEA",
                             "opponent": "DEN", "game_id": "2022_01_DEN_SEA", "player_id": "q", "season": 2022}])
    oof = pd.DataFrame([{"game_id": "2022_01_DEN_SEA", "player_id": "q", "candidate": "market_ridge",
                        "attempts": 28, "prediction": 30, "expected_margin": -6, "game_total": 44}])
    offers = []
    for event, date, home, visitor, updated in [(1, "2023-01-14 21:30:00", "SF", "SEA", "2023-01-14 21:29:00"),
        (2, "2022-09-13 00:15:00", "SEA", "DEN", "2022-09-13 00:14:00"),
        (3, "2022-09-13 00:15:00", "SEA", "DEN", "2022-09-13 00:16:00")]:
        for book in ["fanduel", "draftkings"]:
            for side in ["over", "under"]:
                offers.append(dict(event_id=event, scheduled=date, updated=updated, home=home, visitor=visitor,
                    player="Geno Smith", season=2022, book=book, side=side, line=32.5, cost=-110))
    joined, diagnostics = prepare_archive(pd.DataFrame(offers), features, oof, "2026-09-14T00:00:00Z")
    assert len(joined) == 1 and joined.iloc[0].event_id == 2
    assert diagnostics["excluded_unmatched_pairs"] == 2


def test_weight_uses_only_fixed_fit_seasons_and_excludes_future():
    rows = []
    for year, count in [(2022, 210), (2023, 10), (2024, 110), (2025, 10), (2026, 20)]:
        for i in range(count):
            rows.append(dict(season=year, kickoff=f"{year}-10-01T12:00:00Z", quote_at_over=f"{year}-10-01T11:00:00Z",
                quote_at_under=f"{year}-10-01T11:00:00Z", attempts=40 if i%2 else 20, prediction=34.,
                expected_margin=0., game_total=44., line=32.5, p_market=.5))
    data = pd.DataFrame(rows)
    first, scores = train_shadow(data, history(), "2026-09-14T00:00:00Z")
    data.loc[data.season >= 2024, "attempts"] = 100
    second, _ = train_shadow(data, history(), "2026-09-14T00:00:00Z")
    assert first["status"] == "ok" and first["weight"] == second["weight"]
    assert scores.season.max() == 2025
    assert first["live_results_used_for_training"] is False
    insufficient, _ = train_shadow(data.iloc[:5], history(), "2026-09-14T00:00:00Z")
    assert insufficient["status"] == "insufficient_data"
