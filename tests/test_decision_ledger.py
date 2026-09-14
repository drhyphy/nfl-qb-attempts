from copy import deepcopy

import pytest

from qb_attempts.decision_ledger import attach_verified_clv, reconcile_ledger, update_ledger


NOW = "2026-09-10T12:00:00Z"
REC = {"game_id": "2026_01_SF_LA", "player_id": "p", "player": "Brock Purdy", "kickoff": "2026-09-11T00:00:00Z",
       "line": 33.5, "odds": -110, "side": "under", "observed_at": "2026-09-10T11:59:00Z", "book": "test"}


def board(rec=None, **changes):
    rec = deepcopy(REC) if rec is None else rec
    return {"generated_at": NOW, "status": "ok", "model_version": "frozen", "recommendations": [rec],
            "watchlist": [], "verified_recommendations": [rec], **changes}


def test_early_entry_without_confirmed_role_and_no_execution():
    ledger = update_ledger(None, {"champion": board()}, NOW)
    assert len(ledger["entries"]) == 1
    assert "role_not_confirmed_at_entry" in ledger["entries"][0]["warnings"]
    assert ledger["actual_wagers"] == [] and ledger["entries"][0]["actual_wager"] is False


def test_research_does_not_silently_enter_verified_cohort():
    ledger = update_ledger(None, {"champion": board(verified_recommendations=[])}, NOW)
    assert not ledger["entries"] and ledger["events"][0]["status"] == "observed"


def test_later_price_does_not_overwrite_entry_and_duplicate_is_idempotent():
    first = update_ledger(None, {"champion": board()}, NOW)
    assert update_ledger(first, {"champion": board()}, NOW) == first
    later = board({**REC, "line": 32.5, "odds": 110}, generated_at="2026-09-10T13:00:00Z")
    second = update_ledger(first, {"champion": later}, "2026-09-10T13:00:00Z")
    assert len(second["entries"]) == 1 and second["entries"][0]["line"] == 33.5
    assert second["entries"][0]["odds"] == -110


def test_one_game_exposure_separate_models_and_persistent_open_cap():
    second = {**REC, "player_id": "opponent", "player": "Other QB"}
    ledger = update_ledger(None, {"champion": board(verified_recommendations=[REC, second]), "claude": board(second)}, NOW)
    assert len(ledger["entries"]) == 2
    later = board({**REC, "game_id": "new_game"}, generated_at="2026-09-10T13:00:00Z")
    ledger = update_ledger(ledger, {"champion": later}, "2026-09-10T13:00:00Z", max_open_per_model=1)
    assert len(ledger["entries"]) == 2
    assert ledger["events"][-1]["reason"] == "open_exposure_cap"


def test_withdrawal_does_not_erase_entry_or_eventual_loss():
    first = update_ledger(None, {"champion": board()}, NOW)
    unavailable = {**REC, "role_evidence": {"reliable": True, "status": "out"}}
    later = update_ledger(first, {"champion": board(unavailable, generated_at="2026-09-10T13:00:00Z")}, "2026-09-10T13:00:00Z")
    assert later["entries"] == first["entries"]
    assert any(e["status"] == "withdrawn" for e in later["events"])
    evidence = {"game_id": REC["game_id"], "final": True, "observed_at": "2026-09-11T05:00:00Z",
                "players": [{"player_id": "p", "attempts": 34, "participated": True}]}
    graded = reconcile_ledger(later, {REC["game_id"]: evidence}, "2026-09-11T06:00:00Z")
    assert graded["entries"][0]["settlement"]["result"] == "loss"
    assert graded["performance"]["champion"]["units"] == -1
    assert graded["entries"][0]["odds"] == -110
    assert graded["entries"][0]["settlement_history"][0]["grade"]["result"] == "loss"


def test_withdrawal_before_entry_does_not_use_capacity_questionable_allows_entry():
    out = {**REC, "role_evidence": {"reliable": True, "status": "out"}}
    ledger = update_ledger(None, {"champion": board(out)}, NOW)
    assert not ledger["entries"]
    questionable = {**REC, "role_evidence": {"reliable": True, "status": "questionable"}}
    assert len(update_ledger(None, {"champion": board(questionable)}, NOW)["entries"]) == 1


@pytest.mark.parametrize("changes", [{"observed_at": "2026-09-10T12:01:00Z"},
    {"source_updated_at": "2026-09-10T12:01:00Z"}, {"kickoff": NOW}])
def test_timestamp_causality(changes):
    assert not update_ledger(None, {"champion": board({**REC, **changes})}, NOW)["entries"]


def test_future_board_and_time_travel_rejected():
    assert not update_ledger(None, {"champion": board(generated_at="2026-09-10T13:00:00Z")}, NOW)["entries"]
    ledger = update_ledger(None, {"champion": board()}, NOW)
    with pytest.raises(ValueError):
        update_ledger(ledger, {"champion": board()}, "2026-09-10T11:00:00Z")


def test_capitalized_side_preserved_and_graded():
    ledger = update_ledger(None, {"champion": board({**REC, "side": "Under"})}, NOW)
    assert ledger["entries"][0]["side"] == "Under"
    evidence = {"game_id": REC["game_id"], "final": True, "observed_at": "2026-09-11T05:00:00Z",
                "players": [{"player_id": "p", "attempts": 34, "participated": True}]}
    assert reconcile_ledger(ledger, {REC["game_id"]: evidence}, "2026-09-11T06:00:00Z")["performance"]["champion"]["losses"] == 1


def test_unverified_closing_quotes_never_create_clv():
    ledger = update_ledger(None, {"champion": board()}, NOW)
    quote = {**REC, "observed_at": "2026-09-10T23:50:00Z", "over_odds": -110, "under_odds": -110}
    assert attach_verified_clv(ledger, [quote], "2026-09-11T06:00:00Z")["entries"][0]["verified_clv"]["value"] is None


def test_verified_clv_revalidates_evidence_and_excludes_withdrawal():
    from qb_attempts.quote_verification import verify_against_offers

    ledger = update_ledger(None, {"champion": board()}, NOW)
    checked = "2026-09-10T23:50:00Z"
    quote = {**REC, "observed_at": checked, "over_odds": -110, "under_odds": -110,
             "event_id": "event", "home_team": "LA", "away_team": "SF"}
    offers = [{**quote, "side": side, "odds": -110, "available": True, "market_id": "market",
               "selection_id": side, "provider_event_id": "event", "jurisdiction": "US-NY",
               "evidence_level": "sportsbook_direct", "source": "direct"} for side in ("over", "under")]
    quote = verify_against_offers([quote], offers, checked)[0]
    assert quote["quote_verification"]["status"] == "verified"
    result = attach_verified_clv(ledger, [quote], "2026-09-11T06:00:00Z")
    assert result["entries"][0]["verified_clv"]["books"] == 1
    ledger["events"].append({"model": "champion", "game_id": REC["game_id"], "player_id": "p",
                             "status": "withdrawn", "at": "2026-09-10T23:00:00Z"})
    assert attach_verified_clv(ledger, [quote], "2026-09-11T06:00:00Z")["entries"][0]["verified_clv"]["value"] is None
