from copy import deepcopy

from qb_attempts.settlement import augment_results, normalize_espn_summary, settle_entry


ENTRY = {"game_id": "2026_01_SF_LA", "player_id": "p", "player": "Brock Purdy",
         "entry_at": "2026-09-10T12:00:00Z", "line": 33.5, "odds": -110, "side": "under"}


def evidence(**changes):
    return {"game_id": ENTRY["game_id"], "observed_at": "2026-09-11T05:00:00Z", "final": True,
            "players": [{"player": "Brock Purdy", "attempts": 34, "participated": True}], **changes}


def test_final_and_participation_required():
    assert settle_entry(ENTRY, evidence(final=False))["result"] == "pending"
    assert settle_entry(ENTRY, evidence(players=[]))["result"] == "pending"
    assert settle_entry(ENTRY, evidence(players=[{"player": "Brock Purdy", "attempts": 0}]))["result"] == "pending"
    assert settle_entry(ENTRY, evidence())["result"] == "loss"


def test_confirmed_dnp_never_win_under_zero():
    row = {"player": "Brock Purdy", "attempts": 0, "participated": False, "confirmed_dnp": True}
    assert settle_entry(ENTRY, evidence(players=[row]))["result"] == "pending"
    row["dnp_evidence"] = {"source": "official inactive list", "url": "https://team.test/inactives"}
    grade = settle_entry(ENTRY, evidence(players=[row]))
    assert grade["result"] == "void" and grade["profit_1u"] == 0
    assert grade["actual"] is None and grade["sportsbook_rules_confirmed"] is False


def test_zero_actual_is_gradeable_only_with_participation():
    grade = settle_entry(ENTRY, evidence(players=[{"player": "Brock Purdy", "attempts": 0, "participated": True}]))
    assert grade["result"] == "win"


def test_future_or_preentry_evidence_rejected():
    assert settle_entry(ENTRY, evidence(), as_of="2026-09-11T04:00:00Z")["result"] == "pending"
    assert settle_entry(ENTRY, evidence(observed_at="2026-09-09T05:00:00Z"))["result"] == "pending"


def test_ambiguous_names_do_not_settle():
    rows = evidence()["players"] * 2
    assert settle_entry(ENTRY, evidence(players=rows))["result"] == "pending"


def test_summary_normalization_requires_official_final_flag():
    payload = {"header": {"id": "123", "competitions": [{"status": {"type": {"state": "post", "completed": True}}}]},
               "boxscore": {"players": [{"statistics": [{"name": "passing", "labels": ["C/ATT", "YDS"],
                    "athletes": [{"athlete": {"id": "42", "displayName": "Brock Purdy"}, "stats": ["20/34", "200"]}]}]}]}}
    row = normalize_espn_summary(payload, ENTRY["game_id"], "2026-09-11T05:00:00Z")
    assert row["final"] is True and row["players"][0]["attempts"] == 34
    del payload["header"]["competitions"][0]["status"]["type"]["completed"]
    assert normalize_espn_summary(payload, ENTRY["game_id"], "2026-09-11T05:00:00Z")["final"] is False


def test_augment_is_derived_and_does_not_erase_original_history():
    summary = {"results": [{**ENTRY, "result": "win", "profit_1u": .91}], "record_policy": "first_decision"}
    original = deepcopy(summary)
    updated = augment_results(summary, {ENTRY["game_id"]: evidence()})
    assert summary == original and updated["losses"] == 1 and updated["units"] == -1
    assert updated["record_policy"] == "first_decision"
