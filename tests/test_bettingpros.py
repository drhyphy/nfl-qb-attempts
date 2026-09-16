from copy import deepcopy

import pytest

from qb_attempts.bettingpros import match_events, parse_bettingpros_offers
from qb_attempts.quote_verification import verify_against_offers

NOW = "2026-09-16T21:25:00+00:00"
GAME = {"game_id": "2026_02_DET_BUF", "season": 2026, "week": 2, "game_type": "REG",
        "home_team": "BUF", "away_team": "DET", "kickoff": "2026-09-18T00:15:00+00:00", "provider_event_id": "21908"}
EVENT = {"id": 21908, "sport": "NFL", "season": 2026, "week": "2", "season_type": "REG", "status": "scheduled",
         "home": "BUF", "visitor": "DET", "scheduled": "2026-09-18 00:15:00"}


def payload():
    return {"_parameters": {"location": "NY", "sport": "NFL", "market_id": "333", "event_id": "21908", "live": "false"},
            "offers": [{"id": "market123", "market_id": 333, "event_id": 21908, "active": True,
                "participants": [{"id": "15501", "name": "Jared Goff", "player": {"position": "QB", "team": "DET"}}],
                "selections": [{"selection": side, "id": side, "active": True, "books": [{"id": 12, "lines": [{
                    "id": "12_" + side, "active": True, "main": True, "is_off": False, "line": 35.5,
                    "cost": -120 if side == "over" else -106, "updated": "2026-09-16 21:18:54",
                    "link": "https://sportsbook.draftkings.com/test"}]}]} for side in ("over", "under")]}]}


def test_exact_event_mapping_and_utc_conversion():
    matched, errors = match_events({"events": [EVENT]}, [GAME], NOW)
    assert not errors and matched["21908"]["game_id"] == GAME["game_id"]
    q, offers, errs = parse_bettingpros_offers(payload(), GAME, NOW)
    assert not errs and len(q) == 1 and len(offers) == 2
    assert q[0]["source_updated_at"] == "2026-09-16T21:18:54+00:00"
    assert offers[0]["id_namespace"] == "bettingpros_provider"
    assert offers[0]["selection_id"] == "12_over"
    assert verify_against_offers(q, offers, NOW)[0]["quote_verification"]["status"] == "verified"


@pytest.mark.parametrize("change", [{"status": "inprogress"}, {"status": None}, {"home": "DET", "visitor": "BUF"},
                                    {"season_type": "PRE"}, {"week": "3"}, {"scheduled": "2026-09-18 00:20:00"}])
def test_reject_inactive_and_mismatched_events(change):
    assert not match_events({"events": [{**EVENT, **change}]}, [GAME], NOW)[0]


def test_duplicate_official_or_provider_identity_rejected():
    assert not match_events({"events": [EVENT]}, [GAME, GAME], NOW)[0]
    assert not match_events({"events": [EVENT, EVENT]}, [GAME], NOW)[0]
    assert not match_events({"events": [EVENT, EVENT, EVENT]}, [GAME], NOW)[0]


@pytest.mark.parametrize("field,value", [("active", False), ("active", None), ("main", False), ("is_off", True), ("is_off", None), ("id", None)])
def test_explicit_line_availability_and_ids_required(field, value):
    data = payload()
    data["offers"][0]["selections"][0]["books"][0]["lines"][0][field] = value
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]


def test_offer_and_selection_active_required():
    for level in ("offer", "selection"):
        data = payload()
        row = data["offers"][0] if level == "offer" else data["offers"][0]["selections"][0]
        del row["active"]
        assert not parse_bettingpros_offers(data, GAME, NOW)[0]


def test_different_lines_and_conflicting_main_quotes_never_pair():
    data = payload()
    data["offers"][0]["selections"][1]["books"][0]["lines"][0]["line"] = 36.5
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]
    data = payload()
    lines = data["offers"][0]["selections"][0]["books"][0]["lines"]
    lines.append({**lines[0], "cost": -125})
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]


def test_multiple_provider_offer_ids_for_same_book_player_rejected():
    data = payload()
    data["offers"].append({**deepcopy(data["offers"][0]), "id": "other_market"})
    q, offers, errors = parse_bettingpros_offers(data, GAME, NOW)
    assert not q and not offers and errors


@pytest.mark.parametrize("value", ["2026-09-16 20:00:00", "2026-09-16 22:00:00", None, "bad"])
def test_old_future_missing_ticks_research_only(value):
    data = payload()
    data["offers"][0]["selections"][1]["books"][0]["lines"][0]["updated"] = value
    q, offers, _ = parse_bettingpros_offers(data, GAME, NOW)
    assert len(q) == 1
    assert verify_against_offers(q, offers, NOW)[0]["quote_verification"]["status"] == "unverified"


def test_oldest_side_timestamp_and_ny_dfs_filter():
    data = payload()
    data["offers"][0]["selections"][1]["books"][0]["lines"][0]["updated"] = "2026-09-16 21:01:00"
    q, _, _ = parse_bettingpros_offers(data, GAME, NOW)
    assert q[0]["source_updated_at"] == "2026-09-16T21:01:00+00:00"
    for selection in data["offers"][0]["selections"]:
        selection["books"][0]["id"] = 36
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]
    data = payload()
    data["_parameters"]["location"] = "NJ"
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]


def test_wrong_event_market_position_and_after_start_rejected():
    for changes in ({"event_id": 999}, {"market_id": 332}):
        data = payload(); data["offers"][0].update(changes)
        assert not parse_bettingpros_offers(data, GAME, NOW)[0]
    data = payload(); data["offers"][0]["participants"][0]["player"]["position"] = "WR"
    assert not parse_bettingpros_offers(data, GAME, NOW)[0]
    assert not parse_bettingpros_offers(payload(), GAME, "2026-09-18T01:00:00Z")[0]
