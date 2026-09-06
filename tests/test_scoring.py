import numpy as np
import pandas as pd
import pytest

from qb_attempts import scoring


@pytest.fixture
def slate():
    now = pd.Timestamp("2026-09-06T10:30:00Z")
    kickoff = pd.Timestamp("2026-09-10T00:20:00Z")
    game = dict(game_id="scheduled", season=2026, week=1, kickoff=kickoff,
                home_team="A", away_team="B", location="Home", home_rest=7., away_rest=7.,
                home_coach="coach", away_coach="other", roof="outdoors")
    roster = pd.DataFrame([dict(full_name="Test Quarterback", team="A", position="QB",
                                week=1, gsis_id="qb", status="ACT")])
    quote = dict(event_id="provider-event", kickoff=kickoff.isoformat(), observed_at=now.isoformat(),
                 source_updated_at=None, home_team="A", away_team="B", player="Test Quarterback",
                 team="A", opponent="B", line=32., book="draftkings", over_odds=125,
                 under_odds=-145, source_url="https://example.com/quote")
    quotes = [quote] + [{**quote, "book": book, "over_odds": -110, "under_odds": -110}
                        for book in ("fanduel", "caesars")]
    history = []
    for week in range(1, 13):
        history.append(dict(game_id=f"prior-{week}", season=2025, week=week,
            kickoff=pd.Timestamp("2025-09-01T17:00Z") + pd.Timedelta(weeks=week),
            team="A", opponent="B", player_id="qb", player="Test Quarterback", home=1,
            rest=7., coach="coach", opp_coach="other", dome=0, attempts=32.,
            team_attempts=32., plays=65., sack_rate=.06, qb_carries=3.))
    artifact = dict(name="recent_mean", estimator=None, residuals=np.arange(-12., 13.))
    context = {"A": {"available": True, "starter": "Test Quarterback", "injuries": [],
                      "season": 2026, "observed_at": now.isoformat(), "source_updated_at": now.isoformat()}}
    return dict(now=now, quotes=quotes, roster=roster, games=pd.DataFrame([game]),
                records=pd.DataFrame(history), artifact=artifact, context=context)


def resolved(slate):
    return scoring.resolve_quotes(slate["quotes"], slate["roster"], slate["games"], slate["now"])


def run(slate):
    quotes, errors = resolved(slate)
    assert not errors
    return scoring.score(quotes, slate["artifact"], slate["records"], slate["context"], slate["now"])


def offer(rows, book="draftkings", side="Over"):
    return next(r for r in rows if r["book"] == book and r["side"] == side)


def test_offered_book_never_influences_its_reference_probability(slate):
    recommendations, _, candidates = run(slate)
    first = offer(candidates)
    assert len(recommendations) == 1
    assert first["reference_books"] == ["caesars", "fanduel"]
    slate["quotes"][0].update(over_odds=130, under_odds=-155)
    _, _, changed = run(slate)
    second = offer(changed)
    assert first["p_market"] == pytest.approx(second["p_market"])
    assert first["p_final"] == pytest.approx(second["p_final"])
    assert first["ev"] != second["ev"]


def test_duplicate_book_gets_one_vote_latest_observation_wins(slate):
    duplicate = {**slate["quotes"][1], "observed_at": (slate["now"] - pd.Timedelta(minutes=2)).isoformat(),
                 "over_odds": 200, "under_odds": -250}
    slate["quotes"].extend([duplicate, duplicate.copy()])
    quotes, errors = resolved(slate)
    assert not errors
    assert len(quotes) == 3
    assert next(q for q in quotes if q["book"] == "fanduel")["over_odds"] == -110
    _, _, candidates = run(slate)
    assert offer(candidates)["other_books"] == 2


def test_latest_book_observation_is_ordered_by_instant_not_iso_text(slate):
    slate["quotes"].append({**slate["quotes"][1], "observed_at": "2026-09-06T06:31:00-04:00",
                            "over_odds": -105, "under_odds": -115})
    quotes, errors = resolved(slate)
    assert not errors
    assert next(q for q in quotes if q["book"] == "fanduel")["over_odds"] == -105


def test_integer_line_push_ev_fair_and_bet_to_are_consistent(slate):
    _, _, candidates = run(slate)
    r = offer(candidates)
    p, push = r["p_final"], r["p_push"]
    assert push > 0
    assert r["ev"] == pytest.approx(p * 1.25 - (1 - p - push))
    assert r["market_only_ev"] == pytest.approx(scoring.ev(r["p_market"], r["odds"], push))
    assert scoring.ev(p, r["bet_to"], push) >= .02 - 1e-12
    assert r["fair_odds"] == scoring.fair_odds(p / (1 - push))


@pytest.mark.parametrize("p,push", [(.4, 0.), (.55, 0.), (.6, .05), (.3, .2), (.8, .1)])
def test_bet_to_rounding_never_promises_unachievable_target_ev(p, push):
    threshold = scoring.bet_to(p, push, target=.02)
    assert scoring.ev(p, threshold, push) >= .02 - 1e-12
    # Numerically one point worse in American odds, with the +/-100 discontinuity handled.
    worse = threshold - 1 if threshold != 100 else -101
    assert scoring.ev(p, worse, push) < .02 + 1e-12


@pytest.mark.parametrize("delta", [pd.Timedelta(minutes=-31), pd.Timedelta(minutes=3)])
def test_stale_or_future_retrieval_is_rejected(slate, delta):
    slate["quotes"] = [{**slate["quotes"][0], "observed_at": (slate["now"] + delta).isoformat()}]
    quotes, errors = resolved(slate)
    assert not quotes
    assert any("stale or future" in error for error in errors)


@pytest.mark.parametrize("delta", [pd.Timedelta(hours=-25), pd.Timedelta(minutes=3)])
def test_source_tick_outside_age_policy_is_rejected_despite_fresh_retrieval(slate, delta):
    slate["quotes"] = [{**slate["quotes"][0], "source_updated_at": (slate["now"] + delta).isoformat()}]
    quotes, errors = resolved(slate)
    assert not quotes
    assert errors


def test_fresh_retrieval_can_confirm_unchanged_price_with_three_hour_old_tick(slate):
    slate["quotes"] = [{**slate["quotes"][0], "source_updated_at": (slate["now"] - pd.Timedelta(hours=3)).isoformat()}]
    quotes, errors = resolved(slate)
    assert len(quotes) == 1
    assert not errors


def test_schedule_mismatch_is_rejected(slate):
    slate["quotes"] = [{**slate["quotes"][0], "kickoff": "2026-09-11T00:20:00Z"}]
    quotes, errors = resolved(slate)
    assert not quotes
    assert any("schedule identity" in error for error in errors)


def test_roster_valid_player_from_third_team_cannot_attach_to_game(slate):
    slate["roster"].loc[:, "team"] = "C"
    slate["quotes"] = [{**slate["quotes"][0], "team": "C", "opponent": "A"}]
    quotes, errors = resolved(slate)
    assert not quotes
    assert errors


@pytest.mark.parametrize("issue", ["inactive", "missing_context", "other_starter", "injured"])
def test_uncertain_or_inactive_starting_role_blocks_recommendations(slate, issue):
    if issue == "inactive":
        slate["roster"].loc[:, "status"] = "RES"
    elif issue == "missing_context":
        slate["context"] = {}
    elif issue == "other_starter":
        slate["context"]["A"]["starter"] = "Someone Else"
    else:
        slate["context"]["A"]["injuries"] = ["Questionable"]
    recommended, watch, candidates = run(slate)
    assert not recommended
    assert watch and candidates
    assert all(row["status"] == "held" and row["reasons"] for row in candidates)


@pytest.mark.parametrize("field,value", [
    ("observed_at", "2026-09-01T10:30:00Z"),
    ("source_updated_at", "2026-09-01T10:30:00Z"),
    ("observed_at", "2026-09-07T10:30:00Z"),
    ("source_updated_at", "2026-09-07T10:30:00Z"),
    ("observed_at", None),
    ("source_updated_at", None),
    ("season", 2025),
])
def test_saved_context_cannot_bypass_date_and_season_checks(slate, field, value):
    slate["context"]["A"][field] = value
    recommended, watch, _ = run(slate)
    assert not recommended
    assert watch and watch[0]["reasons"]


def test_no_edge_is_valid_no_bet_state_and_empty_slate_is_valid(slate):
    for quote in slate["quotes"]:
        quote.update(over_odds=-110, under_odds=-110)
    recommended, watch, candidates = run(slate)
    assert not recommended
    assert len(watch) == 1
    assert len(candidates) == 6
    assert scoring.score([], slate["artifact"], slate["records"], slate["context"], slate["now"]) == ([], [], [])


def test_different_line_cannot_supply_consensus_support(slate):
    slate["quotes"][2]["line"] = 32.5
    recommended, _, candidates = run(slate)
    assert not recommended
    own = offer(candidates)
    assert own["other_books"] == 1
    assert "Fewer than 2 other books at this line" in own["reasons"]


def test_nonlocal_book_can_supply_reference_but_cannot_be_selected(slate):
    slate["quotes"][2]["book"] = "bet365"
    _, _, candidates = run(slate)
    assert "bet365" in offer(candidates)["reference_books"]
    assert all(row["book"] != "bet365" for row in candidates)
