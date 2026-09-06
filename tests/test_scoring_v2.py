"""The V2 policy must allow independent edges while retaining data gates."""
import numpy as np
import pandas as pd
import pytest

from qb_attempts import scoring, scoring_v2


class FixedForecast:
    def __init__(self, mean=36.):
        self.mean = mean

    def predict(self, frame):
        return np.repeat(self.mean, len(frame))


class FixedExposure:
    def predict(self, frame):
        return np.tile([.2, .5, .3], (len(frame), 1))


@pytest.fixture
def slate():
    now = pd.Timestamp("2026-09-06T10:30Z")
    kickoff = pd.Timestamp("2026-09-10T00:20Z")
    base = dict(event_id="source-game", game_id="g", season=2026, week=1,
                kickoff=kickoff, as_of=now, home_team="A", away_team="B",
                home=1, rest=7., coach="coach", dome=0, player_id="qb",
                player="Test Quarterback", team="A", opponent="B", roster_status="ACT",
                line=32.5, over_odds=-110, under_odds=-110,
                source_url="https://example.com/odds", observed_at=now.isoformat(), source_updated_at=None)
    quotes = [{**base, "book": book} for book in ("draftkings", "fanduel", "caesars")]
    records = []
    for week in range(1, 13):
        records.append(dict(game_id=f"prior{week}", season=2025, week=week,
            kickoff=pd.Timestamp("2025-09-01T17:00Z") + pd.Timedelta(weeks=week),
            team="A", opponent="B", player_id="qb", player="Test Quarterback", home=1,
            rest=7., coach="coach", opp_coach="other", dome=0, attempts=32.,
            team_attempts=32., plays=65., sack_rate=.06, qb_carries=3.,
            pbp_plays=60., neutral_plays=30., neutral_dropbacks=18.,
            lead_plays=20., lead_dropbacks=9., trail_plays=10., trail_dropbacks=7.,
            dropbacks=34., pass_attempts=30.))
    artifact = dict(estimator=FixedForecast(), exposure=FixedExposure(), residuals=np.arange(-12., 13.))
    context = {"A": dict(available=True, starter="Test Quarterback", injuries=[], season=2026,
                         observed_at=now.isoformat(), source_updated_at=now.isoformat())}
    markets = {"g": dict(game_id="g", home_team="A", away_team="B", kickoff=kickoff.isoformat(),
                          observed_at=now.isoformat(), home_expected_margin=3.5, total_line=45.5)}
    return dict(quotes=quotes, artifact=artifact, records=pd.DataFrame(records), context=context,
                now=now, game_markets=markets)


def offer(candidates, side="Over", book="draftkings"):
    return next(row for row in candidates if row["book"] == book and row["side"] == side)


def test_independent_model_can_qualify_when_market_only_ev_is_negative(slate):
    recommended, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert recommended
    assert row["status"] == "qualified"
    assert row["market_only_ev"] < 0 < row["robust_ev"] < row["ev"]
    assert row["independent_model_ev"] > 0
    assert row["model_weight"] == .5


def test_large_model_disagreement_and_large_ev_are_diagnostic_not_veto(slate):
    slate["artifact"]["estimator"] = FixedForecast(44.)
    recommended, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert recommended and row["status"] == "qualified"
    assert row["ev"] > .15
    assert "Large model–market disagreement" in row["warnings"]
    assert not row["reasons"]
    assert row["model_weight"] == .5


def test_probability_stress_can_hold_an_otherwise_above_threshold_edge(slate, monkeypatch):
    for quote in slate["quotes"]:
        quote.update(over_odds=100, under_odds=100)
    monkeypatch.setattr(scoring_v2, "probabilities", lambda *args: (.536, .464, 0.))
    recommended, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert row["ev"] == pytest.approx(.036)
    assert row["robust_ev"] == pytest.approx(-.004)
    assert not recommended
    assert row["reasons"] == ["Edge fails probability stress test"]


def test_market_alone_cannot_create_recommendation_against_independent_model(slate, monkeypatch):
    slate["quotes"][0].update(over_odds=120, under_odds=-140)
    for quote in slate["quotes"][1:]:
        quote.update(over_odds=-130, under_odds=110)
    monkeypatch.setattr(scoring_v2, "probabilities", lambda *args: (.45, .55, 0.))
    _, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert row["ev"] > .03 and row["robust_ev"] > 0
    assert row["independent_model_ev"] < 0
    assert row["status"] == "held"
    assert "Independent model does not support positive EV" in row["reasons"]


@pytest.mark.parametrize("issue", ["missing", "nan_margin", "nan_total", "old", "future", "naive"])
def test_missing_or_unverified_current_game_market_blocks_scoring(slate, issue):
    market = slate["game_markets"]["g"]
    if issue == "missing":
        slate["game_markets"] = {}
    elif issue == "nan_margin":
        market["home_expected_margin"] = np.nan
    elif issue == "nan_total":
        market["total_line"] = np.nan
    elif issue == "old":
        market["observed_at"] = (slate["now"] - pd.Timedelta(minutes=31)).isoformat()
    elif issue == "future":
        market["observed_at"] = (slate["now"] + pd.Timedelta(minutes=3)).isoformat()
    else:
        market["observed_at"] = "2026-09-06T10:30:00"
    assert scoring_v2.score(**slate) == ([], [], [])


@pytest.mark.parametrize("listed_home,expected", [(True, 3.5), (False, -3.5)])
def test_neutral_venue_uses_designated_home_identity_for_spread(slate, listed_home, expected):
    for quote in slate["quotes"]:
        quote["home"] = 0
        if not listed_home:
            quote.update(home_team="B", away_team="A")
    if not listed_home:
        slate["game_markets"]["g"].update(home_team="B", away_team="A")
    _, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert row["expected_margin"] == expected
    assert row["implied_team_points"] == pytest.approx((45.5 + expected) / 2)


@pytest.mark.parametrize("change", ["limited", "coach", "team"])
def test_declared_changes_reduce_model_weight_without_disagreement_multiplier(slate, change):
    slate["artifact"]["estimator"] = FixedForecast(44.)
    if change == "limited":
        slate["records"] = slate["records"].iloc[:6].copy()
    elif change == "coach":
        for quote in slate["quotes"]:
            quote["coach"] = "new coach"
    else:
        slate["records"].loc[:, "team"] = "OLD"
    _, _, candidates = scoring_v2.score(**slate)
    assert offer(candidates)["model_weight"] == .35


def test_exact_line_reference_required_but_one_independent_book_is_sufficient(slate):
    slate["quotes"] = slate["quotes"][:2]
    recommended, _, candidates = scoring_v2.score(**slate)
    assert recommended and offer(candidates)["other_books"] == 1
    assert "Only one reference book" in offer(candidates)["warnings"]
    slate["quotes"][1]["line"] = 33.5
    recommended, _, candidates = scoring_v2.score(**slate)
    assert not recommended
    assert "No independent other book at this line" in offer(candidates)["reasons"]


def test_offered_price_does_not_enter_reference_and_panels_include_push(slate):
    for quote in slate["quotes"]:
        quote["line"] = 32.
    _, _, candidates = scoring_v2.score(**slate)
    row = offer(candidates)
    assert row["p_push"] > 0
    for weight in (0., .35, .5, .65, 1.):
        panel = row["policy_sensitivity"][str(weight)]
        assert panel["p_win"] == pytest.approx(weight * row["p_model"] + (1 - weight) * row["p_market"])
        assert panel["ev"] == pytest.approx(scoring.ev(panel["p_win"], row["odds"], row["p_push"]))
    assert row["robust_ev"] == pytest.approx(scoring.ev(row["p_final"] - .02 * (1-row["p_push"]), row["odds"], row["p_push"]))
    assert scoring.ev(row["p_final"], row["bet_to"], row["p_push"]) >= .03 - 1e-12
    slate["quotes"][0].update(over_odds=100, under_odds=-120)
    _, _, changed = scoring_v2.score(**slate)
    assert offer(changed)["p_market"] == row["p_market"]
    assert offer(changed)["p_final"] == row["p_final"]


@pytest.mark.parametrize("issue", ["inactive", "starter", "injury", "stale_context"])
def test_high_independent_edge_cannot_override_identity_health_context(slate, issue):
    slate["artifact"]["estimator"] = FixedForecast(44.)
    if issue == "inactive":
        for quote in slate["quotes"]:
            quote["roster_status"] = "RES"
    elif issue == "starter":
        slate["context"]["A"]["starter"] = "Someone Else"
    elif issue == "injury":
        slate["context"]["A"]["injuries"] = ["Questionable"]
    else:
        slate["context"]["A"]["observed_at"] = "2026-09-01T10:30Z"
    recommended, watch, _ = scoring_v2.score(**slate)
    assert not recommended
    assert watch and all(row["reasons"] for row in watch)
