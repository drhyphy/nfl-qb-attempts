"""Provider-shaped fixtures for identity, signed markets and snapshot safety."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from qb_attempts.game_odds import fetch_game_markets, parse_scoreboard


class GameMarketTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp.now(tz="UTC").floor("s")
        self.kickoff = self.now + pd.Timedelta(days=3)
        self.games = pd.DataFrame([{"game_id": "2026_01_WAS_LA", "espn": "123", "home_team": "LA", "away_team": "WAS",
                                    "kickoff": self.kickoff, "home_score": None, "away_score": None, "result": None}])
        self.quote = {"provider": {"name": "DraftKings"}, "spread": -3.5, "overUnder": 44.5,
                      "homeTeamOdds": {"favorite": True, "team": {"abbreviation": "LAR"}},
                      "awayTeamOdds": {"favorite": False, "team": {"abbreviation": "WSH"}},
                      "pointSpread": {"home": {"close": {"line": "-3.5", "odds": "-110"}},
                                      "away": {"close": {"line": "+3.5", "odds": "-110"}}},
                      "total": {"over": {"close": {"line": "o44.5", "odds": "-105"}},
                                "under": {"close": {"line": "u44.5", "odds": "-115"}}}}
        self.competition = {"date": self.kickoff.isoformat(), "timeValid": True,
                            "status": {"type": {"state": "pre", "name": "STATUS_SCHEDULED", "completed": False}},
                            "competitors": [{"homeAway": "home", "team": {"abbreviation": "LAR"}},
                                            {"homeAway": "away", "team": {"abbreviation": "WSH"}}],
                            "odds": [self.quote]}
        self.payload = {"events": [{"id": "123", "date": self.kickoff.isoformat(), "competitions": [self.competition]}]}

    def parse(self):
        return parse_scoreboard(self.payload, self.games, observed_at=self.now.isoformat())

    def test_home_sign_paired_prices_and_truthful_timestamps(self):
        markets, errors = self.parse()
        self.assertEqual(errors, [])
        q = markets["2026_01_WAS_LA"]
        self.assertEqual(q["home_expected_margin"], 3.5)
        self.assertEqual(q["total_line"], 44.5)
        self.assertEqual(q["books"], ["draftkings"])
        self.assertEqual(q["spread_observed_at"], self.now.isoformat())
        self.assertIsNone(q["source_updated_at"])
        self.assertEqual(q["book_quotes"][0]["under_odds"], -115)

    def test_away_favorite_uses_home_line_not_unsigned_summary(self):
        self.quote["pointSpread"]["home"]["close"]["line"] = "+3.5"
        self.quote["pointSpread"]["away"]["close"]["line"] = "-3.5"
        self.quote["homeTeamOdds"]["favorite"] = False
        self.quote["awayTeamOdds"]["favorite"] = True
        self.assertEqual(self.parse()[0]["2026_01_WAS_LA"]["home_expected_margin"], -3.5)

    def test_pickem_is_real_zero_not_missing_line(self):
        self.quote["pointSpread"]["home"]["close"]["line"] = "0"
        self.quote["pointSpread"]["away"]["close"]["line"] = "0"
        self.assertEqual(self.parse()[0]["2026_01_WAS_LA"]["home_expected_margin"], 0)

    def test_missing_or_bad_spread_is_never_filled_from_summary(self):
        original = copy.deepcopy(self.quote)
        for change in ({"line": None}, {"line": True}, {"line": "nan"}, {"line": "-3.2"}, {"line": "3.5"},
                       {"odds": None}, {"odds": "-95"}, {"odds": "-110.5"}):
            with self.subTest(change=change):
                self.quote.clear()
                self.quote.update(copy.deepcopy(original))
                self.quote["pointSpread"]["home"]["close"].update(change)
                self.assertEqual(self.parse()[0], {})

    def test_both_total_lines_and_prices_required(self):
        original = copy.deepcopy(self.quote)
        for change in ({"line": "u45.5"}, {"line": "o44.5"}, {"line": None}, {"odds": 0}):
            with self.subTest(change=change):
                self.quote.clear()
                self.quote.update(copy.deepcopy(original))
                self.quote["total"]["under"]["close"].update(change)
                self.assertEqual(self.parse()[0], {})
        del self.quote["total"]["under"]["close"]
        self.quote["total"]["under"]["open"] = {"line": "u44.5", "odds": "-110"}
        self.assertEqual(self.parse()[0], {})

    def test_reject_live_complete_postponed_and_unknown_status(self):
        for change in ({"state": "in"}, {"completed": True}, {"name": "STATUS_POSTPONED"}, {"completed": None}):
            with self.subTest(change=change):
                self.competition["status"]["type"] = dict(state="pre", name="STATUS_SCHEDULED", completed=False)
                self.competition["status"]["type"].update(change)
                self.assertEqual(self.parse()[0], {})

    def test_reject_past_beyond_horizon_and_mismatched_event_time(self):
        for kickoff in (self.now - pd.Timedelta(seconds=1), self.now + pd.Timedelta(days=10),
                        self.kickoff + pd.Timedelta(hours=1)):
            with self.subTest(kickoff=kickoff):
                self.competition["date"] = kickoff.isoformat()
                self.assertEqual(self.parse()[0], {})
        self.competition["date"] = str(self.kickoff.tz_localize(None))
        self.assertEqual(self.parse()[0], {})

    def test_completed_schedule_game_does_not_reenter_via_future_date(self):
        self.games.loc[0, "home_score"] = 20
        self.assertEqual(self.parse()[0], {})

    def test_id_team_orientation_and_ambiguity_checked(self):
        self.games.loc[0, "espn"] = "456"
        self.assertEqual(self.parse()[0], {})
        self.games.loc[0, "espn"] = None
        self.assertTrue(self.parse()[0])  # Ordered team/time match only when ID absent.
        self.games.loc[0, "home_team"] = "WAS"
        self.games.loc[0, "away_team"] = "LA"
        self.assertEqual(self.parse()[0], {})
        self.games.loc[0, "home_team"] = "LA"
        self.games.loc[0, "away_team"] = "WAS"
        self.games = pd.concat([self.games, self.games], ignore_index=True)
        self.assertEqual(self.parse()[0], {})

    def test_favorite_contradiction_or_quote_team_mismatch_rejected(self):
        self.quote["homeTeamOdds"]["favorite"] = False
        self.assertEqual(self.parse()[0], {})
        self.quote["homeTeamOdds"]["favorite"] = True
        self.quote["homeTeamOdds"]["team"]["abbreviation"] = "SEA"
        self.assertEqual(self.parse()[0], {})

    def test_multiple_books_median_and_duplicate_conflict(self):
        other = copy.deepcopy(self.quote)
        other["provider"]["name"] = "FanDuel"
        other["pointSpread"]["home"]["close"]["line"] = "-4.5"
        other["pointSpread"]["away"]["close"]["line"] = "+4.5"
        self.competition["odds"].append(other)
        q = self.parse()[0]["2026_01_WAS_LA"]
        self.assertEqual(q["home_expected_margin"], 4)
        self.assertEqual(q["books"], ["draftkings", "fanduel"])
        other["provider"]["name"] = "DraftKings"
        self.assertEqual(self.parse()[0], {})

    def test_provider_identity_and_live_quotes_required(self):
        for provider in (None, [], {}, {"name": ""}, {"name": "consensus"}, {"name": "prizepicks"}):
            with self.subTest(provider=provider):
                self.quote["provider"] = provider
                self.assertEqual(self.parse()[0], {})
        self.quote["provider"] = {"name": "DraftKings"}
        self.quote["isLive"] = True
        self.assertEqual(self.parse()[0], {})

    def test_schedule_date_time_requires_no_assumed_kickoff(self):
        local = self.kickoff.tz_convert("America/New_York")
        self.games = self.games.drop(columns="kickoff")
        self.games["gameday"] = local.strftime("%Y-%m-%d")
        self.games["gametime"] = local.strftime("%H:%M:%S")
        self.assertTrue(self.parse()[0])
        self.games["gametime"] = None
        self.assertEqual(self.parse()[0], {})

    def test_schema_and_observation_errors(self):
        for payload in ([], {}, {"events": None}):
            markets, errors = parse_scoreboard(payload, self.games, observed_at=self.now.isoformat())
            self.assertEqual(markets, {})
            self.assertTrue(errors)
        markets, errors = parse_scoreboard(self.payload, self.games, observed_at="2026-09-06 12:00")
        self.assertEqual(markets, {})
        self.assertTrue(errors)

    def test_http_snapshots_unique_and_never_use_old_body_on_failure(self):
        response = requests.Response()
        response.status_code = 200
        response.url = "https://example.test/public-scoreboard"
        response._content = json.dumps(self.payload).encode()
        with TemporaryDirectory() as directory, patch("qb_attempts.game_odds.requests.get", return_value=response) as get:
            first, errors = fetch_game_markets(self.games, Path(directory), now=self.now)
            self.assertTrue(first)
            self.assertEqual(errors, [])
            second, _ = fetch_game_markets(self.games, Path(directory), now=self.now)
            self.assertTrue(second)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
            meta = json.loads(next(Path(directory).glob("*/scoreboard.0.meta.json")).read_text())
            self.assertEqual(meta["url"], response.url)
            self.assertGreaterEqual(pd.Timestamp(meta["observed_at"]), self.now)
            get.side_effect = requests.ConnectionError("offline")
            failed, errors = fetch_game_markets(self.games, Path(directory), now=self.now)
            self.assertEqual(failed, {})
            self.assertIn("offline", errors[0])
            self.assertEqual(len(list(Path(directory).iterdir())), 3)

    def test_empty_window_makes_no_http_request(self):
        self.games.loc[0, "kickoff"] = self.now + pd.Timedelta(days=10)
        with TemporaryDirectory() as directory, patch("qb_attempts.game_odds.requests.get") as get:
            self.assertEqual(fetch_game_markets(self.games, Path(directory), now=self.now), ({}, []))
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
