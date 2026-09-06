"""Public feed parser checks; fixtures are minimal provider-shaped examples."""
import copy
import json
import unittest

from qb_attempts.odds_sources import (
    BoardEntry, parse_board_html, parse_comparison_payload, parse_schedule_html,
)


class PublicOddsParserTests(unittest.TestCase):
    def setUp(self):
        self.entry = BoardEntry("nfl/123", "Test Quarterback")
        self.observed = "2026-09-06T12:00:00+00:00"
        self.schedule = {"nfl/123": {"home_team": "LA", "away_team": "WAS", "kickoff": "2026-09-13T20:25:00+00:00"}}
        self.payload = {
            "event": {"sport": "nfl", "home": {"key": "LAR"}, "away": {"key": "WSH"}},
            "markets": [{"id": "nfl.123.0.555.paatt", "stat": "pass attempts", "available": True,
                         "sportsbook": 15, "date": 1788609600,
                         "player": {"first_name": "Test", "last_name": "Quarterback", "team": {"key": "WSH"}},
                         "comparison": {"draftkings": {"sportsbook": 15, "available": True, "value": 32.5, "over": -110, "under": -110},
                                        "riverscasino": {"sportsbook": 110, "available": True, "value": 33, "over": 100, "under": -120}}}],
        }

    def parse(self):
        return parse_comparison_payload(self.payload, self.entry, self.schedule, observed_at=self.observed)

    def test_paired_prices_teams_integer_lines_and_book_normalization(self):
        rows = self.parse()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["team"], "WAS")
        self.assertEqual(rows[0]["opponent"], "LA")
        self.assertEqual(rows[1]["book"], "betrivers")
        self.assertEqual(rows[1]["line"], 33.0)
        self.assertEqual(rows[0]["event_id"], "nfl/123")

    def test_update_timestamp_does_not_spread_across_books(self):
        rows = self.parse()
        self.assertIsNotNone(rows[0]["source_updated_at"])
        self.assertIsNone(rows[1]["source_updated_at"])
        self.assertEqual(rows[1]["observed_at"], self.observed)

    def test_missing_date_started_game_and_team_mismatch_rejected(self):
        original = copy.deepcopy(self.schedule)
        for schedule in ({}, {"nfl/123": dict(original["nfl/123"], kickoff="2026-09-05T20:25:00+00:00")},
                         {"nfl/123": dict(original["nfl/123"], away_team="SEA")}):
            with self.subTest(schedule=schedule):
                self.schedule = schedule
                self.assertEqual(self.parse(), [])

    def test_wrong_market_live_and_partial_game_rejected(self):
        original = copy.deepcopy(self.payload)
        for change in ({"stat": "completions"}, {"stat": "pass-attempts"}, {"available": False},
                       {"id": "nfl.123.1.555.paatt"}, {"id": "nfl.999.0.555.paatt"},
                       {"is_live": True}, {"line_status": "suspended"}):
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                self.payload["markets"][0].update(change)
                self.assertEqual(self.parse(), [])

    def test_unknown_player_or_team_rejected(self):
        self.payload["markets"][0]["player"]["last_name"] = "SomeoneElse"
        self.assertEqual(self.parse(), [])
        self.payload["markets"][0]["player"]["last_name"] = "Quarterback"
        self.payload["markets"][0]["player"]["team"]["key"] = "SEA"
        self.assertEqual(self.parse(), [])

    def test_invalid_price_unpaired_live_and_unavailable_quote_rejected(self):
        original = copy.deepcopy(self.payload)
        for change in ({"under": None}, {"over": 0}, {"over": True}, {"over": -99}, {"over": -110.1},
                       {"over": float("nan")}, {"value": float("inf")}, {"value": 31.2}, {"value": True},
                       {"available": False}, {"is_live": True}, {"line_status": "suspended"}):
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                self.payload["markets"][0]["comparison"]["draftkings"].update(change)
                rows = self.parse()
                self.assertEqual([r["book"] for r in rows], ["betrivers"])

    def test_pickem_synthetic_odds_are_not_sportsbook_prices(self):
        quotes = self.payload["markets"][0]["comparison"]
        for name in ("prizepicks", "sleeper", "underdog", "consensus"):
            quotes[name] = copy.deepcopy(quotes["draftkings"])
        self.assertEqual(len(self.parse()), 2)

    def test_board_filters_pregame_exact_stat_and_deduplicates(self):
        row = '<li class="border" data-name="test" data-state="Pregame"><div data-role="chassis" data-event="nfl/123" data-market="pass attempts" data-filter="Test Quarterback"></div></li>'
        html = row + row + row.replace("Pregame", "Live") + row.replace("pass attempts", "completions")
        self.assertEqual(parse_board_html(html), [self.entry])

    def test_schedule_requires_timezone_scheduled_state_and_known_teams(self):
        event = {"@type": "SportsEvent", "identifier": 123, "eventStatus": "https://schema.org/EventScheduled",
                 "startDate": "2026-09-13T16:25:00-04:00", "homeTeam": {"name": "LAR Rams"}, "awayTeam": {"name": "WSH Commanders"}}
        def html(e):
            return '<script type="application/ld+json">' + json.dumps(e) + '</script>'
        parsed = parse_schedule_html(html(event))
        self.assertEqual(parsed["nfl/123"]["kickoff"], "2026-09-13T20:25:00+00:00")
        self.assertEqual(parsed["nfl/123"]["home_team"], "LA")
        for change in ({"startDate": "2026-09-13T16:25:00"}, {"startDate": None},
                       {"eventStatus": "https://schema.org/EventPostponed"}, {"homeTeam": {"name": "XXX Unknown"}}):
            with self.subTest(change=change):
                self.assertEqual(parse_schedule_html(html(dict(event, **change))), {})


if __name__ == "__main__":
    unittest.main()
