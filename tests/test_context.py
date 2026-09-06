import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import requests

from qb_attempts.context import BASE_URL, FALLBACK_BASE_URL, ESPN_TEAM_IDS, _fetch_team, parse_depthchart
from qb_attempts.odds_sources import NFL_TEAMS


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"timestamp": "2026-09-06T12:00:00Z", "status": "success", "season": {"year": 2026},
                        "team": {"abbreviation": "LAR"}, "depthchart": [
                            {"positions": {"qb": {"athletes": [{"id": "1", "displayName": "Starter QB", "injuries": []},
                                                                  {"id": "2", "displayName": "Backup QB", "injuries": [{"status": "Out"}]}]},
                                           "lt": {"athletes": [{"id": "3", "displayName": "Left Tackle", "injuries": [{"status": "Questionable", "date": "2026-09-05T19:00:00Z"}]}]},
                                           "wr1": {"athletes": [{"id": "4", "displayName": "Receiver", "injuries": []}, {"id": "5", "displayName": "Reserve", "injuries": [{"status": "Injured Reserve"}]}]}}}]}

    def parse(self):
        return parse_depthchart(self.payload, "LA", observed_at="2026-09-06T13:00:00Z", source_url="https://example.test/depthcharts")

    def test_first_qb_and_backup_injury_not_assigned_to_starter(self):
        row = self.parse()
        self.assertTrue(row["available"])
        self.assertEqual(row["starter"], "Starter QB")
        self.assertEqual(row["espn_id"], "1")
        self.assertEqual(row["injuries"], [])
        self.assertEqual([i["depth"] for i in row["offense_injuries"]], [1, 2])
        self.assertEqual(row["source_updated_at"], "2026-09-06T12:00:00+00:00")

    def test_starter_injury_preserved_and_availability_is_source_availability(self):
        self.payload["depthchart"][0]["positions"]["qb"]["athletes"][0]["injuries"] = [{"status": "Questionable"}]
        row = self.parse()
        self.assertTrue(row["available"])
        self.assertEqual(row["injuries"], ["Questionable"])

    def test_conflicting_first_qbs_do_not_invent_starter(self):
        other = copy.deepcopy(self.payload["depthchart"][0])
        other["positions"]["qb"]["athletes"].reverse()
        self.payload["depthchart"].append(other)
        row = self.parse()
        self.assertFalse(row["available"])
        self.assertIsNone(row["starter"])

    def test_consistent_multiple_formations_permitted(self):
        self.payload["depthchart"].append(copy.deepcopy(self.payload["depthchart"][0]))
        self.assertTrue(self.parse()["available"])
        self.assertEqual(len(self.parse()["offense_injuries"]), 2)

    def test_missing_qb_or_empty_qb_chart_unavailable(self):
        self.payload["depthchart"][0]["positions"]["qb"]["athletes"] = []
        self.assertFalse(self.parse()["available"])
        self.payload["depthchart"] = []
        self.assertFalse(self.parse()["available"])

    def test_prior_season_wrong_team_stale_and_missing_dates_unavailable(self):
        original = copy.deepcopy(self.payload)
        for change in ({"season": {"year": 2025}}, {"team": {"abbreviation": "SEA"}},
                       {"timestamp": "2026-09-01T12:00:00Z"}, {"timestamp": "2026-09-07T12:00:00Z"},
                       {"timestamp": None}, {"timestamp": "2026-09-06T12:00:00"}, {"status": "error"}):
            with self.subTest(change=change):
                self.payload = dict(original, **change)
                self.assertFalse(self.parse()["available"])

    def test_unknown_injury_is_not_clean_health(self):
        self.payload["depthchart"][0]["positions"]["qb"]["athletes"][0]["injuries"] = [{}]
        self.assertEqual(self.parse()["injuries"], ["Unspecified injury"])

    def response(self, url, status=200):
        now = datetime.now(timezone.utc)
        payload = copy.deepcopy(self.payload)
        payload.update(timestamp=now.isoformat(), season={"year": now.year - (now.month <= 2)})
        response = requests.Response()
        response.status_code, response.url = status, url
        response._content = json.dumps(payload).encode() if status == 200 else b"Forbidden"
        return response

    def test_web_primary_uses_verified_numeric_id_and_no_unnecessary_fallback(self):
        url = f"{BASE_URL}/14/depthcharts"
        with tempfile.TemporaryDirectory() as directory, patch("qb_attempts.context.requests.get", return_value=self.response(url)) as get:
            result = _fetch_team("LA", Path(directory))
            self.assertTrue(result["available"])
            self.assertEqual(result["source_url"], url)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args.args[0], url)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_403_moves_to_public_fallback_and_preserves_failed_body(self):
        web, fallback = f"{BASE_URL}/14/depthcharts", f"{FALLBACK_BASE_URL}/14/depthcharts"
        responses = [self.response(web, 403), self.response(fallback)]
        with tempfile.TemporaryDirectory() as directory, patch("qb_attempts.context.requests.get", side_effect=responses) as get:
            result = _fetch_team("LA", Path(directory))
            self.assertTrue(result["available"])
            self.assertEqual(result["source_url"], fallback)
            self.assertEqual(get.call_count, 2)
            self.assertEqual((Path(directory) / "LA.0.0.body.json").read_bytes(), b"Forbidden")
            self.assertEqual(len(list(Path(directory).iterdir())), 4)

    def test_both_endpoints_fail_closed(self):
        web, fallback = f"{BASE_URL}/14/depthcharts", f"{FALLBACK_BASE_URL}/14/depthcharts"
        with tempfile.TemporaryDirectory() as directory, patch("qb_attempts.context.requests.get", side_effect=[self.response(web, 403), self.response(fallback, 403)]) as get:
            with self.assertRaises(RuntimeError):
                _fetch_team("LA", Path(directory))
            self.assertEqual(get.call_count, 2)

    def test_verified_mapping_covers_all_32_distinct_teams(self):
        self.assertEqual(set(ESPN_TEAM_IDS), NFL_TEAMS)
        self.assertEqual(len(set(ESPN_TEAM_IDS.values())), 32)


if __name__ == "__main__":
    unittest.main()
