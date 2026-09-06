import copy
import unittest

from qb_attempts.context import parse_depthchart


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


if __name__ == "__main__":
    unittest.main()
