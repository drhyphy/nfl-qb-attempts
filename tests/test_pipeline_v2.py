"""Cloud-safe daily pipeline regressions; all feeds and artifacts are isolated."""
import contextlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_script('daily_pipeline_test', REPO / 'scripts/run_daily.py')
tendencies = load_script('tendencies_pipeline_test', REPO / 'scripts/fetch_tendencies.py')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        now = pd.Timestamp.now(tz='UTC')
        self.season = now.year - (now.month <= 2)
        self.game_id = f'{self.season}_01_NE_SEA'

    def run_case(self, root, markets, capture=False):
        model = root / 'data/model/model.pkl'
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(pickle.dumps({'version': pipeline.VERSION, 'metrics': {}}))
        tendency_file = root / 'data/tendencies/team_games.parquet'
        tendency_file.parent.mkdir(parents=True, exist_ok=True)
        tendency_file.touch()
        quote = {'game_id': self.game_id, 'player_id': 'p1', 'team': 'NE', 'home_team': 'SEA', 'week': 1}
        games = pd.DataFrame([{'kickoff': pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=1), 'season': self.season}])
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(pipeline, 'ROOT', root))
            stack.enter_context(patch.object(sys, 'argv', ['run_daily.py'] + (['--capture-only', '--train'] if capture else [])))
            stack.enter_context(patch.object(pipeline, 'load_inputs', return_value=(pd.DataFrame(), games)))
            stack.enter_context(patch.object(pipeline.pd, 'read_parquet', return_value=pd.DataFrame()))
            stack.enter_context(patch.object(pipeline, 'fetch_quotes', return_value=([quote], [])))
            stack.enter_context(patch.object(pipeline, 'resolve_quotes', return_value=([quote], [])))
            context = stack.enter_context(patch.object(pipeline, 'fetch_context', return_value=({'NE': {'available': True}}, [])))
            market = stack.enter_context(patch.object(pipeline, 'fetch_game_markets', return_value=(markets, [])))
            stack.enter_context(patch.object(pipeline, 'records_with_context', return_value=pd.DataFrame()))
            stack.enter_context(patch.object(pipeline, 'grade_history', return_value={'settled': 0}))
            score = stack.enter_context(patch('qb_attempts.scoring_v2.score', return_value=([], [], [
                {'game_id': self.game_id, 'ev': .02, 'policy_sensitivity': {'0.5': .02}},
            ])))
            subprocess = stack.enter_context(patch.object(pipeline.subprocess, 'run'))
            try:
                pipeline.main()
            except SystemExit as error:
                code = error.code
            else:
                code = 0
            if capture:
                context.assert_not_called()
                market.assert_not_called()
                score.assert_not_called()
                subprocess.assert_not_called()
        return code

    def market(self, age=0):
        return {self.game_id: {'observed_at': (pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=age)).isoformat(),
                              'home_expected_margin': 3.5, 'total_line': 44.5}}

    def test_success_preserves_immutable_evaluations_and_prior_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_case(root, self.market()), 0)
            first = next((root / 'data/published/history').glob('*.json'))
            original = first.read_bytes()
            board = json.loads((root / 'site/public/board.json').read_text())
            self.assertEqual(board['status'], 'ok')
            self.assertEqual(board['game_market_coverage']['verified'], 1)
            evaluation_path = root / f"data/published/evaluations/{board['run_id']}.json"
            evaluation_bytes = evaluation_path.read_bytes()
            evaluation = json.loads(evaluation_bytes)
            self.assertEqual(evaluation['evaluated_quotes'][0]['policy_sensitivity'], {'0.5': .02})
            self.assertEqual(len(evaluation['quotes']), 1)
            self.assertEqual(self.run_case(root, self.market()), 0)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(evaluation_path.read_bytes(), evaluation_bytes)
            self.assertEqual(len(list(first.parent.glob('*.json'))), 2)
            self.assertEqual(len(list(evaluation_path.parent.glob('*.json'))), 2)

    def test_missing_and_stale_game_markets_publish_source_failure(self):
        for markets in ({}, self.market(age=31)):
            with self.subTest(markets=markets), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertEqual(self.run_case(root, markets), 2)
                board = json.loads((root / 'site/public/board.json').read_text())
                self.assertEqual(board['status'], 'source_failure')
                self.assertEqual(board['game_market_coverage']['verified'], 0)
                self.assertEqual(board['recommendations'], [])
                self.assertTrue(board['source_errors'])

    def test_capture_only_skips_training_and_v2_feeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_case(root, {}, capture=True), 0)
            self.assertEqual(len(list((root / 'data/published/closing').glob('*.json'))), 1)
            self.assertFalse((root / 'data/published/history').exists())
            self.assertFalse((root / 'data/published/evaluations').exists())

    def test_verified_historical_cache_still_refreshes_current_season(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'data/tendencies'
            output.mkdir(parents=True)
            prior_season = self.season - 1
            # Small, self-contained aggregate; no checkout data or network needed.
            rows = []
            for team, opponent in [('NE', 'SEA'), ('SEA', 'NE')]:
                rows.append({'game_id': f'{prior_season}_01_NE_SEA', 'season': prior_season,
                             'week': 1, 'team': team, 'opponent': opponent,
                             'game_date': f'{prior_season}-09-10',
                             'plays': 60, 'dropbacks': 36, 'pass_attempts': 33,
                             'neutral_plays': 30, 'neutral_dropbacks': 18,
                             'lead_plays': 10, 'lead_dropbacks': 5,
                             'trail_plays': 20, 'trail_dropbacks': 13, 'xpass_plays': 60,
                             'dropback_rate': .6, 'attempts_per_dropback': 33 / 36,
                             'neutral_dropback_rate': .6, 'lead_dropback_rate': .5,
                             'trail_dropback_rate': .65, 'dropback_oe': .02})
            aggregate = output / 'team_games.parquet'
            pd.DataFrame(rows).to_parquet(aggregate, index=False)
            manifest_path = output / 'manifest.json'
            manifest_path.write_text(json.dumps({
                'aggregation_version': tendencies.AGGREGATION_VERSION,
                'aggregate_sha256': tendencies.sha256(aggregate),
                'files': [{'season': prior_season, 'cached': False}],
            }))

            def current_only(season, **kwargs):
                self.assertEqual(season, self.season)
                return None, {'season': season, 'unavailable': True}

            with patch.object(tendencies, 'ROOT', root), \
                    patch.object(sys, 'argv', ['fetch_tendencies.py', '--start-season', str(prior_season)]), \
                    patch.object(tendencies, 'fetch_season', side_effect=current_only) as fetch, \
                    contextlib.redirect_stdout(io.StringIO()):
                tendencies.main()
                fetch.assert_called_once()
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['team_games'], len(rows))
            prior = next(entry for entry in manifest['files'] if entry['season'] == prior_season)
            self.assertTrue(prior['cached'])
            self.assertEqual(prior['cache_source'], 'verified_combined_aggregate')
            self.assertEqual(manifest['aggregate_sha256'], tendencies.sha256(aggregate))


if __name__ == '__main__':
    unittest.main()
