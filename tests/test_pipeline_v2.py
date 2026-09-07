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

    def run_case(self, root, markets, capture=False, challenger_error=None,
                 champion_result=None, challenger_result=None):
        model = root / 'data/model/model.pkl'
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(pickle.dumps({'version': pipeline.VERSION, 'metrics': {}}))
        tendency_file = root / 'data/tendencies/team_games.parquet'
        tendency_file.parent.mkdir(parents=True, exist_ok=True)
        tendency_file.touch()
        kickoff = (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=1)).isoformat()
        quote = {'game_id': self.game_id, 'player_id': 'p1', 'team': 'NE', 'home_team': 'SEA', 'week': 1,
                 'kickoff': kickoff, 'player': 'Quarterback', 'book': 'a', 'line': 30.5}
        games = pd.DataFrame([{'game_id': self.game_id, 'kickoff': pd.Timestamp(kickoff), 'season': self.season}])
        candidate = {**quote, 'side': 'Over', 'p_final': .6, 'p_push': 0., 'mean': 33.,
                     'odds': -110, 'ev': .02, 'policy_sensitivity': {'0.5': .02}}
        if champion_result is None:
            champion_result = ([], [], [candidate])
        if challenger_result is None:
            challenger_result = {'status': 'ok', 'model_version': 'claude-test-v1',
                                 'recommendations': [], 'watchlist': [],
                                 'evaluated_quotes': [{**candidate, 'p_final': .4, 'mean': 29.}]}
        # Keep caller-provided candidates on this run's shared canonical event.
        champion_result = tuple([{**row, 'game_id': self.game_id, 'kickoff': kickoff}
                                  for row in group] for group in champion_result)
        challenger_result = {**challenger_result, **{field: [
            {**row, 'game_id': self.game_id, 'kickoff': kickoff} for row in challenger_result.get(field, [])]
            for field in ('recommendations', 'watchlist', 'evaluated_quotes')}}
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
            grade = stack.enter_context(patch.object(pipeline, 'grade_history', return_value={'settled': 0}))
            score = stack.enter_context(patch('qb_attempts.scoring_v2.score', return_value=champion_result))
            challenger_load = stack.enter_context(patch('qb_attempts.challenger.load_challenger', return_value={}))
            challenger_score = stack.enter_context(patch('qb_attempts.challenger.score_challenger',
                                                        return_value=challenger_result, side_effect=challenger_error))
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
                challenger_load.assert_not_called()
                challenger_score.assert_not_called()
            self.grade_paths = [call.args[0] for call in grade.call_args_list]
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

    def prediction(self, side='Over', probability=.6, mean=33.):
        return {'player_id': 'p1', 'player': 'Quarterback', 'team': 'NE', 'book': 'a',
                'line': 30.5, 'side': side, 'p_final': probability, 'p_push': 0.,
                'mean': mean, 'odds': -110, 'ev': .12, 'robust_ev': .08}

    def test_shared_snapshots_and_separate_first_decision_histories_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champion = self.prediction()
            challenger = self.prediction(side='Under', probability=.7, mean=29.)
            self.assertEqual(self.run_case(root, self.market(),
                champion_result=([champion], [], [champion]),
                challenger_result={'status': 'ok', 'model_version': 'claude-test-v1',
                                   'recommendations': [challenger], 'evaluated_quotes': [challenger]}), 0)
            published = root / 'data/published'
            board = json.loads((published / 'latest.json').read_text())
            paired = board['comparison']['paired_forecasts']
            self.assertEqual(len(paired), 1)
            self.assertAlmostEqual(paired[0]['champion']['p_over'], .6)
            self.assertAlmostEqual(paired[0]['challenger']['p_over'], .3)
            run_id = board['run_id']
            shared_file = published / f'history/{run_id}.json'
            champion_file = published / f'comparison/champion/{run_id}.json'
            challenger_file = published / f'challengers/claude/history/{run_id}.json'
            frozen_bytes = {path: path.read_bytes() for path in (shared_file, champion_file, challenger_file)}
            self.assertEqual(json.loads(champion_file.read_text())['recommendations'][0]['side'], 'Over')
            self.assertEqual(json.loads(challenger_file.read_text())['recommendations'][0]['side'], 'Under')
            self.assertEqual(set(self.grade_paths), {published / 'history',
                published / 'comparison/champion', published / 'challengers/claude/history'})
            self.assertEqual(self.run_case(root, self.market()), 0)
            for path, original in frozen_bytes.items():
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(len(list(path.parent.glob('*.json'))), 2)

    def test_challenger_failure_preserves_champion_card_but_fails_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champion = self.prediction()
            self.assertEqual(self.run_case(root, self.market(), challenger_error=RuntimeError('broken challenger'),
                                           champion_result=([champion], [], [champion])), 2)
            published = root / 'data/published'
            board = json.loads((published / 'latest.json').read_text())
            self.assertEqual(board['status'], 'ok')
            self.assertEqual(board['recommendations'][0]['side'], 'Over')
            self.assertEqual(board['challenger']['status'], 'source_failure')
            self.assertIn('broken challenger', board['challenger']['source_errors'][0])
            self.assertEqual(board['comparison']['paired_forecasts'], [])
            self.assertFalse((published / 'comparison/champion').exists())
            self.assertFalse((published / 'challengers/claude/history').exists())
            self.assertEqual(len(list((published / 'history').glob('*.json'))), 1)
            # Comparison starts only when both models successfully run together.
            self.assertEqual(self.run_case(root, self.market()), 0)
            self.assertEqual(len(list((published / 'history').glob('*.json'))), 2)
            self.assertEqual(len(list((published / 'comparison/champion').glob('*.json'))), 1)
            self.assertEqual(len(list((published / 'challengers/claude/history').glob('*.json'))), 1)

    def test_champion_source_failure_cannot_publish_shared_forecast_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_case(root, {}), 2)
            board = json.loads((root / 'data/published/latest.json').read_text())
            self.assertEqual(board['status'], 'source_failure')
            self.assertEqual(board['comparison']['paired_forecasts'], [])
            self.assertFalse((root / 'data/published/comparison/champion').exists())
            self.assertFalse((root / 'data/published/challengers/claude/history').exists())

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
