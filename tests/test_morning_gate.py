import copy
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('morning_gate', Path(__file__).resolve().parents[1] / 'scripts/morning_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def board_at(timestamp, status='ok'):
    model = dict(generated_at=timestamp, status=status, recommendations=[], watchlist=[])
    return dict(model, challenger=copy.deepcopy(model))


@pytest.mark.parametrize('day,utc_hour', [('2026-09-08', 10), ('2026-12-08', 11)])
def test_dst_boundary(day, utc_hour):
    before = datetime.fromisoformat(f'{day}T{utc_hour:02}:29:59+00:00')
    after = datetime.fromisoformat(f'{day}T{utc_hour:02}:30:00+00:00')
    assert not gate.should_run('schedule', False, None, before)[0]
    assert gate.should_run('schedule', False, None, after)[0]


@pytest.mark.parametrize('status', ['ok', 'no_odds'])
def test_successful_publication_skips_late_and_recovery_runs(status):
    board = board_at('2026-09-08T10:34:00Z', status)
    now = datetime.fromisoformat('2026-09-08T16:00:00+00:00')
    for event in ['schedule', 'workflow_dispatch']:
        assert not gate.should_run(event, False, board, now)[0]
    assert gate.should_run('workflow_dispatch', True, board, now)[0]


@pytest.mark.parametrize('timestamp', ['2026-09-07T11:00:00Z', '2026-09-08T10:29:00Z',
                                       '2026-09-09T11:00:00Z', '2026-09-08T11:01:00Z',
                                       '2026-09-08T10:35:00', 'bad', None])
def test_stale_early_future_or_malformed_publication_retries(timestamp):
    now = datetime.fromisoformat('2026-09-08T11:00:00+00:00')
    assert gate.should_run('schedule', False, board_at(timestamp), now)[0]


@pytest.mark.parametrize('which', ['champion', 'challenger'])
@pytest.mark.parametrize('failure', ['source_failure', 'stale_history', 'missing_timestamp', 'malformed_picks'])
def test_either_model_failure_retries(which, failure):
    board = board_at('2026-09-08T10:35:00Z')
    model = board if which == 'champion' else board['challenger']
    if failure == 'missing_timestamp':
        del model['generated_at']
    elif failure == 'malformed_picks':
        model['recommendations'] = None
    else:
        model['status'] = failure
    assert not gate.healthy_today(board, datetime.fromisoformat('2026-09-08T11:00:00+00:00'))


@pytest.mark.parametrize('board', [None, [], {}, {'challenger': None}])
def test_missing_or_broken_publication_retries(board):
    assert gate.should_run('schedule', False, board, datetime.fromisoformat('2026-09-08T11:00:00+00:00'))[0]
