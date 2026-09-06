from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qb_attempts.features import load_inputs
from qb_attempts.game_features import records_with_context
from qb_attempts.game_model import VERSION
from qb_attempts.odds_sources import fetch_quotes
from qb_attempts.scoring import resolve_quotes
from qb_attempts.context import fetch_context
from qb_attempts.game_odds import fetch_game_markets
from qb_attempts.tracking import grade_history


def json_default(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(str(type(value)))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, default=json_default, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--quotes-file', type=Path)
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--game-markets-file', type=Path)
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    if args.refresh_data:
        subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_data.py')], check=True)
    stats, games = load_inputs(ROOT / 'data/raw')
    now = pd.Timestamp.now(tz='UTC')
    future = games[games.kickoff > now].sort_values('kickoff')
    season = int(future.iloc[0].season) if not future.empty else now.year - (now.month <= 2)
    roster = pd.read_parquet(ROOT / f'data/raw/roster_weekly_{season}.parquet')
    if not args.capture_only:
        from qb_attempts.scoring_v2 import POLICY, score
        tendencies_path = ROOT / 'data/tendencies/team_games.parquet'
        if args.refresh_data or not tendencies_path.exists():
            subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_tendencies.py')], check=True)
        model_path = ROOT / 'data/model/model.pkl'
        artifact = None
        if model_path.exists():
            with model_path.open('rb') as handle:
                artifact = pickle.load(handle)
        if args.train or artifact is None or artifact.get('version') != VERSION:
            subprocess.run([sys.executable, str(ROOT / 'scripts/train_v2.py'), '--output', str(ROOT / 'data/model')], check=True)
            with model_path.open('rb') as handle:
                artifact = pickle.load(handle)
        if artifact.get('version') != VERSION:
            raise ValueError('Trained artifact version does not match the V2 pipeline')

    if args.quotes_file:
        quotes, errors = json.loads(args.quotes_file.read_text()), []
    else:
        quotes, errors = fetch_quotes(ROOT / 'data/raw/odds')
    now = pd.Timestamp.now(tz='UTC')
    resolved, resolution_errors = resolve_quotes(quotes, roster, games, now)
    errors += resolution_errors
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    if args.capture_only:
        # Closing collection never depends on model training or other feeds.
        destination = ROOT / f'data/published/closing/{run_id}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as handle:
            json.dump({'observed_at': now.isoformat(), 'quotes': resolved, 'source_errors': errors},
                      handle, indent=2, default=json_default, allow_nan=False)
        print(json.dumps({'captured': len(resolved), 'errors': errors}))
        return

    if args.context_file:
        context = json.loads(args.context_file.read_text())
    else:
        context, context_errors = fetch_context(sorted({q['team'] for q in resolved}), ROOT / 'data/raw/context')
        errors += context_errors
    if args.game_markets_file:
        game_markets = json.loads(args.game_markets_file.read_text())
    else:
        game_markets, market_errors = fetch_game_markets(games, ROOT / 'data/raw/game_markets')
        errors += market_errors

    # Final resolution and decision time follow all source retrieval and training.
    now = pd.Timestamp.now(tz='UTC')
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    resolved, final_errors = resolve_quotes(quotes, roster, games, now)
    errors += final_errors
    requested_games = {q['game_id'] for q in resolved}
    valid_markets = {}
    for game_id, market in game_markets.items():
        try:
            observed = pd.Timestamp(market['observed_at'])
            margin, total = float(market['home_expected_margin']), float(market['total_line'])
            valid = (observed.tzinfo is not None and now - pd.Timedelta(minutes=POLICY['max_observation_age_minutes']) <= observed <= now + pd.Timedelta(minutes=2)
                     and math.isfinite(margin) and math.isfinite(total) and abs(margin) <= 40 and 10 < total < 100)
        except (KeyError, ValueError, TypeError):
            valid = False
        if valid:
            valid_markets[game_id] = market
        elif game_id in requested_games:
            errors.append(f'Game market {game_id}: stale or invalid spread/total observation; no V2 evaluation')
    game_markets = valid_markets
    missing_games = sorted(requested_games - set(game_markets))
    errors += [f"Game market {game_id}: missing current paired spread and total; no V2 evaluation" for game_id in missing_games]
    tendencies = pd.read_parquet(tendencies_path)
    records = records_with_context(stats, games, tendencies)
    recommendations, watch, candidates = score(resolved, artifact, records, context, now, game_markets)
    history = ROOT / 'data/published/history'
    closings = []
    for path in (ROOT / 'data/published/closing').glob('*.json'):
        closings.extend(json.loads(path.read_text()).get('quotes', []))
    performance = grade_history(history, stats, games, closings)
    verified_contexts = sum(bool(c.get('available')) for c in context.values())
    verified_markets = len(requested_games & set(game_markets))
    source_failure = bool(resolved) and (not verified_contexts or not verified_markets)
    board = {
        'generated_at': now.isoformat(), 'run_id': run_id,
        'status': 'source_failure' if source_failure else 'ok' if resolved else 'no_odds',
        'slate': f"{season} · WEEK {min([q['week'] for q in resolved], default=1)}",
        'quotes': len(resolved), 'players': len({q['player_id'] for q in resolved}),
        'context_coverage': {'verified': verified_contexts, 'requested': len({q['team'] for q in resolved})},
        'game_market_coverage': {'verified': verified_markets, 'requested': len(requested_games),
                                 'missing_game_ids': missing_games},
        'game_markets': game_markets,
        'model_version': artifact['version'], 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'policy': POLICY, 'recommendations': recommendations, 'watchlist': watch,
        'source_errors': sorted(set(errors)), 'validation': artifact['metrics'], 'performance': performance,
        'source_timestamp_note': 'observed_at is feed retrieval, not sportsbook tick time; source_updated_at is null unless independently supplied for that book.',
    }
    # Preserve the exact decision before changing the mutable current board.
    destination = history / f'{run_id}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(board, handle, indent=2, default=json_default, allow_nan=False)
    # All contemporaneous evaluated offers remain auditable separately from the
    # recommendation history used by first-decision grading.
    evaluations_path = ROOT / f'data/published/evaluations/{run_id}.json'
    evaluations_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluations_path.open('x') as handle:
        json.dump({'generated_at': now.isoformat(), 'run_id': run_id, 'model_version': artifact['version'],
                   'model_sha256': board['model_sha256'], 'policy': POLICY, 'game_markets': game_markets,
                   'quotes': resolved, 'evaluated_quotes': candidates, 'source_errors': board['source_errors']},
                  handle, indent=2, default=json_default, allow_nan=False)
    write(ROOT / 'data/published/latest.json', board)
    write(ROOT / 'data/published/performance.json', performance)
    write(ROOT / 'data/published/evaluated_quotes.json', candidates)
    write(ROOT / 'site/public/board.json', board)
    print(json.dumps({'run_id': run_id, 'quotes': len(resolved), 'players': board['players'],
                      'game_market_coverage': board['game_market_coverage'],
                      'recommended': [{key: row.get(key) for key in ['player', 'side', 'line', 'odds', 'book', 'mean', 'ev', 'robust_ev']} for row in recommendations],
                      'top_held': [{key: row.get(key) for key in ['player', 'side', 'line', 'odds', 'book', 'ev', 'reasons']} for row in watch[:8]],
                      'source_errors': errors}, indent=2))
    if source_failure:
        sys.exit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        now = datetime.now(timezone.utc)
        failure = {'generated_at': now.isoformat(), 'status': 'source_failure', 'model_version': VERSION,
                   'slate': 'NFL', 'quotes': 0, 'players': 0, 'recommendations': [], 'watchlist': [],
                   'source_errors': [f'{type(exc).__name__}: {exc}'], 'performance': {'settled': 0},
                   'game_markets': {}, 'game_market_coverage': {'verified': 0, 'requested': 0}}
        write(ROOT / 'data/published/latest.json', failure)
        write(ROOT / 'site/public/board.json', failure)
        traceback.print_exc()
        sys.exit(1)
