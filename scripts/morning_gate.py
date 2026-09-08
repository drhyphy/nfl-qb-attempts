"""Skip redundant morning runs only after both models are healthy on Pages.

Standard library only: this runs before dependency installation in Actions.
"""
import argparse
import json
import os
from datetime import datetime, time
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo('America/New_York')
BOARD_URL = 'https://drhyphy.github.io/nfl-qb-attempts/board.json'


def healthy_today(board, now):
    """A no-odds day is a completed evaluation; source/model failures are not."""
    now = now.astimezone(EASTERN)
    if not isinstance(board, dict):
        return False
    for model in (board, board.get('challenger')):
        if not isinstance(model, dict) or model.get('status') not in {'ok', 'no_odds'}:
            return False
        if not all(isinstance(model.get(key), list) for key in ('recommendations', 'watchlist')):
            return False
        try:
            generated = datetime.fromisoformat(model['generated_at'].replace('Z', '+00:00'))
            if generated.tzinfo is None:
                return False
            generated = generated.astimezone(EASTERN)
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        if generated.date() != now.date() or generated.time() < time(6, 30) or generated > now:
            return False
    return True


def should_run(event, force, board, now):
    if event == 'workflow_dispatch' and force:
        return True, 'Explicit manual refresh'
    if now.astimezone(EASTERN).time() < time(6, 30):
        return False, 'Before 6:30 a.m. Eastern'
    if healthy_today(board, now):
        return False, 'Both models already published successfully this morning'
    return True, 'Published morning board missing, stale, or unhealthy'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', default=os.getenv('EVENT', 'schedule'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    now = datetime.now(EASTERN)
    board = None
    # A forced refresh and an early UTC candidate need no network lookup.
    if not (args.event == 'workflow_dispatch' and args.force) and now.time() >= time(6, 30):
        try:
            request = Request(f'{BOARD_URL}?morning_check={int(now.timestamp())}',
                              headers={'Cache-Control': 'no-cache', 'User-Agent': 'qb-board-morning-check'})
            with urlopen(request, timeout=20) as response:
                board = json.load(response)
        except Exception as error:
            print(f'Could not verify published board ({type(error).__name__}); allow recovery')
    accepted, reason = should_run(args.event, args.force, board, now)
    print(f'run={str(accepted).lower()}: {reason}')
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'run={str(accepted).lower()}\n')


if __name__ == '__main__':
    main()
