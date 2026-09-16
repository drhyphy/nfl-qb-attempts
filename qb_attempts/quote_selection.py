"""Choose a current source quote before scoring; never relabel an old price.

One main line per sportsbook/player/event prevents a stale alternate source
from winning the card or giving a book two votes in market consensus.
"""
from datetime import datetime


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return stamp.timestamp() if stamp.tzinfo else float('-inf')
    except (ValueError, TypeError):
        return float('-inf')


def select_current_quotes(quotes):
    selected = {}
    def rank(q):
        evidence = q.get('quote_verification') or {}
        return (evidence.get('status') == 'verified',
                evidence.get('evidence_level') == 'sportsbook_direct',
                q.get('source') == 'bettingpros',
                _time(q.get('observed_at')), _time(q.get('source_updated_at')))
    for quote in quotes:
        key = tuple(quote[k] for k in ('game_id', 'player_id', 'book'))
        if key not in selected or rank(quote) > rank(selected[key]):
            selected[key] = quote
    return sorted(selected.values(), key=lambda q: (str(q['kickoff']), q['player_id'], q['book']))
