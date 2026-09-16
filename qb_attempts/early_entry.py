"""Operational eligibility without altering either frozen research policy.

An expected starter need not await a final lineup. Unknown/questionable roles
warn; fresh, explicit unavailability withdraws current offers. Price evidence is
a separate requirement, applicable days before kickoff as well as on game day.
"""
from copy import deepcopy

import pandas as pd

from .scoring import name_key

ROLE_REASONS = {
    'Roster does not list active QB', 'Starting role not independently verified',
    'QB injury designation needs review', 'Starting context stale or unverified',
    'Adapter: roster does not list active QB', 'Adapter: starting role not independently verified',
    'Adapter: QB injury designation needs review', 'Adapter: starting context stale or unverified',
    'Slate exposure limit', 'One QB per game',
}


def quote_key(row):
    return tuple(row.get(k) for k in ('game_id', 'player_id', 'book', 'line'))


def role_evidence(quote, context, now):
    ctx = context.get(quote.get('team'), {})
    warnings = []
    fresh = False
    try:
        observed, updated = pd.Timestamp(ctx['observed_at']), pd.Timestamp(ctx['source_updated_at'])
        fresh = (ctx.get('available') and observed.tzinfo is not None and updated.tzinfo is not None
                 and now - pd.Timedelta(hours=2) <= observed <= now + pd.Timedelta(minutes=2)
                 and now - pd.Timedelta(hours=48) <= updated <= now + pd.Timedelta(minutes=2)
                 and str(ctx.get('season')) == str(quote.get('season')))
    except (KeyError, ValueError, TypeError):
        pass
    matches = bool(fresh and name_key(ctx.get('starter', '')) == name_key(quote.get('player', '')))
    injuries = [str(x).lower().strip() for x in ctx.get('injuries', [])] if matches else []
    out = any(x in {'out', 'inactive', 'injured reserve', 'suspended'} for x in injuries)
    if not matches:
        warnings.append('Expected starting role uncertain; final confirmation is not required for early entry')
    elif injuries and not out:
        warnings.append('QB injury designation: ' + ', '.join(injuries))
    if quote.get('roster_status') != 'ACT':
        warnings.append('Weekly roster status needs review; it is not final game-day participation evidence')
    return {'status': 'out' if out else 'expected_starter' if matches else 'uncertain',
            'reliable': bool(matches), 'confirmed': False, 'source_url': ctx.get('source_url'),
            'observed_at': ctx.get('observed_at'), 'warnings': warnings}


def select_early_entries(candidates, quotes, context, now, *, model='champion', ledger=None, limit=5):
    """Return verified current card + research watchlist, keeping inputs intact."""
    lookup = {quote_key(q): q for q in quotes}
    rows = []
    for original in candidates:
        row = deepcopy(original)
        q = lookup.get(quote_key(row), {})
        evidence = deepcopy(q.get('quote_verification') or {'status': 'unverified', 'reason': 'No price verification'})
        role = role_evidence(q, context, now)
        reasons = [r for r in row.get('reasons', []) if r not in ROLE_REASONS and r != 'one QB per game']
        model_reasons = reasons.copy()
        # Keep a useful source failure reason rather than overwriting every
        # unsupported/missing market with an absent-timestamp message.
        if evidence.get('status') == 'verified':
            try:
                verified_at = pd.Timestamp(evidence['observed_at'])
                if verified_at.tzinfo is None or not now - pd.Timedelta(minutes=30) <= verified_at <= now:
                    raise ValueError('stale verification')
                if evidence.get('evidence_level') == 'timestamped_comparison':
                    updated = pd.Timestamp(evidence.get('source_updated_at'))
                    if pd.isna(updated) or updated.tzinfo is None or not now - pd.Timedelta(minutes=30) <= updated <= now:
                        raise ValueError('stale source update')
            except (KeyError, ValueError, TypeError):
                evidence['status'] = 'unverified'
                evidence['reason'] = 'Verification or book update is missing, stale or future'
        # Numeric model probabilities, native EV thresholds and other holds stay intact.
        if evidence.get('status') != 'verified':
            reasons.append('Price not verified: ' + str(evidence.get('reason', evidence.get('status'))))
        if role['status'] == 'out':
            reasons.append('Explicit current QB unavailability')
        try:
            if pd.Timestamp(row['kickoff']) <= now:
                reasons.append('Game has started')
        except (KeyError, ValueError, TypeError):
            reasons.append('Invalid kickoff')
        row.update(quote_verification=evidence, role_evidence=role, reasons=reasons,
                   model_reasons=model_reasons, model_status='held' if model_reasons else 'qualifies',
                   price_status=evidence.get('status', 'unverified'),
                   warnings=list(row.get('warnings') or []) + role['warnings'],
                   status='held' if reasons else 'qualified', research_status=original.get('status'))
        row['source'] = q.get('source')
        row['bet_url'] = (q.get('sportsbook_urls') or {}).get(str(row.get('side', '')).lower())
        if any(other.get('game_id') == row.get('game_id') and other.get('player_id') == row.get('player_id')
               and other.get('book') != row.get('book') and other.get('line') == row.get('line')
               and (other.get('quote_verification') or {}).get('status') != 'verified' for other in quotes):
            row['warnings'].append('Market consensus includes unverified comparison-feed prices')
        rows.append(row)
    rank = lambda r: (r['status'] == 'qualified', r['price_status'] == 'verified',
                     r.get('robust_ev') if r.get('robust_ev') is not None else r.get('ev', -1), r.get('ev', -1))
    best = {}
    for row in sorted(rows, key=rank, reverse=True):
        best.setdefault((row.get('game_id'), row.get('player_id')), row)
    entries = [e for e in (ledger or {}).get('entries', []) if e['model'] == model]
    open_count = sum(e.get('settlement', {}).get('result', 'pending') == 'pending' for e in entries)
    occupied = {e['game_id']: e for e in entries}
    selected, games = [], set()
    for row in sorted(best.values(), key=rank, reverse=True):
        if row['status'] != 'qualified':
            continue
        prior = occupied.get(row['game_id'])
        # A refresh can display an existing entry only at its same side/line/player.
        same = prior and all(str(prior.get(k)).lower() == str(row.get(k)).lower() for k in ('player_id', 'side', 'line'))
        if row['game_id'] in games or (prior and not same) or (not prior and open_count >= limit):
            row['status'] = 'held'
            row['reasons'].append('Cumulative game exposure or open-position limit')
            continue
        if not prior:
            open_count += 1
        row['paper_entry_status'] = 'existing_entry_no_additional_stake' if prior else 'new_entry'
        games.add(row['game_id'])
        selected.append(row)
    return selected, sorted(best.values(), key=lambda r: r.get('ev', -1), reverse=True)
