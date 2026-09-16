"""Exact-price evidence for early NFL attempts offers, independent of QB roles.

The built-in live adapter is FanDuel's public New York sportsbook feed.
Validated collectors may supply timestamped comparison evidence explicitly.
A fresh comparison-page retrieval is deliberately not evidence of availability.
Missing markets, IDs, statuses, or contradictory prices never verify a quote.
Original quotes/prices are immutable; evidence is added in quote_verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from .odds_sources import canonical_book, normalize_team, _name_key

FD_API = 'https://sbapi.ny.sportsbook.fanduel.com/api'
FD_KEY = 'FhMFpcPWXMeyZxOx'  # public site configuration, not an account secret
MAX_AGE_SECONDS = 1800
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json',
           'Referer': 'https://sportsbook.fanduel.com/', 'Cache-Control': 'no-cache'}
TEAM_NAMES = dict(zip(
    ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills','Carolina Panthers',
     'Chicago Bears','Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos',
     'Detroit Lions','Green Bay Packers','Houston Texans','Indianapolis Colts','Jacksonville Jaguars',
     'Kansas City Chiefs','Las Vegas Raiders','Los Angeles Chargers','Los Angeles Rams','Miami Dolphins',
     'Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants','New York Jets',
     'Philadelphia Eagles','Pittsburgh Steelers','San Francisco 49ers','Seattle Seahawks','Tampa Bay Buccaneers',
     'Tennessee Titans','Washington Commanders'],
    'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LA MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS'.split()))


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (ValueError, TypeError):
        return None


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and not isinstance(value, bool) else None
    except (ValueError, TypeError):
        return None


def _event(event):
    parts = str(event.get('name', '')).split(' @ ')
    if len(parts) != 2 or any(p not in TEAM_NAMES for p in parts):
        return None
    kick = _time(event.get('openDate'))
    if kick is None:
        return None
    return {'away_team': TEAM_NAMES[parts[0]], 'home_team': TEAM_NAMES[parts[1]], 'kickoff': kick.isoformat()}


def _same_game(q, offer):
    qkick, okick = _time(q.get('kickoff')), _time(offer.get('kickoff'))
    if any(normalize_team(x.get(k)) not in TEAM_NAMES.values() for x in (q, offer) for k in ('home_team', 'away_team')):
        return False
    return (qkick is not None and okick is not None and abs((qkick-okick).total_seconds()) <= 600
            and all(normalize_team(q.get(k)) == normalize_team(offer.get(k)) for k in ('home_team', 'away_team')))


def parse_fanduel(payload, observed_at, proof_path=None):
    """Return evidence rows, including unavailable selections for diagnostics.

    Only explicit full-game ordinary O/U market titles are accepted. Alt ladders,
    special offers, period props, and similarly named passing yards are excluded.
    """
    attachments = payload.get('attachments', {})
    events = attachments.get('events', {})
    rows = []
    for market in attachments.get('markets', {}).values():
        match = re.fullmatch(r'(.+?)\s+-\s+(?:Pass|Passing) Attempts', str(market.get('marketName', '')), re.I)
        if match is None:
            continue
        event = events.get(str(market.get('eventId')), {})
        game = _event(event)
        if game is None:
            continue
        for runner in market.get('runners', []):
            name = str(runner.get('runnerName', ''))
            side_match = re.fullmatch(r'(?:' + re.escape(match[1]) + r'\s+)?(Over|Under)(?:\s+\d+(?:\.\d+)?)?', name, re.I)
            if side_match is None:
                continue
            odds = _number(runner.get('winRunnerOdds', {}).get('americanDisplayOdds', {}).get('americanOdds'))
            line = _number(runner.get('handicap'))
            # Do not guess a missing handicap from the market or runner title.
            if odds is None or abs(odds) < 100 or line is None or line < 0:
                continue
            available = (market.get('marketStatus') == 'OPEN' and market.get('inPlay') is False
                         and runner.get('runnerStatus') == 'ACTIVE')
            rows.append({**game, 'player': match[1], 'book': 'fanduel', 'line': line,
                         'side': side_match[1].lower(), 'odds': odds, 'available': available,
                         'market_status': market.get('marketStatus'), 'runner_status': runner.get('runnerStatus'),
                         'provider_event_id': str(market.get('eventId')),
                         'market_id': str(market.get('marketId') or ''),
                         'selection_id': str(runner.get('selectionId') or ''),
                         'source': 'fanduel_direct', 'evidence_level': 'sportsbook_direct',
                         'jurisdiction': 'US-NY', 'source_updated_at': None,
                         'observed_at': str(observed_at), 'proof_path': str(proof_path) if proof_path else None})
    return rows


def _reason(offer, now):
    if (_number(offer.get('line')) is None or _number(offer.get('odds')) is None
            or abs(float(offer['odds'])) < 100 or offer.get('side') not in ('over', 'under')):
        return 'invalid_line_or_price'
    observed = _time(offer.get('observed_at'))
    if observed is None or not 0 <= (now-observed).total_seconds() <= MAX_AGE_SECONDS:
        return 'stale_or_future_observation'
    kick = _time(offer.get('kickoff'))
    if kick is None or kick <= now:
        return 'game_started_or_unknown'
    if offer.get('available') is not True:
        return 'selection_unavailable_or_status_unknown'
    if not offer.get('market_id') or not offer.get('selection_id'):
        return 'missing_market_or_selection_id'
    if offer.get('jurisdiction') != 'US-NY':
        return 'jurisdiction_unverified'
    level = offer.get('evidence_level')
    if level not in ('sportsbook_direct', 'timestamped_comparison'):
        return 'unsupported_evidence'
    if level == 'timestamped_comparison':
        updated = _time(offer.get('source_updated_at'))
        if updated is None or not 0 <= (now-updated).total_seconds() <= MAX_AGE_SECONDS:
            return 'missing_stale_or_future_source_timestamp'
    return None


def verify_against_offers(quotes, offers, now):
    """Pure matcher. A paired quote needs both exact sides from one evidence market.

    Timestamped comparison evidence is supported for a future validated adapter;
    it is explicitly not described as independent sportsbook corroboration.
    """
    now = _time(now)
    if now is None:
        raise ValueError('now must be timezone-aware')
    enriched = []
    for q in quotes:
        candidates = [o for o in offers if canonical_book(q.get('book')) == canonical_book(o.get('book'))
                      and _name_key(q.get('player')) == _name_key(o.get('player')) and _same_game(q, o)]
        sides = {}
        issues = set()
        for side in ('over', 'under'):
            matches = []
            for o in candidates:
                if o.get('side') != side:
                    continue
                if _number(q.get('line')) != _number(o.get('line')) or _number(q.get(side+'_odds')) != _number(o.get('odds')):
                    issues.add('line_or_price_changed')
                    continue
                reason = _reason(o, now)
                if reason:
                    issues.add(reason)
                else:
                    matches.append(o)
            sides[side] = matches
        pair = next(((over, under) for over in sides['over'] for under in sides['under']
                     if all(over.get(k) == under.get(k) for k in ('source', 'market_id', 'provider_event_id', 'observed_at'))), None)
        # Contradictory observations from the same source/selection at one capture
        # are not safe to resolve by whichever one happens to appear first.
        if pair:
            for chosen in pair:
                conflicts = [o for o in candidates if all(o.get(k) == chosen.get(k) for k in
                             ('source', 'market_id', 'selection_id', 'observed_at'))
                             and (o.get('odds') != chosen.get('odds') or o.get('available') != chosen.get('available'))]
                if conflicts:
                    issues.add('conflicting_source_quotes')
                    pair = None
                    break
        verified = pair is not None
        if not verified and not issues:
            issues.add('matching_paired_market_not_found')
        evidence = {'status': 'verified' if verified else 'unverified', 'checked_at': now.isoformat(),
                    'reason': 'exact_open_paired_quote' if verified else '; '.join(sorted(issues)),
                    'evidence_level': pair[0]['evidence_level'] if verified else None,
                    'source': pair[0]['source'] if verified else None,
                    'jurisdiction': 'US-NY' if verified else None,
                    'source_updated_at': min((o['source_updated_at'] for o in pair if o.get('source_updated_at')), default=None) if verified else None,
                    'observed_at': pair[0]['observed_at'] if verified else None,
                    'market_id': pair[0]['market_id'] if verified else None,
                    'observed_candidates': candidates if not verified else [],
                    'sides': {side: row for side, row in zip(('over','under'), pair)} if verified else {},
                    'original_provenance': {k: q.get(k) for k in ('source','source_url','observed_at','source_updated_at','provider_event_id','event_id')}}
        enriched.append({**q, 'quote_verification': evidence})
    return enriched


def _fetch(url, params, directory, label):
    """Retain response evidence even when a public endpoint fails."""
    path = directory / (label + '.json')
    requested_at = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.get(url, params={**params, '_verification': uuid4().hex}, headers=HEADERS, timeout=15)
        captured = datetime.now(timezone.utc).isoformat()
        body = response.content
        path.write_bytes(body)
        metadata = {'url': response.url, 'requested_at': requested_at, 'observed_at': captured,
                    'status_code': response.status_code, 'sha256': hashlib.sha256(body).hexdigest(),
                    'http_date': response.headers.get('Date'), 'http_age': response.headers.get('Age')}
        path.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2))
        response.raise_for_status()
        age = response.headers.get('Age')
        if age is not None and (not age.isdigit() or int(age) > 120):
            raise ValueError('stale or invalid HTTP cache age')
        return response.json(), captured, path
    except Exception as exc:
        path.with_suffix('.error.json').write_text(json.dumps({'url': url, 'requested_at': requested_at,
                                                              'error': str(exc)}, indent=2))
        raise


def verify_quotes(quotes, now, output_dir, evidence_offers=None):
    """Enrich resolved paired quotes without changing policy, roles, or prices.

    Deduplicates event requests, bounds threads/timeouts, and saves raw proofs in
    a unique subdirectory. Additional evidence_offers must use the same exact
    side-level schema as parse_fanduel; timestamped comparison rows require their
    own current source_updated_at and explicit availability. Their source and
    jurisdiction remain distinct from FanDuel direct evidence.
    """
    quotes = list(quotes)
    if not quotes:
        return [], []
    if _time(now) is None:
        raise ValueError('now must be timezone-aware')
    directory = Path(output_dir) / ('quote-verification-' + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    errors, offers = [], list(evidence_offers or [])
    fd_quotes = [q for q in quotes if canonical_book(q.get('book')) == 'fanduel']
    if fd_quotes:
        try:
            page, _, _ = _fetch(FD_API+'/content-managed-page',
                                {'page':'CUSTOM','customPageId':'nfl','_ak':FD_KEY,'timezone':'America/New_York'}, directory, 'fanduel-nfl')
            events = page.get('attachments', {}).get('events', {})
            ids = [eid for eid, event in events.items() if _event(event) is not None
                   and any(_same_game(q, _event(event)) for q in fd_quotes)]
            if len(ids) > 32:
                raise ValueError('unexpected number of matching NFL events')
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_fetch, FD_API+'/event-page',
                           {'_ak':FD_KEY,'eventId':eid,'tab':'passing-props'}, directory, 'fanduel-event-'+str(eid)): eid for eid in ids}
                for future in as_completed(futures):
                    eid = futures[future]
                    try:
                        payload, captured, path = future.result()
                        offers.extend(parse_fanduel(payload, captured, path))
                    except Exception as exc:
                        errors.append(f'FanDuel NY event {eid} verification failed: {exc}')
        except Exception as exc:
            errors.append(f'FanDuel NY discovery failed: {exc}')
    # Evaluate at completion, since the caller's scoring cutoff can precede the
    # HTTP observation by several seconds. Never use an old caller timestamp to
    # mislabel a newly fetched observation as being from the future.
    checked_at = max(_time(now), datetime.now(timezone.utc))
    result = verify_against_offers(quotes, offers, checked_at)
    directory.joinpath('verification-summary.json').write_text(json.dumps({
        'checked_at': checked_at.isoformat(), 'quotes': len(quotes), 'offers': len(offers),
        'verified': sum(q['quote_verification']['status']=='verified' for q in result),
        'errors': errors,
        'evidence_sources': sorted({str(o.get('source')) for o in offers}),
        'unsupported_books': sorted({canonical_book(q.get('book')) for q in quotes}
                                    - {'fanduel'} - {canonical_book(o.get('book')) for o in offers})}, indent=2))
    return result, errors
