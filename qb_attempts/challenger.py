"""Claude's frozen ridge-EB challenger, with explicit common input/safety adapter.

Prediction/distribution and market-center policy belong to the source model.
Canonical events, paired full-game offers, current game odds and starter/injury
checks belong to this shared deployment adapter. No candidate reselection occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from challenger.engine.features import FEATURES, History, team_key
from challenger.engine.model import predict_rows, probabilities
from challenger.engine.market import bet_to, devig, ev, fair_odds, implied_center
from .odds_sources import NON_SPORTSBOOKS
from .scoring import NY_BOOKS, name_key, payout

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'claude-qbatt-v1.0-adapter1'
ADAPTATIONS = [
    'Shared canonical event/player identity and paired full-game pregame offers; same observation clock as champion.',
    'Shared 30-minute observation/24-hour known quote update checks; current paired spread/total required.',
    'Shared independently verified active starter, injury and context freshness gates; these are adapter rules, not Claude policy.',
    'Selected ridge base refits on causally rebuilt completed-game features, including new seasons; no candidate or policy reselection.',
    'Original saved residual pool and EB corrections remain frozen. Original fit_only residual-update routine is not used.',
    'Original market center includes the target book and all offered lines; identical book/line observations are deduplicated.',
]


class NumericPipeline:
    """Portable StandardScaler + Ridge inference from original numeric parameters."""
    def __init__(self, values):
        self.values = values

    def predict(self, frame):
        v = self.values
        return ((np.asarray(frame, dtype=float) - np.asarray(v['mean'])) / np.asarray(v['scale'])) @ np.asarray(v['coef']) + v['intercept']


def pack_pipeline(model):
    scaler = model.named_steps['standardscaler']
    ridge = model.named_steps['ridge']
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_))


def load_challenger(root: Path | None = None, *, prefer_runtime: bool = True):
    root = Path(root or ROOT / 'challenger')
    runtime = root / 'runtime'
    model_path = runtime / 'model.json'
    records_path = runtime / 'records.parquet'
    use_runtime = prefer_runtime and model_path.exists() and records_path.exists() and (runtime / 'refresh.json').exists()
    if not use_runtime:
        model_path, records_path = root / 'frozen/model.json', root / 'frozen/records.parquet'
    refresh = json.loads((runtime / 'refresh.json').read_text()) if use_runtime else {}
    if use_runtime:
        for path, key in ((model_path, 'model_sha256'), (records_path, 'records_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != refresh.get(key):
                raise ValueError('Incomplete or mismatched challenger runtime refresh')
    serialized = json.loads(model_path.read_text())
    if serialized['name'] != 'ridge_eb' or serialized['base_name'] != 'ridge' or serialized['features'] != FEATURES:
        raise ValueError('Challenger artifact does not match the frozen selected ridge-EB architecture')
    artifact = {**serialized, 'estimator': NumericPipeline(serialized['estimator']), 'residuals': np.asarray(serialized['residuals']),
                'scale': None if serialized['scale'] is None else (NumericPipeline(serialized['scale'][0]), serialized['scale'][1])}
    records = pd.read_parquet(records_path)
    records['kickoff'] = pd.to_datetime(records.kickoff, utc=True)
    return dict(artifact=artifact, records=records, policy=json.loads((root / 'frozen/policy.json').read_text()),
                validation=json.loads((root / 'frozen/validation.json').read_text()),
                provenance=json.loads((root / 'provenance.json').read_text()),
                model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                refresh=refresh,
                runtime_active=use_runtime)


def _timestamp_ok(value, now, age):
    try:
        timestamp = pd.Timestamp(value)
        return bool(pd.notna(timestamp) and timestamp.tzinfo is not None and now-age <= timestamp <= now+pd.Timedelta(minutes=2))
    except (ValueError, TypeError):
        return False


def _shared_reasons(q, context, now):
    reasons = []
    if q.get('roster_status') != 'ACT':
        reasons.append('Adapter: roster does not list active QB')
    ctx = context.get(q['team'], {})
    if not ctx.get('available') or name_key(ctx.get('starter', '')) != name_key(q['player']):
        reasons.append('Adapter: starting role not independently verified')
    if ctx.get('injuries'):
        reasons.append('Adapter: QB injury designation needs review')
    if (not _timestamp_ok(ctx.get('observed_at'), now, pd.Timedelta(hours=2))
            or not _timestamp_ok(ctx.get('source_updated_at'), now, pd.Timedelta(hours=48))
            or str(ctx.get('season', '')) != str(q['season'])):
        reasons.append('Adapter: starting context stale or unverified')
    return reasons


def native_policy_reasons(estimate, price, gap, n_books, quote_age, book, policy):
    """Original hard decision rules, distinct from additional shared safety rules."""
    reasons = []
    if estimate < policy['min_ev']: reasons.append('ev<floor')
    if estimate > policy['max_ev']: reasons.append('ev>cap:verify')
    if n_books < policy['min_books']: reasons.append('thin market')
    if price < policy['min_odds']: reasons.append('price too short')
    if abs(gap) > policy['max_center_gap']: reasons.append('model/market gap>6')
    if quote_age is not None and quote_age > policy['max_quote_age_h']: reasons.append('stale quote')
    if policy['ny_only'] and book not in NY_BOOKS: reasons.append('not NY-legal')
    return reasons


def final_center(mu_model, mu_market, policy):
    gap = mu_model - mu_market
    return mu_market + policy['lam'] * (gap if abs(gap) >= policy.get('dead_zone', 0.0) else 0.0)


def _history_missing(games, bundle, now):
    frozen_through = int(bundle['provenance'].get('historical_records_through', 2025))
    due = games[(games.season > frozen_through) & games.home_score.notna()
                & (games.kickoff + pd.Timedelta(hours=6) < now)]
    processed = set(bundle['refresh'].get('processed_game_ids', []))
    return sorted(set(due.game_id) - processed)


def score_challenger(quotes, games, game_markets, context, now, bundle=None):
    """Score the champion's resolved quote snapshot; never fetch a separate market.

    Invalid identity/freshness/pairing records cannot influence market centers.
    Eligible native selections are then intersected with explicit shared safety
    rules. One QB per game is original policy; there is no five-bet slate cap.
    """
    bundle = bundle or load_challenger()
    artifact, policy = bundle['artifact'], bundle['policy']
    missing_history = _history_missing(games, bundle, now)
    output = dict(status='no_odds' if not quotes else 'ok', version=VERSION,
                  source_model_version=artifact['version'], model_sha256=bundle['model_sha256'],
                  policy=policy, validation=bundle['validation'], recommendations=[], watchlist=[], evaluated_quotes=[],
                  metadata=dict(adaptations=ADAPTATIONS, source_git_commit=bundle['provenance']['source_git_commit'],
                                source_snapshot_at=bundle['provenance']['snapshotted_at'], residual_count=len(artifact['residuals']),
                                calibration='Frozen original residual pool and saved EB corrections',
                                historical_records=len(bundle['records']), trained_through=artifact['trained_through'],
                                refit_at=artifact.get('refit_at'), refresh=bundle['refresh'],
                                missing_history_games=missing_history, shared_input_quotes=len(quotes),
                                market_center_includes_target_book=True, mean_blend_weight=policy['lam'],
                                eligible_quotes=0, excluded_quotes=[]))
    if missing_history:
        output['status'] = 'stale_history'
        return output
    if not quotes:
        return output
    schedule = {g['game_id']: g for g in games.to_dict('records')}
    accepted = {}
    requests = {}
    for q in sorted(quotes, key=lambda x: str(x.get('observed_at', ''))):
        try:
            g = schedule[q['game_id']]
            if q['book'] in NON_SPORTSBOOKS or not q['book']: raise ValueError('not a real sportsbook')
            for side in ('over_odds', 'under_odds'): payout(q[side])
            if not np.isfinite(float(q['line'])) or not 0 <= float(q['line']) <= 100: raise ValueError('invalid line')
            if not q.get('player_id') or team_key(q['team']) not in (g['home_team'], g['away_team']): raise ValueError('identity mismatch')
            side = 'home' if team_key(q['team']) == g['home_team'] else 'away'
            other = 'away' if side == 'home' else 'home'
            if team_key(q['opponent']) != g[f'{other}_team']: raise ValueError('opponent mismatch')
            if team_key(q['home_team']) != g['home_team'] or team_key(q['away_team']) != g['away_team']: raise ValueError('event participants mismatch')
            if abs(pd.Timestamp(q['kickoff']) - g['kickoff']) > pd.Timedelta(minutes=10) or g['kickoff'] <= now: raise ValueError('event time mismatch or started')
            if not _timestamp_ok(q.get('observed_at'), now, pd.Timedelta(minutes=30)): raise ValueError('stale quote observation')
            if q.get('source_updated_at') and not _timestamp_ok(q['source_updated_at'], now, pd.Timedelta(hours=24)): raise ValueError('stale known quote update')
            market = game_markets.get(q['game_id'], {})
            if not _timestamp_ok(market.get('observed_at'), now, pd.Timedelta(minutes=30)): raise ValueError('missing/stale game market')
            margin, total = float(market['home_expected_margin']), float(market['total_line'])
            if not np.isfinite([margin, total]).all() or abs(margin) > 40 or not 10 < total < 100: raise ValueError('invalid game market')
            request = {**q, 'kickoff':g['kickoff'], 'as_of':now, 'exp_margin':margin if side == 'home' else -margin,
                       'total_line':total, 'opp_coach':g[f'{other}_coach'], 'div_game':int(g['div_game']),
                       'home':int(side == 'home' and g['location'] != 'Neutral'), 'coach':g[f'{side}_coach'],
                       'rest':float(g[f'{side}_rest']) if pd.notna(g[f'{side}_rest']) else 7.,
                       'dome':int(g['roof'] in ('dome', 'closed'))}
            requests[(q['game_id'], q['player_id'])] = request
            accepted[(q['game_id'], q['player_id'], q['book'], q['line'])] = q
        except (KeyError, ValueError, TypeError) as error:
            output['metadata']['excluded_quotes'].append(dict(player=q.get('player'), book=q.get('book'), reason=str(error)))
    if not accepted:
        output['status'] = 'source_failure'
        return output
    features = History(bundle['records']).features(list(requests.values()))
    if not np.isfinite(features[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite challenger feature; refusing stale or invented feature fallback')
    features['mu_model'], features['sf'] = predict_rows(artifact, features)
    info = {(r['game_id'], r['player_id']):r for r in features.to_dict('records')}
    centers = {}
    for q in accepted.values():
        key = (q['game_id'], q['player_id'])
        f = info[key]
        mu_book = implied_center(q['line'], devig(q['over_odds'], q['under_odds']), artifact['residuals'], f['sf'])
        if np.isfinite(mu_book): centers.setdefault(key, []).append((q['book'], mu_book))
    rows = []
    for q in accepted.values():
        key = (q['game_id'], q['player_id'])
        if key not in centers: continue
        f = info[key]
        mean_market = float(np.median([mu for _, mu in centers[key]]))
        books = sorted({book for book, _ in centers[key]})
        mean_final = float(final_center(f['mu_model'], mean_market, policy))
        p_final = probabilities(mean_final, artifact['residuals'], q['line'], f['sf'])
        p_model = probabilities(f['mu_model'], artifact['residuals'], q['line'], f['sf'])
        p_market = probabilities(mean_market, artifact['residuals'], q['line'], f['sf'])
        quote_age = ((now-pd.Timestamp(q['source_updated_at'])).total_seconds()/3600) if q.get('source_updated_at') else None
        shared = _shared_reasons(q, context, now)
        warnings = []
        if f['qb_team_change'] or f['coach_change']: warnings.append('new team/coach (info)')
        if f['qb_games'] < 4: warnings.append('<4 NFL starts (info)')
        for i, side in enumerate(('Over', 'Under')):
            odds, p, push = int(q['over_odds' if i == 0 else 'under_odds']), float(p_final[i]), float(p_final[2])
            estimate = float(ev(p, odds, push))
            native = native_policy_reasons(estimate, odds, f['mu_model']-mean_market, len(books), quote_age, q['book'], policy)
            reasons = native + shared
            row = {k:q.get(k) for k in ('game_id','player_id','player','team','opponent','line','book','source_url','observed_at','source_updated_at','season','week')}
            row.update(kickoff=pd.Timestamp(q['kickoff']).isoformat(), side=side, odds=odds,
                       mean=float(f['mu_model']), mu_model=float(f['mu_model']), mu_market=mean_market, mu_final=mean_final,
                       mean_market=mean_market, mean_final=mean_final, scale=float(f['sf']),
                       p_model=float(p_model[i]), p_market=float(p_market[i]), p_final=p, p_push=push,
                       model_p_push=float(p_model[2]), market_p_push=float(p_market[2]),
                       ev=estimate, robust_ev=None, independent_model_ev=float(ev(p_model[i],odds,p_model[2])),
                       market_only_ev=float(ev(p_market[i],odds,p_market[2])),
                       fair_odds=fair_odds(p/max(1e-9,1-push)), bet_to=bet_to(p,push), bet_to_target=.02,
                       interval80=np.clip(f['mu_model']+f['sf']*np.quantile(artifact['residuals'],[.1,.9]),0,90).tolist(),
                       expected_margin=float(f['exp_margin']), game_total=float(f['total_line']),
                       implied_team_points=float((f['total_line']+f['exp_margin'])/2), game_market=game_markets[q['game_id']],
                       model_weight=float(policy['lam'] if abs(f['mu_model']-mean_market)>=policy['dead_zone'] else 0),
                       n_books=len(books), other_books=len(set(books)-{q['book']}), reference_books=books,
                       native_status='qualified' if not native else 'held', native_reasons=native,
                       shared_safety_reasons=shared, status='qualified' if not reasons else 'held', reasons=reasons,
                       warnings=warnings, starting_status=context.get(q['team'],{}),
                       feature_flags={k:f[k] for k in ('coach_change','qb_team_change','qb_games')},
                       model_version=VERSION, policy_version=policy['version'])
            rows.append(row)
    rows.sort(key=lambda r:r['ev'], reverse=True)
    # Source board uses rounded (4dp) EV ordering. Keep exact probabilities in
    # audit output but emulate rounded-EV rank for the original selection rule.
    ranked = sorted(rows, key=lambda r:round(r['ev'],4), reverse=True)
    picks, seen_games, seen_players = [], set(), set()
    for row in ranked:
        if row['status'] != 'qualified': continue
        if row['game_id'] in seen_games or row['player_id'] in seen_players: continue
        seen_games.add(row['game_id']); seen_players.add(row['player_id']); picks.append(row)
    best = {}
    for row in ranked: best.setdefault((row['game_id'],row['player_id']),row)
    output.update(recommendations=picks, watchlist=list(best.values()), evaluated_quotes=rows)
    output['metadata']['eligible_quotes'] = len(accepted)
    output['metadata']['evaluated_players'] = len(best)
    output['metadata']['native_qualified_offers'] = sum(r['native_status']=='qualified' for r in rows)
    output['metadata']['shared_safety_held_offers'] = sum(bool(r['shared_safety_reasons']) for r in rows)
    if rows and all(any('starting role' in reason or 'context stale' in reason for reason in r['shared_safety_reasons']) for r in rows):
        output['status']='source_failure'
    return output
