"""Causal game-market and state-conditioned passing features for V2."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from .features import FEATURES, History, game_records

MARKET_FEATURES = FEATURES + ['expected_margin', 'game_total', 'margin_abs', 'margin_x_team_rate', 'margin_x_qb_carries']
TENDENCY_FEATURES = MARKET_FEATURES + ['neutral_pass12', 'lead_pass12', 'trail_pass12', 'attempts_per_dropback12', 'opponent_neutral_pass12', 'state_pass_spread_interaction', 'tendency_games']

def records_with_context(stats, games, tendencies):
    records = game_records(stats, games)
    look = games.set_index('game_id')
    records['expected_margin'] = [float(look.loc[r.game_id, 'spread_line']) * (1 if r.team == look.loc[r.game_id, 'home_team'] else -1) for r in records.itertuples()]
    records['game_total'] = [float(look.loc[r.game_id, 'total_line']) for r in records.itertuples()]
    if tendencies is not None and not tendencies.empty:
        cols = ['game_id', 'team', 'neutral_plays', 'neutral_dropbacks', 'lead_plays', 'lead_dropbacks', 'trail_plays', 'trail_dropbacks', 'dropbacks', 'pass_attempts', 'plays']
        t = tendencies[cols].rename(columns={'plays': 'pbp_plays'})
        records = records.merge(t, on=['game_id', 'team'], how='left', validate='one_to_one')
    return records


def _tendency(history, prefix, fallback):
    num = sum(float(r.get(prefix + '_dropbacks', 0) or 0) for r in history if pd.notna(r.get(prefix + '_dropbacks')))
    den = sum(float(r.get(prefix + '_plays', 0) or 0) for r in history if pd.notna(r.get(prefix + '_plays')))
    return (num + 120 * fallback) / (den + 120)


def game_features(records, requests):
    """V1 features remain identical; each added rate is based strictly on prior games."""
    base = History(records).features(requests)
    by_team, by_defense = defaultdict(list), defaultdict(list)
    history = sorted(records.to_dict('records'), key=lambda r: r['kickoff'])
    i, out = 0, []
    for row in sorted(base.to_dict('records'), key=lambda r: min(r['kickoff'], r.get('as_of', r['kickoff']))):
        cutoff = min(row['kickoff'], row.get('as_of', row['kickoff']))
        while i < len(history) and history[i]['kickoff'] + pd.Timedelta(hours=6) < cutoff:
            h = history[i]
            if pd.notna(h.get('pbp_plays')):
                by_team[h['team']].append(h)
                by_defense[h['opponent']].append(h)
            i += 1
        th = by_team[row['team']][-12:]
        oh = by_defense[row['opponent']][-12:]
        neutral = _tendency(th, 'neutral', .60)
        lead = _tendency(th, 'lead', .50)
        trail = _tendency(th, 'trail', .72)
        db = sum(float(h['dropbacks']) for h in th)
        attempts = sum(float(h['pass_attempts']) for h in th)
        margin, total = float(row.get('expected_margin', np.nan)), float(row.get('game_total', np.nan))
        out.append({**row, 'expected_margin': margin, 'game_total': total,
                    'margin_abs': abs(margin), 'margin_x_team_rate': margin * (row['team_rate5'] - .5),
                    'margin_x_qb_carries': margin * row['qb_carries5'], 'neutral_pass12': neutral,
                    'lead_pass12': lead, 'trail_pass12': trail,
                    'attempts_per_dropback12': (attempts + 90.) / (db + 100.),
                    'opponent_neutral_pass12': _tendency(oh, 'neutral', .60),
                    'state_pass_spread_interaction': margin * (trail - lead), 'tendency_games': len(th)})
    return pd.DataFrame(out)
