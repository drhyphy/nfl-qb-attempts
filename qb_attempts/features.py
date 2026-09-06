from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_MAP={'LAR':'LA','WSH':'WAS','JAC':'JAX','OAK':'LV','SD':'LAC'}
def team_key(x):return TEAM_MAP.get(str(x),str(x))
def kickoff_utc(g):
    return pd.Timestamp(f"{g['gameday']} {g['gametime'] or '13:00'}",tz='America/New_York').tz_convert('UTC')
def load_inputs(raw:Path):
    stats=pd.concat([pd.read_parquet(p) for p in sorted(raw.glob('stats_player_week_*.parquet'))],ignore_index=True)
    stats=stats[stats.season_type=='REG'].copy()
    for c in ['team','opponent_team']:stats[c]=stats[c].map(team_key)
    stats=stats.drop_duplicates(['game_id','player_id'])
    games=pd.read_parquet(raw/'games.parquet');games=games[games.game_type=='REG'].copy()
    for c in ['home_team','away_team']:games[c]=games[c].map(team_key)
    games['kickoff']=games.apply(kickoff_utc,axis=1)
    return stats,games

def game_records(stats,games):
    """Use listed starting QB, including early exits. Never select by realized attempts."""
    totals=stats.groupby(['game_id','team'])[['attempts','carries','sacks_suffered']].sum()
    lookup=stats.set_index(['game_id','player_id'])
    rows=[]
    for g in games.sort_values('kickoff').to_dict('records'):
        if pd.isna(g['home_score']):continue
        for side,other in [('home','away'),('away','home')]:
            key=(g['game_id'],g[side+'_team']);pid=g[side+'_qb_id']
            if key not in totals.index or not isinstance(pid,str):continue
            t=totals.loc[key];pkey=(g['game_id'],pid)
            # A listed starter without a statistical row has zero attempts (early exit).
            s=lookup.loc[pkey] if pkey in lookup.index else None
            rows.append(dict(game_id=g['game_id'],season=int(g['season']),week=int(g['week']),kickoff=g['kickoff'],team=g[side+'_team'],opponent=g[other+'_team'],player_id=pid,player=g[side+'_qb_name'],home=int(side=='home' and g['location']!='Neutral'),rest=float(g[side+'_rest']),coach=g[side+'_coach'],opp_coach=g[other+'_coach'],dome=int(g['roof'] in ('dome','closed')),attempts=float(s['attempts']) if s is not None else 0.,team_attempts=float(t.attempts),plays=float(t.attempts+t.carries+t.sacks_suffered),sack_rate=float(t.sacks_suffered/max(1,t.attempts+t.sacks_suffered)),qb_carries=float(s['carries']) if s is not None else 0.))
    return pd.DataFrame(rows)

FEATURES=['qb_mean5','qb_mean12','qb_sd12','qb_carries5','qb_games','team_attempts5','team_attempts12','team_plays5','team_plays12','team_rate5','team_sack5','opp_attempts5','opp_attempts12','opp_plays5','opp_sack5','home','rest','week','dome','coach_change','qb_team_change','season_games','team_coach_games']

def vector(row,players,teams,defenses):
    ph=players[row['player_id']];th=teams[row['team']];oh=defenses[row['opponent']]
    def avg(h,key,n,default):return float(np.mean([r[key] for r in h[-n:]])) if h else default
    tm=avg(th,'team_attempts',12,33.)
    pmean=avg(ph,'attempts',12,tm)
    # Explicit shrinkage for sparse QB history; no confidence from an imputed rookie.
    n=min(len(ph),12);pmean=(n*pmean+4*tm)/(n+4)
    recent=avg(ph,'attempts',5,pmean);recent=(min(len(ph),5)*recent+3*pmean)/(min(len(ph),5)+3)
    recent_team_rate=avg(th,'team_attempts',5,33.)/max(1,avg(th,'plays',5,64.))
    return dict(qb_mean5=recent,qb_mean12=pmean,qb_sd12=float(np.std([r['attempts'] for r in ph[-12:]])) if len(ph)>2 else 8.,qb_carries5=avg(ph,'qb_carries',5,3.),qb_games=min(len(ph),32),team_attempts5=avg(th,'team_attempts',5,33.),team_attempts12=tm,team_plays5=avg(th,'plays',5,64.),team_plays12=avg(th,'plays',12,64.),team_rate5=recent_team_rate,team_sack5=avg(th,'sack_rate',5,.065),opp_attempts5=avg(oh,'team_attempts',5,33.),opp_attempts12=avg(oh,'team_attempts',12,33.),opp_plays5=avg(oh,'plays',5,64.),opp_sack5=avg(oh,'sack_rate',5,.065),home=row['home'],rest=min(float(row['rest']),21.),week=row['week'],dome=row['dome'],coach_change=int(bool(th) and th[-1]['coach']!=row['coach']),qb_team_change=int(bool(ph) and ph[-1]['team']!=row['team']),season_games=sum(r['season']==row['season'] for r in ph),team_coach_games=sum(r['coach']==row['coach'] for r in th[-17:]))

class History:
    def __init__(self,records):self.records=records
    def features(self,requests):
        """Process kickoff batches; same-game and simultaneous-game labels cannot leak."""
        players=defaultdict(list);teams=defaultdict(list);defenses=defaultdict(list)
        events=sorted(self.records.to_dict('records'),key=lambda r:r['kickoff'])
        output=[];i=0
        for row in sorted(requests,key=lambda r:min(r['kickoff'],r.get('as_of',r['kickoff']))):
            cutoff=min(row['kickoff'],row.get('as_of',row['kickoff']))
            while i<len(events) and events[i]['kickoff']+pd.Timedelta(hours=6)<cutoff:
                r=events[i];players[r['player_id']].append(r);teams[r['team']].append(r);defenses[r['opponent']].append(r);i+=1
            output.append({**row,**vector(row,players,teams,defenses)})
        return pd.DataFrame(output)

def build_dataset(stats,games):
    records=game_records(stats,games)
    return History(records).features(records.to_dict('records')),records
