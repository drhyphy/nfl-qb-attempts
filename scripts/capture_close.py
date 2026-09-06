"""Capture only if an official kickoff is within one hour; never recommend."""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qb_attempts.features import kickoff_utc,team_key
from qb_attempts.odds_sources import fetch_quotes
from qb_attempts.scoring import resolve_quotes
BASE='https://github.com/nflverse/nflverse-data/releases/download/'
def main():
 now=pd.Timestamp.now(tz='UTC');g=pd.read_parquet(BASE+'schedules/games.parquet')
 g=g[g.game_type=='REG'].copy();g['kickoff']=g.apply(kickoff_utc,axis=1)
 for c in ('home_team','away_team'):g[c]=g[c].map(team_key)
 due=g[(g.kickoff>now)&(g.kickoff<=now+pd.Timedelta(minutes=60))]
 if due.empty:print('No kickoffs in next hour; no odds request.');return
 season=int(due.iloc[0].season);r=pd.read_parquet(BASE+f'weekly_rosters/roster_weekly_{season}.parquet')
 q,e=fetch_quotes(ROOT/'data/raw/closing');now=pd.Timestamp.now(tz='UTC')
 q,er=resolve_quotes(q,r,g,now);e+=er
 q=[x for x in q if x['game_id'] in set(due.game_id)]
 p=ROOT/'data/published/closing'/f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json";p.parent.mkdir(parents=True,exist_ok=True)
 def default(x):
  if hasattr(x,'isoformat'):return x.isoformat()
  if hasattr(x,'item'):return x.item()
  raise TypeError()
 with p.open('x') as f:json.dump({'observed_at':now.isoformat(),'quotes':q,'source_errors':e},f,indent=2,default=default)
 print(f'Archived {len(q)} exact-event near-kickoff quotes.')
if __name__=='__main__':main()
