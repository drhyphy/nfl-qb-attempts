from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib,json,requests
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
BASE='https://github.com/nflverse/nflverse-data/releases/download/'
def fetch(item):
    name,url,optional=item
    p=RAW/name
    if p.exists() and name.startswith('stats_player') and int(name.split('_')[-1].split('.')[0])<datetime.now().year:
        return {'name':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cached':True,'url':url}
    r=requests.get(url,timeout=90)
    if optional and r.status_code==404:return {'name':name,'unavailable':True,'url':url}
    r.raise_for_status();tmp=p.with_suffix('.tmp');tmp.write_bytes(r.content);tmp.replace(p)
    return {'name':name,'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'observed_at':datetime.now(timezone.utc).isoformat()}
def main():
    RAW.mkdir(parents=True,exist_ok=True)
    year=datetime.now().year
    items=[(f'stats_player_week_{s}.parquet',BASE+f'stats_player/stats_player_week_{s}.parquet',s==year) for s in range(2017,year+1)]
    items += [('games.parquet',BASE+'schedules/games.parquet',False),(f'roster_weekly_{year}.parquet',BASE+f'weekly_rosters/roster_weekly_{year}.parquet',False)]
    with ThreadPoolExecutor(max_workers=5) as pool: manifest=list(pool.map(fetch,items))
    (RAW/'manifest.json').write_text(json.dumps({'retrieved_at':datetime.now(timezone.utc).isoformat(),'files':manifest},indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
