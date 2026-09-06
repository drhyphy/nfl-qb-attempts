from __future__ import annotations
import argparse,hashlib,json,pickle,sys,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qb_attempts.features import load_inputs,game_records
from qb_attempts.odds_sources import fetch_quotes
from qb_attempts.scoring import resolve_quotes,score,POLICY
from qb_attempts.context import fetch_context
from qb_attempts.tracking import grade_history

def json_default(x):
 if hasattr(x,'isoformat'):return x.isoformat()
 if hasattr(x,'item'):return x.item()
 raise TypeError(str(type(x)))
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,default=json_default,allow_nan=False));t.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--refresh-data',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--quotes-file',type=Path);ap.add_argument('--context-file',type=Path);ap.add_argument('--capture-only',action='store_true');args=ap.parse_args()
 if args.refresh_data:subprocess.run([sys.executable,str(ROOT/'scripts/fetch_data.py')],check=True)
 if args.train or not (ROOT/'data/model/model.pkl').exists():subprocess.run([sys.executable,str(ROOT/'scripts/train.py')],check=True)
 s,g=load_inputs(ROOT/'data/raw');now=pd.Timestamp.now(tz='UTC');future=g[g.kickoff>now]
 season=int(future.iloc[0].season) if not future.empty else datetime.now().year
 roster=pd.read_parquet(ROOT/f'data/raw/roster_weekly_{season}.parquet')
 if args.quotes_file:quotes=json.loads(args.quotes_file.read_text());errors=[]
 else:quotes,errors=fetch_quotes(ROOT/'data/raw/odds')
 # Decision timestamp follows completed retrieval, never precedes it.
 now=pd.Timestamp.now(tz='UTC');resolved,resolution_errors=resolve_quotes(quotes,roster,g,now);errors+=resolution_errors
 run_id=now.strftime('%Y%m%dT%H%M%S%fZ')
 if args.capture_only:
  write(ROOT/f'data/published/closing/{run_id}.json',{'observed_at':now.isoformat(),'quotes':resolved,'source_errors':errors});print(json.dumps({'captured':len(resolved),'errors':errors}));return
 if args.context_file:context=json.loads(args.context_file.read_text())
 else:context,context_errors=fetch_context(sorted({q['team'] for q in resolved}),ROOT/'data/raw/context');errors+=context_errors
 with (ROOT/'data/model/model.pkl').open('rb') as f:artifact=pickle.load(f)
 now=pd.Timestamp.now(tz='UTC');run_id=now.strftime('%Y%m%dT%H%M%S%fZ')
 resolved,final_errors=resolve_quotes(quotes,roster,g,now);errors+=final_errors
 recommendations,watch,candidates=score(resolved,artifact,game_records(s,g),context,now)
 history=ROOT/'data/published/history'
 closings=[]
 for p in (ROOT/'data/published/closing').glob('*.json'):closings.extend(json.loads(p.read_text()).get('quotes',[]))
 performance=grade_history(history,s,g,closings)
 board={'generated_at':now.isoformat(),'run_id':run_id,'status':('source_failure' if resolved and not any(c.get('available') for c in context.values()) else 'ok' if resolved else 'no_odds'),'slate':f"{season} · WEEK {min([q['week'] for q in resolved],default=1)}",'quotes':len(resolved),'players':len({q['player_id'] for q in resolved}),'context_coverage':{'verified':sum(bool(c.get('available')) for c in context.values()),'requested':len({q['team'] for q in resolved})},'model_version':artifact['version'],'model_sha256':hashlib.sha256((ROOT/'data/model/model.pkl').read_bytes()).hexdigest(),'policy':POLICY,'recommendations':recommendations,'watchlist':watch,'source_errors':sorted(set(errors)),'validation':artifact['metrics'],'performance':performance,'source_timestamp_note':'observed_at is feed retrieval, not sportsbook tick time; source_updated_at is null unless independently supplied for that book.'}
 # Store the exact first decision before updating the mutable current board.
 dest=history/f'{run_id}.json';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('x') as f:json.dump(board,f,indent=2,default=json_default,allow_nan=False)
 write(ROOT/'data/published/latest.json',board);write(ROOT/'data/published/performance.json',performance)
 write(ROOT/'data/published/evaluated_quotes.json',candidates)
 write(ROOT/'site/public/board.json',board)
 print(json.dumps({'run_id':run_id,'quotes':len(resolved),'players':board['players'],'recommended':[{k:r[k] for k in ['player','side','line','odds','book','mean','ev','robust_ev']} for r in recommendations],'top_held':[{k:r[k] for k in ['player','side','line','odds','book','ev','reasons']} for r in watch[:8]],'source_errors':errors},indent=2))
 if board['status']=='source_failure':sys.exit(2)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  now=datetime.now(timezone.utc);failure={'generated_at':now.isoformat(),'status':'source_failure','model_version':'attempts-v1.0','slate':'NFL','quotes':0,'players':0,'recommendations':[],'watchlist':[],'source_errors':[f'{type(exc).__name__}: {exc}'],'performance':{'settled':0}}
  write(ROOT/'data/published/latest.json',failure);write(ROOT/'site/public/board.json',failure)
  traceback.print_exc();sys.exit(1)
