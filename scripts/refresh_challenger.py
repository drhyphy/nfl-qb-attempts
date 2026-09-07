"""Refresh the portable Claude challenger without retuning its chosen model/policy.

The source project's recorded 2017-2025 history stays immutable. Completed new
seasons use the original team aggregation and History feature builder. Only the
selected ridge base is refit; original calibration residuals and EB are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from challenger.engine.features import FEATURES, PBP_COLS, History, game_records, team_game_table, team_key
from challenger.engine.model import fit_ridge
from qb_attempts.challenger import pack_pipeline
from qb_attempts.features import load_inputs


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def refresh_challenger(stats,games,*,root=None,now=None,refresh_data=False,cache=None):
    root=Path(root or ROOT/'challenger');now=now or pd.Timestamp.now(tz='UTC');cache=Path(cache or ROOT/'data/pbp_cache')
    frozen=pd.read_parquet(root/'frozen/records.parquet');frozen['kickoff']=pd.to_datetime(frozen.kickoff,utc=True)
    model=json.loads((root/'frozen/model.json').read_text());through=int(frozen.season.max())
    due=games[(games.season>through)&games.home_score.notna()&(games.kickoff+pd.Timedelta(hours=6)<now)].copy()
    sources=[];parts=[frozen];processed=[]
    for season in sorted(due.season.unique()):
        cache.mkdir(parents=True,exist_ok=True);path=cache/f'play_by_play_{season}.parquet'
        url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
        # The champion's tendency refresh just fetched this same public file.
        # Reuse only a recent on-disk copy, never an old seasonal cache silently.
        recent=path.exists() and now.timestamp()-path.stat().st_mtime<6*3600
        if refresh_data and not recent:
            with requests.get(url,timeout=(30,180),stream=True) as response:
                response.raise_for_status();tmp=path.with_suffix('.parquet.tmp')
                try:
                    with tmp.open('wb') as handle:
                        for chunk in response.iter_content(1024*1024):handle.write(chunk)
                    tmp.replace(path)
                finally:tmp.unlink(missing_ok=True)
        elif not path.exists():
            raise ValueError(f'Missing completed-season PBP {season}; run --refresh-data')
        elif not recent:
            raise ValueError(f'Stale completed-season PBP {season}; run --refresh-data')
        pbp=pd.read_parquet(path,columns=PBP_COLS);pbp=pbp[pbp.season_type=='REG'].copy()
        for col in ('posteam','defteam','home_team','away_team'):pbp[col]=pbp[col].map(team_key)
        season_games=due[due.season==season]
        missing_pbp=set(season_games.game_id)-set(pbp.game_id)
        missing_stats=set(season_games.game_id)-set(stats.game_id)
        if missing_pbp or missing_stats:
            raise ValueError(f'Completed-game history incomplete: PBP {sorted(missing_pbp)}, stats {sorted(missing_stats)}')
        tg=team_game_table(pbp);addition=game_records(stats,season_games,tg)
        if not addition.empty:parts.append(addition)
        processed.extend(season_games.game_id.tolist())
        sources.append(dict(season=int(season),url=url,raw_sha256=sha256(path),raw_bytes=path.stat().st_size,
                            observed_at=pd.Timestamp(path.stat().st_mtime,unit='s',tz='UTC').isoformat()))
    records=pd.concat(parts,ignore_index=True).sort_values(['kickoff','game_id','team']).reset_index(drop=True)
    if records.duplicated(['game_id','player_id']).any():raise ValueError('Duplicate challenger history rows')
    if len(records)>len(frozen):
        features=History(records).features(records.to_dict('records')).sort_values(['kickoff','game_id','team']).reset_index(drop=True)
        training=features[features.season>=2019].copy()
        if not np.isfinite(training[FEATURES].to_numpy(float)).all():raise ValueError('Missing challenger training feature; refusing imputation')
        model['estimator']=pack_pipeline(fit_ridge(training));model['trained_through']=int(training.season.max())
        model['refit_at']=now.isoformat();model['adapted_training_rows']=len(training)
    # Preserve exact original numerical coefficients until new completed games exist.
    runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    tmp=runtime/'records.parquet.tmp';records.to_parquet(tmp,index=False);tmp.replace(runtime/'records.parquet')
    write_json(runtime/'model.json',model)
    metadata=dict(refreshed_at=now.isoformat(),historical_frozen_through=through,records=len(records),
                  processed_game_ids=sorted(processed),sources=sources,model_sha256=sha256(runtime/'model.json'),
                  records_sha256=sha256(runtime/'records.parquet'),
                  calibration_policy='Frozen original residual pool and saved EB corrections; base-only refit, no source fit_only residual update.',
                  base_refitted=bool(len(records)>len(frozen)))
    write_json(runtime/'refresh.json',metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-data',action='store_true');args=parser.parse_args()
    stats,games=load_inputs(ROOT/'data/raw')
    print(json.dumps(refresh_challenger(stats,games,refresh_data=args.refresh_data),indent=2))


if __name__=='__main__':main()
