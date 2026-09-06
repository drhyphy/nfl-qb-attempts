from pathlib import Path
import argparse
import json
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qb_attempts.features import load_inputs
from qb_attempts.game_features import records_with_context,game_features
from qb_attempts.game_model import train_validate_v2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'data/model_v2_research')
    args=ap.parse_args()
    stats,games=load_inputs(ROOT/'data/raw')
    tendencies=pd.read_parquet(ROOT/'data/tendencies/team_games.parquet')
    records=records_with_context(stats,games,tendencies)
    features=game_features(records,records.to_dict('records'))
    print('Features ready:',len(features),'rows; valid markets',features[['expected_margin','game_total']].notna().all(axis=1).sum(),flush=True)
    artifact=train_validate_v2(features,args.output)
    print(json.dumps(artifact['metrics'],indent=2))
if __name__=='__main__':main()
