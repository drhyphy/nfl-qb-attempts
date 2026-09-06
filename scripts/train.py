from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qb_attempts.features import load_inputs,build_dataset
from qb_attempts.model import train_validate
s,g=load_inputs(ROOT/'data/raw');df,_=build_dataset(s,g);df.to_parquet(ROOT/'data/features.parquet',index=False)
a=train_validate(df,ROOT/'data/model');print(json.dumps(a['metrics'],indent=2))
