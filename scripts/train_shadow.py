"""Build frozen research artifact without modifying production models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qb_attempts.shadow import prepare_archive, train_shadow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offers", type=Path, default=Path("/Users/samuel/Projects/nfl-qb-attempts-edge/data/bettingpros/offers.parquet"))
    parser.add_argument("--features", type=Path, default=ROOT/"data/features.parquet")
    parser.add_argument("--oof", type=Path, default=ROOT/"data/model_v2_research/oof_predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"data/shadow")
    parser.add_argument("--as-of", default=pd.Timestamp.now(tz="UTC").isoformat())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        sources = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in [("offers", args.offers), ("features", args.features), ("oof", args.oof)]}
        oof = pd.read_csv(args.oof)
        archive, diagnostics = prepare_archive(pd.read_parquet(args.offers), pd.read_parquet(args.features), oof, args.as_of)
        diagnostics["sources"] = sources
        artifact, evaluation = train_shadow(archive, oof, args.as_of, diagnostics)
        archive.to_json(args.output/"joined_archive.json", orient="records", date_format="iso", indent=2)
        evaluation.to_json(args.output/"evaluation.json", orient="records", date_format="iso", indent=2)
    except (OSError, ValueError, KeyError) as exc:
        from qb_attempts.shadow import VERSION
        artifact = {"version": VERSION, "generated_at": args.as_of, "status": "insufficient_data",
                    "shadow_only": True, "promotion_allowed": False,
                    "diagnostics": {"reason": str(exc)}, "metrics": {}}
    (args.output/"artifact.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in artifact.items() if k != "residual_history"}, indent=2))


if __name__ == "__main__":
    main()
