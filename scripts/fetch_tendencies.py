"""Fetch public nflverse PBP and cache compact per-season team aggregates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qb_attempts.tendencies import AGGREGATION_VERSION, PBP_COLUMNS, aggregate_team_games

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
LICENSE = "https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_season(season: int, *, cache: Path, current_season: int, refresh: bool = False) -> tuple[pd.DataFrame | None, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    agg_path = cache / f"team_games_{season}.parquet"
    meta_path = cache / f"team_games_{season}.json"
    raw_path = cache / f"play_by_play_{season}.parquet"
    url = f"{BASE}/play_by_play_{season}.parquet"
    if not refresh and season < current_season and agg_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("aggregation_version") == AGGREGATION_VERSION and meta.get("aggregate_sha256") == sha256(agg_path):
            return pd.read_parquet(agg_path), {**meta, "cached": True}
    observed_at = datetime.now(timezone.utc).isoformat()
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        if response.status_code == 404 and season == current_season:
            return None, {"season": season, "url": url, "observed_at": observed_at, "unavailable": True}
        response.raise_for_status()
        temp = raw_path.with_suffix(".parquet.tmp")
        try:
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            temp.replace(raw_path)
        finally:
            temp.unlink(missing_ok=True)
    fields = set(pq.ParquetFile(raw_path).schema.names)
    pbp = pd.read_parquet(raw_path, columns=[col for col in PBP_COLUMNS if col in fields])
    aggregate = aggregate_team_games(pbp)
    temp_agg = agg_path.with_suffix(".parquet.tmp")
    aggregate.to_parquet(temp_agg, index=False)
    temp_agg.replace(agg_path)
    meta = {"season": season, "url": url, "sha256": sha256(raw_path), "observed_at": observed_at,
            "raw_bytes": raw_path.stat().st_size, "raw_rows": len(pbp), "team_games": len(aggregate),
            "aggregate_sha256": sha256(agg_path), "aggregation_version": AGGREGATION_VERSION,
            "xpass_available_plays": int(aggregate.xpass_plays.sum()), "cached": False}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{season}: {len(aggregate)} team games", flush=True)
    return aggregate, meta


def main():
    now = datetime.now(timezone.utc)
    current_season = now.year - (now.month <= 2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2017)
    parser.add_argument("--end-season", type=int, default=current_season)
    parser.add_argument("--refresh", action="store_true", help="Redownload historical seasons too")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cache, output = ROOT / "data/pbp_cache", ROOT / "data/tendencies"
    output.mkdir(parents=True, exist_ok=True)
    # The ~200KB committed aggregate is a portable historical cache. Its hash
    # and aggregation version must match before reuse; raw PBP stays untracked.
    # Current-season data is always fetched again, including after a prior 404.
    bundled, bundled_meta = None, {}
    bundled_path, bundled_manifest = output / "team_games.parquet", output / "manifest.json"
    if not args.refresh and bundled_path.exists() and bundled_manifest.exists():
        prior = json.loads(bundled_manifest.read_text())
        if prior.get("aggregation_version") == AGGREGATION_VERSION and prior.get("aggregate_sha256") == sha256(bundled_path):
            bundled = pd.read_parquet(bundled_path)
            bundled_meta = {int(meta["season"]): meta for meta in prior.get("files", []) if not meta.get("unavailable")}

    def get_season(season):
        if bundled is not None and season < current_season and season in bundled_meta:
            frame = bundled.loc[bundled.season == season].copy()
            if not frame.empty:
                return frame, {**bundled_meta[season], "cached": True, "cache_source": "verified_combined_aggregate"}
        return fetch_season(season, cache=cache, current_season=current_season, refresh=args.refresh)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(get_season, range(args.start_season, args.end_season + 1)))
    frames = [frame for frame, _ in results if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("No team tendencies available")
    games = pd.concat(frames, ignore_index=True).sort_values(["season", "week", "game_id", "team"])
    if games.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team game aggregates")
    artifact = output / "team_games.parquet"
    temporary = artifact.with_suffix(".parquet.tmp")
    games.to_parquet(temporary, index=False)
    temporary.replace(artifact)
    manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "aggregation_version": AGGREGATION_VERSION,
                "source": "nflverse play-by-play releases", "license": "CC-BY-4.0", "license_url": LICENSE,
                "attribution": "nflverse / nflfastR contributors", "team_games": len(games),
                "aggregate_sha256": sha256(artifact), "files": [meta for _, meta in results],
                "definitions": {
                    "plays": "Run/pass offensive plays; excludes no-plays, kneels, spikes, two-point tries, preseason.",
                    "dropbacks": "qb_dropback, including sacks and scrambles; distinct from pass attempts.",
                    "pass_attempts": "pass_attempt excluding sacks/scrambles; excludes spikes by preference filter; not official total.",
                    "neutral": "Preplay absolute score differential <=7 and game seconds remaining >120.",
                    "lead_trail": "Preplay score differential >7 / <-7, respectively.",
                    "dropback_oe": "Mean(qb_dropback - xpass), fraction units, valid xpass observations only.",
                    "xpass_vintage": "Unknown upstream model training vintage; retrospective research feature, not fully vintage-correct.",
                    "missing_xpass": "dropback_oe is missing when unavailable; neutral rates remain independently observable.",
                    "usage": "Postgame data; shift chronologically before using as a pregame predictor."},
                "documentation_url": "https://nflfastr.com/reference/add_xpass.html"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(games)} team games to {artifact}")


if __name__ == "__main__":
    main()
