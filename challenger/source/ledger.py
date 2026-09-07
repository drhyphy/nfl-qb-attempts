"""Append-only decision ledger + settlement + CLV grading.

Rules (from the WNBA/MLB/Kalshi playbooks):
  * The FIRST recorded recommendation per (season, week, player_id) is the graded decision.  Later
    re-runs may add rows for other books/lines but never overwrite or replace a decision.
  * Settlement uses nflverse official attempts once the week's stats file is published; a listed
    starter with no stat row is graded as 0 attempts only if the game is final AND the team has
    stats (feed gap protection); otherwise stays pending.  DNP-before-kickoff (announced scratch)
    -> void, handled manually via `void` column.
  * CLV = fair prob at the last snapshot before kickoff for the SAME player/line (market center
    based), times decimal odds, minus 1.  Missing close stays missing; never a prior week's quote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from market import payout

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "ledger" / "ledger.csv"
SNAPS = ROOT / "data" / "odds" / "snapshots.csv"


def record_picks(picks: pd.DataFrame, now):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    picks = picks.copy()
    picks["recorded_at"] = now.isoformat()
    picks["result"], picks["actual"], picks["profit_1u"], picks["void"] = None, np.nan, np.nan, 0
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        key = ["season", "week", "player_id"]
        existing = set(map(tuple, old[key].astype(str).to_numpy()))
        picks = picks[[tuple(map(str, k)) not in existing for k in picks[key].to_numpy()]]
        if picks.empty:
            print("ledger: no new decisions (first-decision rule)"); return
        picks.to_csv(LEDGER, mode="a", header=False, index=False)
    else:
        picks.to_csv(LEDGER, index=False)
    print(f"ledger: recorded {len(picks)} decision(s)")


def settle():
    if not LEDGER.exists():
        print("no ledger"); return
    led = pd.read_csv(LEDGER)
    pend = led[led.result.isna() & (led.void != 1)]
    if pend.empty:
        print("ledger: nothing pending"); return led
    games = pd.read_csv(ROOT / "data" / "raw" / "games.csv")
    for season in pend.season.unique():
        f = ROOT / "data" / "raw" / f"stats_player_week_{season}.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f, columns=["player_id", "game_id", "attempts", "season_type"])
        s = s[s.season_type == "REG"]
        team_has = set(s.game_id.unique())
        for i, r in pend[pend.season == season].iterrows():
            g = games[games.game_id == r.game_id]
            if g.empty or pd.isna(g.iloc[0].home_score):
                continue  # not final
            row = s[(s.game_id == r.game_id) & (s.player_id == r.player_id)]
            if row.empty and r.game_id not in team_has:
                continue  # stats not published yet
            actual = float(row.attempts.iloc[0]) if not row.empty else 0.0
            if actual == r.line:
                res, pnl = "push", 0.0
            elif (actual > r.line) == (r.side == "Over"):
                res, pnl = "win", payout(int(r.odds))
            else:
                res, pnl = "loss", -1.0
            led.loc[i, ["result", "actual", "profit_1u"]] = res, actual, pnl
    led.to_csv(LEDGER, index=False)
    return led


def report(led: pd.DataFrame | None = None):
    led = pd.read_csv(LEDGER) if led is None else led
    done = led[led.result.notna() & (led.void != 1)]
    print(f"\nLEDGER: {len(led)} decisions, {len(done)} settled")
    if done.empty:
        return
    w = (done.result == "win").sum(); l = (done.result == "loss").sum(); p = (done.result == "push").sum()
    print(f"  {w}-{l}-{p}  flat P&L {done.profit_1u.sum():+.2f}u  ROI {done.profit_1u.mean():+.1%}  avg EV claimed {done.ev.mean():.3f}")
    for col in ("side", "book", "week"):
        print(done.groupby(col).agg(n=("profit_1u", "size"), pnl=("profit_1u", "sum"), roi=("profit_1u", "mean")).round(3).to_string())
    if "clv" in done and done.clv.notna().any():
        print(f"  avg CLV {done.clv.mean():+.3%} (n={done.clv.notna().sum()}), positive-CLV rate {(done.clv > 0).mean():.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "settle":
        report(settle())
    else:
        report()
