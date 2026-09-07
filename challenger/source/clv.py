"""Closing-line-value tracking.

  python3 clv.py snapshot   -> pull current quotes for the live week, append to data/odds/snapshots.csv
  python3 clv.py grade      -> for each ledger decision, take the LAST snapshot before kickoff for the same
                               player, form the market center from real books at that time, and price the
                               decision's exact line/side: clv = p_close_fair * decimal_odds - 1.
Missing close (no snapshot within 3h before kickoff) stays missing. A prior week's quote is never used.
The playbook rule: judge the model by avg CLV before W-L. Positive CLV + losing = variance; negative
CLV + winning = luck.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import load_games
from ledger import LEDGER
from market import devig, implied_center, payout
from model import load_artifact, probabilities
from odds import fetch_all, name_key

ROOT = Path(__file__).resolve().parents[1]
SNAPS = ROOT / "data" / "odds" / "snapshots.csv"
REAL = {"draftkings", "fanduel", "betmgm", "caesars", "hardrock", "thescorebet", "fanatics", "betrivers", "ballybet", "bet365"}


def snapshot():
    now = pd.Timestamp.now(tz="UTC")
    games = load_games()
    up = games[(games.kickoff > now) & (games.kickoff <= now + pd.Timedelta(days=8))]
    if up.empty:
        print("no upcoming games"); return
    week = int(up.sort_values("kickoff").iloc[0].week); season = int(up.iloc[0].season)
    q = fetch_all(season, week)
    if q.empty:
        print("no quotes"); return
    q["season"], q["week"], q["snap_at"] = season, week, now.isoformat()
    SNAPS.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(SNAPS, mode="a", header=not SNAPS.exists(), index=False)
    print(f"snapshot: {len(q)} quotes appended at {now:%Y-%m-%d %H:%M}Z")


def grade():
    if not LEDGER.exists() or not SNAPS.exists():
        print("need ledger and snapshots"); return
    led = pd.read_csv(LEDGER)
    sn = pd.read_csv(SNAPS, parse_dates=["snap_at"])
    sn["nk"] = sn.player.map(name_key)
    art = load_artifact(); resid = art["residuals"]
    if "clv" not in led:
        led["clv"] = np.nan; led["close_center"] = np.nan; led["close_minutes_before"] = np.nan
    for i, r in led.iterrows():
        if pd.notna(r.clv):
            continue
        kick = pd.Timestamp(r.kickoff)
        s = sn[(sn.nk == name_key(r.player)) & (sn.season == r.season) & (sn.week == r.week) & (sn.snap_at < kick) & (sn.snap_at >= kick - pd.Timedelta(hours=3)) & sn.book.isin(REAL)]
        if s.empty:
            continue
        last = s[s.snap_at == s.snap_at.max()]
        sf = 1.0
        centers = [implied_center(l, devig(o, u), resid, sf) for l, o, u in zip(last.line, last.over_odds, last.under_odds)]
        centers = [c for c in centers if pd.notna(c)]
        if len(centers) < 2:
            continue
        mu = float(np.median(centers))
        po, pu, push = probabilities(mu, resid, r.line, sf)
        p = po if r.side == "Over" else pu
        p_cond = p / max(1e-9, 1 - push)
        led.loc[i, "clv"] = p_cond * (1 + payout(int(r.odds))) - 1
        led.loc[i, "close_center"] = mu
        led.loc[i, "close_minutes_before"] = (kick - last.snap_at.iloc[0]).total_seconds() / 60
    led.to_csv(LEDGER, index=False)
    g = led[led.clv.notna()]
    print(f"CLV graded {len(g)} decisions; avg CLV {g.clv.mean():+.3%}; positive rate {(g.clv > 0).mean():.2f}" if len(g) else "no CLV graded yet")


if __name__ == "__main__":
    {"snapshot": snapshot, "grade": grade}[sys.argv[1] if len(sys.argv) > 1 else "snapshot"]()
