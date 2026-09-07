"""Live board: project this week's starting QBs, price every executable quote, rank by EV.

Pipeline
  1. Slate = upcoming REG games (kickoff > now) in the nearest week, from the nflverse schedule
     (which carries current spread/total, rest, roof, coaches).  Missing spread/total -> ESPN
     scoreboard fallback; still missing -> the game is skipped (never an invented 0/45).
  2. Starter per team = nflverse depth chart QB1 (latest snapshot); the QB must also appear on
     the sportsbook board (books only post props for expected starters).  Disagreement -> held.
  3. Features via the SAME History.features() path as training (as_of = now).
  4. Model mean + heteroscedastic scale -> mu_model.
  5. Market center mu_market = median over real books of the mean implied by each book's
     de-vigged price at its own line (distribution inversion).  This uses every book's line,
     unlike exact-line consensus, and is the engine behind cross-book line-shopping value.
  6. mu_final = mu_market + lambda * (mu_model - mu_market); lambda fixed from the market
     backtest (POLICY['lambda']).  Probabilities at each quote from the final distribution.
  7. EV per side per quote; gates; one recommendation per QB (best EV); NY-legal flag.
Outputs output/board_<date>.csv/json + bet card text; appends recommended rows to the ledger.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import History, load_games, team_key
from market import bet_to, devig, ev, fair_odds, implied_center
from model import load_artifact, predict_rows, probabilities
from odds import fetch_all, name_key

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"

# Frozen launch policy. lambda/thresholds come from output/market_backtest_summary.json (see PLAYBOOK);
# do not tune them on the live board.
POLICY = dict(
    version="market-center-v1",
    lam=0.35,             # model weight on the mean; overwritten from models/policy.json (the frozen source of truth)
    dead_zone=1.5,        # |mu_model - mu_market| below this -> use the market center as-is
    min_ev=0.03,          # EV floor on the blended probability
    max_ev=0.15,          # above this something is wrong (stale/live line, wrong player)
    min_books=3,          # real books needed to form a market center
    min_odds=-250,        # never lay more than -250
    max_quote_age_h=36,   # book update older than this -> held (stale line risk)
    min_qb_games=0,       # rookies allowed: shrinkage handles them; flagged in reasons
    max_center_gap=6.0,   # |mu_model - mu_market| beyond this -> held for review (tree/shrinkage compression lesson)
    max_per_game=1,
    ny_only=True,         # recommendations only at NY-licensed books; watchlist shows every book
)
NY_LEGAL = {"draftkings", "fanduel", "betmgm", "caesars", "betrivers", "fanatics", "thescorebet", "ballybet"}


def load_policy():
    p = ROOT / "models" / "policy.json"
    if p.exists():
        POLICY.update(json.loads(p.read_text()))
    return POLICY


def current_slate(games: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """All not-yet-kicked games of the nearest upcoming week (a week spans Thu -> Mon night)."""
    up = games[(games.kickoff > now) & (games.kickoff <= now + pd.Timedelta(days=12))]
    if up.empty:
        return up
    first = up.sort_values("kickoff").iloc[0]
    return up[(up.week == first.week) & (up.season == first.season)].copy()


def starters(slate: pd.DataFrame) -> dict:
    d = pd.read_parquet(DATA / "raw" / "depth_charts_2026.parquet")
    d = d[d.pos_abb == "QB"].copy()
    d["dt"] = pd.to_datetime(d.dt)
    d["team"] = d.team.map(team_key)
    latest = d.groupby("team").dt.transform("max")
    d = d[(d.dt == latest) & (d.pos_rank == 1)]
    return {r.team: dict(player_id=r.gsis_id, player=r.player_name, dt=r.dt.isoformat()) for r in d.itertuples()}


def build_requests(slate, starters_map, now):
    reqs = []
    for g in slate.to_dict("records"):
        if pd.isna(g["spread_line"]) or pd.isna(g["total_line"]):
            print(f"  skip {g['game_id']}: no spread/total in schedule")
            continue
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = g[f"{side}_team"], g[f"{other}_team"]
            st = starters_map.get(team)
            if not st:
                continue
            reqs.append(dict(game_id=g["game_id"], season=int(g["season"]), week=int(g["week"]), kickoff=g["kickoff"], as_of=now, team=team, opponent=opp,
                             player_id=st["player_id"], player=st["player"], home=int(side == "home" and g["location"] != "Neutral"),
                             rest=float(g[f"{side}_rest"]) if pd.notna(g[f"{side}_rest"]) else 7.0, coach=g[f"{side}_coach"], opp_coach=g[f"{other}_coach"],
                             dome=int(g["roof"] in ("dome", "closed")), div_game=int(g["div_game"]),
                             exp_margin=float(g["spread_line"]) if side == "home" else -float(g["spread_line"]), total_line=float(g["total_line"])))
    return reqs


def main(record=True):
    load_policy()
    now = pd.Timestamp.now(tz="UTC")
    games = load_games()
    slate = current_slate(games, now)
    if slate.empty:
        print("No upcoming games in the next 8 days."); return
    season, week = int(slate.season.iloc[0]), int(slate.week.iloc[0])
    print(f"Slate: {season} week {week}, {len(slate)} games, now {now:%Y-%m-%d %H:%M}Z")
    art = load_artifact()
    records = pd.read_parquet(DATA / "records.parquet")
    st = starters(slate)
    reqs = build_requests(slate, st, now)
    feats = History(records).features(reqs)
    mu, sf = predict_rows(art, feats)
    feats["mu_model"], feats["sf"] = mu, sf
    resid = art["residuals"]
    print("Fetching odds...")
    quotes = fetch_all(season, week)
    if quotes.empty:
        print("No quotes."); return
    quotes["nk"] = quotes.player.map(name_key)
    feats["nk"] = feats.player.map(name_key)
    quotes["team"] = quotes.team.map(team_key)
    q = quotes.drop(columns=["kickoff"]).rename(columns={"player": "book_player"}).merge(
        feats[["game_id", "player_id", "player", "nk", "team", "kickoff", "mu_model", "sf", "qb_games", "qb_team_change", "coach_change", "exp_margin", "total_line"]],
        on=["nk", "team"], how="left")
    unmatched = q[q.player_id.isna()]
    if not unmatched.empty:
        print("  board QBs not matched to a projected starter (held):", sorted(unmatched.book_player.unique()))
    q = q[q.player_id.notna()].copy()
    q = q[pd.to_datetime(q.kickoff, utc=True) > now]
    q["p_book_over"] = [devig(o, u) for o, u in zip(q.over_odds, q.under_odds)]
    q["mu_book"] = [implied_center(l, p, resid, s) for l, p, s in zip(q.line, q.p_book_over, q.sf)]
    center = q.groupby(["game_id", "player_id"]).agg(mu_market=("mu_book", "median"), n_books=("book", "nunique"), line_med=("line", "median"),
                                                     line_lo=("line", "min"), line_hi=("line", "max")).reset_index()
    q = q.merge(center, on=["game_id", "player_id"])
    gap = q.mu_model - q.mu_market
    # dead zone: disagreements under 1.5 attempts carried no information in the archive (see policy.json)
    q["mu_final"] = q.mu_market + POLICY["lam"] * gap.where(gap.abs() >= POLICY.get("dead_zone", 0.0), 0.0)
    q["quote_age_h"] = (now - pd.to_datetime(q.source_updated_at, utc=True, errors="coerce")).dt.total_seconds() / 3600
    rows = []
    for r in q.itertuples():
        po, pu, push = probabilities(r.mu_final, resid, r.line, r.sf)
        pmo, pmu, _ = probabilities(r.mu_model, resid, r.line, r.sf)
        pko, pku, _ = probabilities(r.mu_market, resid, r.line, r.sf)
        for side, p, pm, pk, odds, pbook in (("Over", po, pmo, pko, r.over_odds, r.p_book_over), ("Under", pu, pmu, pku, r.under_odds, 1 - r.p_book_over)):
            e = ev(p, odds, push)
            reasons = []
            if e < POLICY["min_ev"]: reasons.append("ev<floor")
            if e > POLICY["max_ev"]: reasons.append("ev>cap:verify")
            if r.n_books < POLICY["min_books"]: reasons.append("thin market")
            if odds < POLICY["min_odds"]: reasons.append("price too short")
            if abs(r.mu_model - r.mu_market) > POLICY["max_center_gap"]: reasons.append("model/market gap>6")
            if pd.notna(r.quote_age_h) and r.quote_age_h > POLICY["max_quote_age_h"]: reasons.append("stale quote")
            if r.qb_team_change or r.coach_change: reasons.append("new team/coach (info)")
            if r.qb_games < 4: reasons.append("<4 NFL starts (info)")
            if POLICY["ny_only"] and r.book not in NY_LEGAL: reasons.append("not NY-legal")
            hard = [x for x in reasons if not x.endswith("(info)")]
            rows.append(dict(season=season, week=week, game_id=r.game_id, player=r.player, player_id=r.player_id, team=r.team, kickoff=r.kickoff,
                             book=r.book, ny_legal=r.book in NY_LEGAL, source=r.source, line=r.line, side=side, odds=int(odds), ev=round(e, 4),
                             p_final=round(p, 4), p_model=round(pm, 4), p_market_center=round(pk, 4), p_book=round(pbook, 4), p_push=round(push, 4),
                             mu_model=round(r.mu_model, 2), mu_market=round(r.mu_market, 2), mu_final=round(r.mu_final, 2), n_books=int(r.n_books),
                             line_med=r.line_med, line_range=f"{r.line_lo}-{r.line_hi}", fair_odds=fair_odds(p / max(1e-9, 1 - push)),
                             bet_to=bet_to(p, push), quote_age_h=round(float(r.quote_age_h), 1) if pd.notna(r.quote_age_h) else None,
                             exp_margin=r.exp_margin, total_line=r.total_line, status="qualified" if not hard else "held", reasons=";".join(reasons),
                             observed_at=r.observed_at, policy=POLICY["version"], model=art["version"]))
    board = pd.DataFrame(rows).sort_values("ev", ascending=False)
    stamp = now.strftime("%Y-%m-%d_%H%M")
    OUT.mkdir(exist_ok=True)
    board.to_csv(OUT / f"board_{stamp}.csv", index=False)
    board.to_csv(OUT / "board_latest.csv", index=False)
    # one recommendation per QB (best qualified EV), at most one per game
    picks, seen_game = [], set()
    for r in board[board.status == "qualified"].sort_values("ev", ascending=False).itertuples():
        if r.game_id in seen_game or any(p.player_id == r.player_id for p in picks):
            continue
        seen_game.add(r.game_id); picks.append(r)
    lines = [f"NFL QB PASS ATTEMPTS — {season} week {week} — generated {now:%Y-%m-%d %H:%M}Z — policy {POLICY['version']} lam={POLICY['lam']}",
             f"{len(feats)} projected starters, {q.player_id.nunique()} on the board, {len(q)} executable quotes from {q.book.nunique()} books", ""]
    lines.append("RECOMMENDED (EV>=%.0f%% on blended prob; best price per QB; one per game)" % (100 * POLICY["min_ev"]))
    lines.append(f"{'QB':18s}{'Tm':4s}{'Side':6s}{'Line':6s}{'Odds':6s}{'Book':12s}{'EV':7s}{'P':6s}{'Pbook':6s}{'Model':7s}{'Mkt':7s}{'Final':7s}{'Bks':4s}{'BetTo':6s} flags")
    for p in picks:
        lines.append(f"{p.player:18s}{p.team:4s}{p.side:6s}{p.line:<6.1f}{p.odds:<6d}{p.book:12s}{p.ev:+.3f} {p.p_final:.3f} {p.p_book:.3f} {p.mu_model:<7.1f}{p.mu_market:<7.1f}{p.mu_final:<7.1f}{p.n_books:<4d}{p.bet_to:<6d} {p.reasons}")
    if not picks:
        lines.append("  (none qualified)")
    lines += ["", "WATCHLIST (best side per QB, all statuses)"]
    best = board.sort_values("ev", ascending=False).drop_duplicates("player_id")
    for p in best.itertuples():
        lines.append(f"{p.player:18s}{p.team:4s}{p.side:6s}{p.line:<6.1f}{p.odds:<6d}{p.book:12s}{p.ev:+.3f} {p.p_final:.3f} {p.p_book:.3f} {p.mu_model:<7.1f}{p.mu_market:<7.1f}{p.mu_final:<7.1f}{p.n_books:<4d}{p.bet_to:<6d} {p.status} {p.reasons}")
    card = "\n".join(lines)
    (OUT / f"bet_card_{stamp}.txt").write_text(card)
    (OUT / "bet_card_latest.txt").write_text(card)
    print(card)
    if record and picks:
        from ledger import record_picks
        record_picks(pd.DataFrame([p._asdict() for p in picks]).drop(columns=["Index"], errors="ignore"), now)
    return board


if __name__ == "__main__":
    main(record="--no-record" not in sys.argv)
