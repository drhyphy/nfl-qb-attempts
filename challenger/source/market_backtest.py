"""Market backtest against the BettingPros per-book closing archive (2022 wk10 - 2025).

Questions answered, in order of importance:
  1. How good is the closing market center (MAE vs actual) compared with each model candidate
     on the SAME QB-games?  (If the model is far behind, its blend weight must be small.)
  2. Does the model add information on top of the market?  Fit lambda in
        mu_final = mu_market + lambda * (mu_model - mu_market)
     by minimizing CRPS / log-loss on 2022-2024 archive rows; test on 2025 (untouched).
  3. Does cross-book line shopping + the blended distribution earn positive ROI at *closing*
     prices?  Controls: lambda=0 (pure consensus shopping) and market-only per-book price.
     Grades every executable quote; reports ROI by season/side/book/EV-bucket with a
     week-cluster bootstrap CI.  Closing prices are the harshest possible benchmark: a
     bettor who fires at 6:30 a.m. gets earlier, softer prices, so a strategy that is
     merely break-even at close is likely +EV earlier (and vice versa is disqualifying).

Truth = nflverse official attempts (BettingPros `actual` disagrees on ~4% of rows).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from model import fit_scale, scale_factor, probabilities
from market import payout, implied, devig, ev, implied_center
from odds import name_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
TEAM_MAP = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC"}
REAL_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "hardrock", "pinnacle", "thescorebet", "fanatics", "betrivers", "ballybet",
              "bet365", "pointsbet", "sugarhouse", "tipico", "borgata", "betr", "fliff", "bet105"}
NY_LEGAL = {"draftkings", "fanduel", "betmgm", "caesars", "betrivers", "fanatics", "thescorebet", "ballybet"}


def load_archive():
    offers = pd.read_parquet(DATA / "bettingpros" / "offers.parquet")
    offers = offers[offers.position.isin(["QB", None]) | offers.position.isna()]
    offers["team"] = offers.team.map(lambda t: TEAM_MAP.get(t, t))
    offers["nk"] = offers.player.map(name_key)
    offers = offers[offers.main.fillna(True) & offers.active.fillna(True) & ~offers.is_off.fillna(False)]
    # pair sides per book
    offers["scheduled_ts"] = pd.to_datetime(offers.scheduled, errors="coerce").dt.tz_localize("UTC")
    piv = offers.pivot_table(index=["season", "week", "event_id", "home", "visitor", "scheduled_ts", "nk", "player", "book", "book_id"], columns="side",
                             values=["line", "cost", "updated"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv = piv[(piv.line_over == piv.line_under) & piv.cost_over.notna() & piv.cost_under.notna()].copy()
    piv = piv.rename(columns={"line_over": "line", "cost_over": "over_odds", "cost_under": "under_odds"})
    piv["updated"] = piv[["updated_over", "updated_under"]].max(axis=1)
    piv["line"] = piv.line.astype(float)
    piv["over_odds"] = piv.over_odds.astype(int)
    piv["under_odds"] = piv.under_odds.astype(int)
    piv = piv[(piv.over_odds.abs() >= 100) & (piv.under_odds.abs() >= 100)]
    piv["updated_ts"] = pd.to_datetime(piv.updated, errors="coerce").dt.tz_localize("UTC")
    # pregame quotes only: last update between 72h before and the scheduled kickoff (BP touches some rows post-game)
    piv["age_h"] = (piv.scheduled_ts - piv.updated_ts).dt.total_seconds() / 3600
    piv = piv[piv.age_h.between(-0.25, 72)]   # a tick logged within 15 min after scheduled kickoff is the close
    piv = piv.rename(columns={"home": "ev_home", "visitor": "ev_visitor"})
    return piv[["season", "week", "event_id", "ev_home", "ev_visitor", "nk", "player", "book", "line", "over_odds", "under_odds", "updated", "age_h"]]


def attach_truth(q: pd.DataFrame, oof: pd.DataFrame):
    rec = pd.read_parquet(DATA / "records.parquet")
    rec["nk"] = rec.player.map(name_key)
    keys = rec[["season", "week", "team", "nk", "player_id", "game_id", "attempts", "kickoff"]]
    # BettingPros labels a player with his CURRENT team, so match on name + week and require the
    # nflverse team to be one of the event's two teams.
    m = q.merge(keys, on=["season", "week", "nk"], how="left")
    m.loc[m.player_id.notna() & ~((m.team == m.ev_home.map(lambda t: TEAM_MAP.get(t, t))) | (m.team == m.ev_visitor.map(lambda t: TEAM_MAP.get(t, t)))), "player_id"] = np.nan
    unmatched = m[m.player_id.isna()]
    m = m[m.player_id.notna()].copy()
    return m, unmatched


def main():
    OUT.mkdir(exist_ok=True)
    q = load_archive()
    oof_all = pd.read_csv(ROOT / "models" / "oof_predictions.csv", parse_dates=["kickoff"])
    q, unmatched = attach_truth(q, oof_all)
    print(f"archive quotes paired: {len(q)}  unmatched-to-starter quotes: {len(unmatched)} ({unmatched.player.nunique()} names)")
    print("  unmatched names:", sorted(unmatched.player.unique())[:30])
    candidates = sorted(oof_all.candidate.unique())
    # one row per QB-game per candidate prediction
    preds = oof_all.pivot_table(index=["game_id", "player_id"], columns="candidate", values="prediction").reset_index()
    q = q.merge(preds, on=["game_id", "player_id"], how="inner")
    q["p_book_over"] = [devig(o, u) for o, u in zip(q.over_odds, q.under_odds)]
    q["ny_legal"] = q.book.isin(NY_LEGAL)
    q["real_book"] = q.book.isin(REAL_BOOKS)
    q["is_consensus"] = q.book.eq("consensus")

    # residual pool + scale per season, strictly from earlier seasons' OOF of the ridge candidate (shape only)
    shape_cand = "ridge"
    results = {}
    rows = []
    for season, qs in q.groupby("season"):
        prior = oof_all[(oof_all.candidate == shape_cand) & (oof_all.season < season)]
        if len(prior) < 300:
            continue
        resid = prior.residual.to_numpy()
        feats = pd.read_parquet(DATA / "features.parquet")
        prior_f = prior.merge(feats.drop(columns=[c for c in ["prediction", "residual", "candidate"] if c in feats]), on=["game_id", "player_id"], suffixes=("", "_f"))
        sm = fit_scale(prior_f)
        qf = qs.merge(feats, on=["game_id", "player_id"], suffixes=("", "_f"))
        sf = scale_factor(sm, qf)
        qf["sf"] = sf
        # implied center per real-book quote
        qf["mu_book"] = [implied_center(l, p, resid, s) for l, p, s in zip(qf.line, qf.p_book_over, qf.sf)]
        rb = qf[qf.real_book & qf.mu_book.notna()]
        center = rb.groupby(["game_id", "player_id"]).agg(mu_market=("mu_book", "median"), n_books=("book", "nunique"),
                                                          line_med=("line", "median"), line_spread=("line", lambda x: x.max() - x.min())).reset_index()
        qf = qf.merge(center, on=["game_id", "player_id"], how="left")
        qf = qf[qf.mu_market.notna()]
        rows.append(qf)
    q = pd.concat(rows, ignore_index=True)
    q.to_parquet(OUT / "market_backtest_quotes.parquet", index=False)
    print("quote age before kickoff (h):", q.age_h.describe().round(1).to_dict())

    one = q.drop_duplicates(["game_id", "player_id"]).copy()   # one row per QB-game for center-level metrics
    print(f"\nQB-games with market center: {len(one)}; by season {one.groupby('season').size().to_dict()}")

    # ---- 1. point accuracy on matched QB-games
    acc = {}
    for s, g in one.groupby("season"):
        d = dict(n=len(g), market_mae=float((g.attempts - g.mu_market).abs().mean()), median_line_mae=float((g.attempts - g.line_med).abs().mean()))
        for c in candidates:
            d[f"{c}_mae"] = float((g.attempts - g[c]).abs().mean())
        acc[int(s)] = d
    acc_df = pd.DataFrame(acc).T
    print("\nMAE vs actual, matched QB-games (market center = median implied mean across real books):")
    print(acc_df.round(3).to_string())

    # ---- 2. lambda fit (dev = seasons < 2025), CRPS-based, per candidate
    dev, test = one[one.season < 2025], one[one.season == 2025]
    lam_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)

    def blend_mae(g, c, lam):
        return float((g.attempts - (g.mu_market + lam * (g[c] - g.mu_market))).abs().mean())

    lam_table = {}
    for c in candidates:
        dev_mae = {float(l): blend_mae(dev, c, l) for l in lam_grid}
        best = min(dev_mae, key=dev_mae.get)
        lam_table[c] = dict(best_lambda_dev=best, dev_mae_at_best=dev_mae[best], dev_mae_market=dev_mae[0.0],
                            test_mae_market=blend_mae(test, c, 0.0), test_mae_at_dev_best=blend_mae(test, c, best),
                            test_mae_curve={float(l): blend_mae(test, c, l) for l in lam_grid})
        print(f"\n{c}: dev-best lambda {best:.2f}  dev MAE {dev_mae[best]:.4f} vs market {dev_mae[0.0]:.4f} | "
              f"2025 MAE at that lambda {lam_table[c]['test_mae_at_dev_best']:.4f} vs market {lam_table[c]['test_mae_market']:.4f}")
    # lambda by disagreement bin and by week bucket (dev rows): does the model deserve more/less weight
    # when it disagrees a lot, or early in the season?
    diag = {}
    for c in candidates:
        d = dev.copy(); d["gap"] = (d[c] - d.mu_market).abs()
        d["gap_bin"] = pd.cut(d.gap, [0, 1.5, 3, 5, 100], labels=["0-1.5", "1.5-3", "3-5", "5+"])
        d["wk_bin"] = pd.cut(d.week, [0, 3, 8, 18], labels=["wk1-3", "wk4-8", "wk9+"])
        out = {}
        for col in ("gap_bin", "wk_bin"):
            for b, g in d.groupby(col, observed=True):
                curve = {float(l): blend_mae(g, c, l) for l in lam_grid}
                bl = min(curve, key=curve.get)
                x = (g[c] - g.mu_market).to_numpy(); y = (g.attempts - g.mu_market).to_numpy()
                out[str(b)] = dict(n=int(len(g)), best_lambda=bl, mae_gain=round(curve[0.0] - curve[bl], 4), ols_slope=round(float((x * y).sum() / max(1e-9, (x * x).sum())), 3))
        diag[c] = out
        print(f"  {c} lambda by |gap| / week bucket:", out)

    # also a regression view: actual - mu_market ~ (model - mu_market), no intercept, dev rows
    reg = {}
    for c in candidates:
        x = (dev[c] - dev.mu_market).to_numpy(); y = (dev.attempts - dev.mu_market).to_numpy()
        reg[c] = float((x * y).sum() / (x * x).sum())
    print("\nOLS slope of (actual - market) on (model - market), dev seasons:", {k: round(v, 3) for k, v in reg.items()})

    # ---- 3. strategy backtest at closing prices, per candidate & lambda, all real-book quotes
    def strategy(qq, c, lam, resid_by_season, thr=0.03, max_ev=0.15, min_books=3, max_age_h=None):
        out = []
        qq = qq[qq.n_books >= min_books]
        if max_age_h is not None:
            qq = qq[qq.age_h.notna() & (qq.age_h <= max_age_h)]
        for s, g in qq.groupby("season"):
            resid = resid_by_season[s]
            mu = g.mu_market + lam * (g[c] - g.mu_market) if c else g.mu_market
            for (i, r), m in zip(g.iterrows(), mu):
                po, pu, push = probabilities(m, resid, r.line, r.sf)
                eo, eu = ev(po, r.over_odds, push), ev(pu, r.under_odds, push)
                side, e, p, odds = ("Over", eo, po, r.over_odds) if eo >= eu else ("Under", eu, pu, r.under_odds)
                if e < thr or e > max_ev:
                    continue
                won = (r.attempts > r.line) if side == "Over" else (r.attempts < r.line)
                pushed = r.attempts == r.line
                pnl = 0.0 if pushed else (payout(odds) if won else -1.0)
                out.append(dict(season=s, week=r.week, game_id=r.game_id, player=r.player, book=r.book, line=r.line, side=side, odds=odds, ev=e, p=p,
                                p_book=r.p_book_over if side == "Over" else 1 - r.p_book_over, actual=r.attempts, won=int(won), push=int(pushed), pnl=pnl,
                                ny_legal=r.ny_legal, line_vs_med=r.line - r.line_med))
        return pd.DataFrame(out)

    resid_by_season = {s: oof_all[(oof_all.candidate == shape_cand) & (oof_all.season < s)].residual.to_numpy() for s in q.season.unique()}
    real = q[q.real_book]

    def summarize(b, label):
        if b.empty:
            print(f"{label}: no bets"); return {}
        by = b.groupby("season").agg(n=("pnl", "size"), roi=("pnl", "mean"), win=("won", "mean")).round(4)
        # week-cluster bootstrap of overall ROI
        wk = b.groupby(["season", "week"]).pnl.agg(["sum", "size"])
        rng = np.random.default_rng(11)
        idx = rng.integers(0, len(wk), (3000, len(wk)))
        boot = wk["sum"].to_numpy()[idx].sum(axis=1) / wk["size"].to_numpy()[idx].sum(axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
        print(f"\n{label}: n={len(b)} ROI={b.pnl.mean():+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] win={b.won.mean():.3f} avgEV={b.ev.mean():.3f}")
        print(by.to_string())
        return dict(n=int(len(b)), roi=float(b.pnl.mean()), roi_ci=[float(ci[0]), float(ci[1])], by_season=by.reset_index().to_dict("records"))

    # ---- probability scoring at each real-book quote: book's own price vs market center vs blend
    def logloss(p, y):
        p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    score = {}
    for s_, g in real.groupby("season"):
        resid = resid_by_season[s_]
        y = (g.attempts > g.line).astype(int).to_numpy(); pushes = (g.attempts == g.line).to_numpy()
        keep = ~pushes
        row = dict(n=int(keep.sum()), book_price=logloss(g.p_book_over.to_numpy()[keep], y[keep]))
        for label, mu in (("market_center", g.mu_market), ("ridge_lam065", g.mu_market + 0.65 * (g.ridge - g.mu_market)),
                          ("ridge_eb_lam065", g.mu_market + 0.65 * (g.ridge_eb - g.mu_market)), ("ridge_lam035", g.mu_market + 0.35 * (g.ridge - g.mu_market)),
                          ("ridge_only", g.ridge)):
            pc = np.array([probabilities(m, resid, l, sfx)[0] / max(1e-9, 1 - probabilities(m, resid, l, sfx)[2]) for m, l, sfx in zip(mu, g.line, g.sf)])
            row[label] = logloss(pc[keep], y[keep])
        score[int(s_)] = row
    print("\nLog loss of P(over) at each real-book quote (lower is better):")
    print(pd.DataFrame(score).T.round(4).to_string())

    strat = {}
    strat["consensus_only_lam0"] = summarize(strategy(real, None, 0.0, resid_by_season), "Consensus shopping only (lambda=0), EV>=3%")
    b0 = strategy(real, None, 0.0, resid_by_season)
    if not b0.empty:
        print(b0.groupby("side").agg(n=("pnl", "size"), roi=("pnl", "mean"), win=("won", "mean")).round(3).to_string())
        print(b0.groupby("book").agg(n=("pnl", "size"), roi=("pnl", "mean")).round(3).to_string())
    strat["consensus_only_fresh6h"] = summarize(strategy(real, None, 0.0, resid_by_season, max_age_h=6), "Consensus shopping only, quotes updated within 6h")
    for c in candidates:
        lam = lam_table[c]["best_lambda_dev"]
        strat[f"{c}_lam{lam}"] = summarize(strategy(real, c, lam, resid_by_season), f"{c} blend lambda={lam} EV>=3%")
    # side/book/ny breakdown for the headline (chosen later) — save the best-dev candidate's bets
    best_c = min(candidates, key=lambda c: lam_table[c]["dev_mae_at_best"])
    bets = strategy(real, best_c, lam_table[best_c]["best_lambda_dev"], resid_by_season)
    bets.to_csv(OUT / "market_backtest_bets.csv", index=False)
    for age in (24, 6):
        strat[f"{best_c}_fresh{age}h"] = summarize(strategy(real, best_c, lam_table[best_c]["best_lambda_dev"], resid_by_season, max_age_h=age),
                                                   f"{best_c} same lambda, only quotes updated within {age}h of kickoff")
    for thr in (0.02, 0.05):
        strat[f"{best_c}_thr{thr}"] = summarize(strategy(real, best_c, lam_table[best_c]["best_lambda_dev"], resid_by_season, thr=thr), f"{best_c} EV floor {thr}")
    if not bets.empty:
        print(f"\nBreakdowns for {best_c} (lambda={lam_table[best_c]['best_lambda_dev']}):")
        for col in ("side", "book", "ny_legal"):
            print(bets.groupby(col).agg(n=("pnl", "size"), roi=("pnl", "mean"), win=("won", "mean")).round(3).to_string())
        bets["ev_bucket"] = pd.cut(bets.ev, [0.03, 0.05, 0.08, 0.12, 0.2])
        print(bets.groupby("ev_bucket", observed=True).agg(n=("pnl", "size"), roi=("pnl", "mean")).round(3).to_string())
        bets["dev"] = bets.season < 2025
        print(bets.groupby("dev").agg(n=("pnl", "size"), roi=("pnl", "mean")).round(4).to_string())

    json.dump(dict(accuracy=acc, lambda_fit=lam_table, lambda_diagnostics=diag, ols_slope=reg, prob_scoring=score, strategies=strat, best_dev_candidate=best_c),
              open(OUT / "market_backtest_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
