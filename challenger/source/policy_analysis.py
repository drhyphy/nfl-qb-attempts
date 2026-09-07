"""Policy selection on the archive backtest output. Dev = 2022-2024 chooses; 2025 checks. Writes models/policy.json."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from model import probabilities
from market import ev, payout

ROOT = Path(__file__).resolve().parents[1]
q = pd.read_parquet(ROOT / "output" / "market_backtest_quotes.parquet")
oof = pd.read_csv(ROOT / "models" / "oof_predictions.csv")
resid = {s: oof[(oof.candidate == "ridge") & (oof.season < s)].residual.to_numpy() for s in q.season.unique()}
real = q[q.real_book & (q.n_books >= 3)].copy()
C = "ridge_eb"

def p_over_cond(mu, g):
    out = np.empty(len(g))
    for i, (m, l, s, se) in enumerate(zip(mu, g.line, g.sf, g.season)):
        o, u, p = probabilities(m, resid[se], l, s); out[i] = o / max(1e-9, o + u)
    return out

def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def blended_mu(g, lam, dead=0.0):
    gap = g[C] - g.mu_market
    adj = np.where(gap.abs() < dead, 0.0, gap)
    return g.mu_market + lam * adj

dev, test = real[real.season < 2025], real[real.season == 2025]
print("== lambda by log loss at real-book quotes (pushes excluded) ==")
res = {}
for lam in [0, .2, .3, .35, .4, .45, .5, .55, .6, .7]:
    row = {}
    for nm, g in (("dev", dev), ("2025", test)):
        g2 = g[g.attempts != g.line]; y = (g2.attempts > g2.line).astype(int).to_numpy()
        row[nm] = logloss(p_over_cond(blended_mu(g2, lam), g2), y)
    res[lam] = row; print(f"  lam {lam:<5} dev {row['dev']:.5f}  2025 {row['2025']:.5f}")
lam_ll = min(res, key=lambda l: res[l]["dev"])
print("dev-best lambda by log loss:", lam_ll)

def strat(g, lam, thr_over=0.03, thr_under=0.03, dead=0.0, max_ev=0.15):
    mu = blended_mu(g, lam, dead); rows = []
    for (i, r), m in zip(g.iterrows(), mu):
        po, pu, push = probabilities(m, resid[r.season], r.line, r.sf)
        eo, eu = ev(po, r.over_odds, push), ev(pu, r.under_odds, push)
        if eo >= eu: side, e, odds = "Over", eo, r.over_odds
        else: side, e, odds = "Under", eu, r.under_odds
        thr = thr_over if side == "Over" else thr_under
        if e < thr or e > max_ev: continue
        won = (r.attempts > r.line) if side == "Over" else (r.attempts < r.line); pushed = r.attempts == r.line
        rows.append(dict(season=r.season, week=r.week, side=side, book=r.book, ev=e, pnl=0.0 if pushed else (payout(odds) if won else -1.0), won=int(won), player=r.player, game_id=r.game_id))
    return pd.DataFrame(rows)

def summ(b, label):
    if b.empty: print(label, "no bets"); return {}
    wk = b.groupby(["season", "week"]).pnl.agg(["sum", "size"]); rng = np.random.default_rng(3)
    idx = rng.integers(0, len(wk), (3000, len(wk))); boot = wk["sum"].to_numpy()[idx].sum(1) / wk["size"].to_numpy()[idx].sum(1)
    ci = np.quantile(boot, [.025, .975])
    d = dict(n=len(b), roi=b.pnl.mean(), ci=ci.tolist(), by_season=b.groupby("season").pnl.agg(["size", "mean"]).round(4).to_dict(), by_side=b.groupby("side").pnl.agg(["size", "mean"]).round(4).to_dict())
    print(f"{label}: n={len(b)} ROI {b.pnl.mean():+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] | " + " ".join(f"{s}:{v['mean']:+.3f}(n={v['size']})" for s, v in b.groupby('season').pnl.agg(['size','mean']).iterrows() for _ in [0]) if False else f"{label}: n={len(b)} ROI {b.pnl.mean():+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    print("    by season:", b.groupby("season").pnl.agg(["size", "mean"]).round(3).T.to_dict())
    print("    by side  :", b.groupby("side").pnl.agg(["size", "mean"]).round(3).T.to_dict())
    return d

out = {"lambda_logloss": res, "lambda_ll_best": lam_ll}
print("\n== strategies at closing prices (min_books>=3, EV cap 15%) ==")
for lam in (0.35, 0.45, 0.55):
    for dead in (0.0, 1.5):
        for nm, g in (("DEV", dev), ("2025", test)):
            out[f"lam{lam}_dead{dead}_{nm}"] = summ(strat(g, lam, dead=dead), f"lam={lam} dead={dead} {nm}")
print("\n== side policy: unders 3%, overs need 6% ==")
for nm, g in (("DEV", dev), ("2025", test)):
    out[f"sidepolicy_{nm}"] = summ(strat(g, lam_ll, thr_over=0.06, thr_under=0.03), f"lam={lam_ll} over>=6% under>=3% {nm}")
    out[f"underonly_{nm}"] = summ(strat(g, lam_ll, thr_over=9, thr_under=0.03), f"lam={lam_ll} unders only {nm}")
json.dump(out, open(ROOT / "output" / "policy_analysis.json", "w"), indent=1, default=float)
