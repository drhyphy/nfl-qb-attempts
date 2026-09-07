"""Pass-attempt mean + distribution model with prespecified candidate selection.

Protocol (fixed before looking at results):
  * History burn-in 2017-2018 (features only). Training rows 2019+.
  * Expanding-season out-of-fold predictions: fit on seasons < S, predict S, for S in 2020..2025.
  * Development window 2022-2024 selects the candidate by CRPS (full-distribution proper score).
    2025 is reported as a holdout but was NOT used for selection.
  * Distribution = point mean + empirical OOF residual pool from *earlier* seasons only, scaled
    per-row by a heteroscedastic scale model (fit on earlier seasons' |residuals|); discrete
    integer support; integer lines keep push mass.
  * Per-QB empirical-Bayes residual correction is a candidate, not a default (WNBA v2.1 lesson:
    helps tail compression, but can chase hot streaks — must earn its place on CRPS).

Candidates:
  recent       : shrunk decayed QB mean (qb_att_d3) — the naive baseline the market surely beats
  ridge        : standardized ridge on all features
  lgbm         : LightGBM, shallow, heavy regularization, monotone constraints on volume features
  ridge_lgbm   : 50/50 blend of the two (different inductive biases; ridge is linear/extrapolates,
                 trees compress extremes)
  ridge_eb     : ridge + per-QB EB correction of recent OOF residuals (K=20 pseudo-games)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
VERSION = "qbatt-v1.0"
TRAIN_START, DEV_START, DEV_END, HOLDOUT = 2019, 2022, 2024, 2025
CANDIDATES = ("recent", "ridge", "lgbm", "ridge_lgbm", "ridge_eb")
MONO_UP = {"qb_att_d3", "qb_att_d8", "qb_att_mean", "tm_att_d3", "tm_att_d8", "tm_plays_d3", "tm_plays_d8", "tm_db_rate_d8",
           "tm_neutral_db_rate_d8", "tm_db_oe_d8", "tm_neutral_db_oe_d8", "total_line", "opp_att_allowed_d8", "opp_plays_allowed_d8",
           "opp_db_rate_allowed_d8", "qb_share_d8"}
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=30.0, max_depth=4, verbose=-1, seed=7, num_threads=4)
LGB_ROUNDS = 350
EB_K = 20.0


def fit_ridge(df):
    m = make_pipeline(StandardScaler(), Ridge(alpha=60.0))
    m.fit(df[FEATURES], df.attempts)
    return m


def fit_lgbm(df):
    mono = [1 if f in MONO_UP else 0 for f in FEATURES]
    ds = lgb.Dataset(df[FEATURES], df.attempts, params={"monotone_constraints": mono})
    return lgb.train(dict(LGB_PARAMS, monotone_constraints=mono), ds, num_boost_round=LGB_ROUNDS)


def predict_base(name, obj, df):
    if name == "recent":
        return df.qb_att_d3.to_numpy().clip(10, 55)
    if name == "ridge":
        return obj.predict(df[FEATURES]).clip(10, 55)
    if name == "lgbm":
        return obj.predict(df[FEATURES]).clip(10, 55)
    if name == "ridge_lgbm":
        return (0.5 * obj[0].predict(df[FEATURES]) + 0.5 * obj[1].predict(df[FEATURES])).clip(10, 55)
    raise ValueError(name)


def fit_base(name, df):
    if name == "recent":
        return None
    if name == "ridge":
        return fit_ridge(df)
    if name == "lgbm":
        return fit_lgbm(df)
    if name == "ridge_lgbm":
        return (fit_ridge(df), fit_lgbm(df))
    raise ValueError(name)


def eb_corrections(oof_prior: pd.DataFrame, k: float = EB_K, half_life: float = 8.0) -> dict:
    """Per-QB decayed mean of out-of-fold residuals, shrunk with K pseudo-games."""
    corr = {}
    for pid, g in oof_prior.sort_values("kickoff").groupby("player_id"):
        r = g.residual.to_numpy()[-24:]
        w = 0.5 ** (np.arange(len(r))[::-1] / half_life)
        corr[pid] = float((w * r).sum() / (w.sum() + k))
    return corr


def apply_eb(pred, df, corr):
    return (pred + df.player_id.map(corr).fillna(0.0).to_numpy()).clip(10, 55)


# ------------------------------------------------------------------ distribution
def fit_scale(oof: pd.DataFrame):
    """Heteroscedastic scale: ridge on log|residual| -> per-row multiplier of the pooled residual sd."""
    y = np.log(np.abs(oof.residual.to_numpy()) + 1.0)
    m = make_pipeline(StandardScaler(), Ridge(alpha=200.0))
    m.fit(oof[FEATURES], y)
    base = float(np.mean(np.exp(m.predict(oof[FEATURES]))))
    return m, base


def scale_factor(scale_model, df):
    if scale_model is None:
        return np.ones(len(df))
    m, base = scale_model
    return (np.exp(m.predict(df[FEATURES])) / base).clip(0.6, 1.6)


_G_CACHE: dict = {}
_XGRID = np.arange(-70.0, 70.0, 0.02)


def _shape_fn(resid, sf=1.0, bw=1.0):
    """Tabulate G(x) = mean_r Phi((x - sf*r)/bw): the CDF of the scaled residual mixture at offset x.
    Cached per (residual pool, rounded sf) so probabilities() is O(1) after the first call."""
    r = np.asarray(resid, dtype=float)
    key = (id(r), len(r), round(float(sf), 2))
    g = _G_CACHE.get(key)
    if g is None:
        if len(_G_CACHE) > 400:
            _G_CACHE.clear()
        rs = np.sort(r) * round(float(sf), 2)
        # mixture CDF on a grid; chunk to bound memory
        g = np.empty(len(_XGRID))
        for i in range(0, len(_XGRID), 1000):
            x = _XGRID[i:i + 1000]
            g[i:i + 1000] = norm.cdf((x[:, None] - rs[None, :]) / bw).mean(axis=1)
        _G_CACHE[key] = g
    return g


def cdf_grid(mu, resid, sf=1.0, bw=1.0, kmax=90):
    """Discrete CDF over integer counts 0..kmax from a smoothed, scaled empirical residual mixture."""
    g = _shape_fn(resid, sf, bw)
    ks = np.arange(0, kmax + 1)
    c = np.interp(ks + 0.5 - mu, _XGRID, g, left=0.0, right=1.0)
    c[-1] = 1.0
    return c


def probabilities(mu, resid, line, sf=1.0):
    """(P(over), P(under), P(push)) at a sportsbook line; integer lines keep push mass."""
    g = _shape_fn(resid, sf)
    below = int(np.ceil(line)) - 1
    at = int(np.floor(line))
    under = float(np.interp(below + 0.5 - mu, _XGRID, g, left=0.0, right=1.0)) if below >= 0 else 0.0
    over = 1.0 - float(np.interp(at + 0.5 - mu, _XGRID, g, left=0.0, right=1.0))
    push = max(0.0, 1.0 - under - over)
    return float(over), float(under), float(push)


def crps_discrete(mu, resid, y, sf=1.0):
    c = cdf_grid(mu, resid, sf)
    ks = np.arange(len(c))
    return float(np.sum((c - (y <= ks)) ** 2))


def point_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=int(len(y)), mae=float(np.abs(y - p).mean()), rmse=float(np.sqrt(np.mean((y - p) ** 2))), bias=float(np.mean(p - y)))


# ------------------------------------------------------------------ validation
def expanding_oof(df: pd.DataFrame, name: str) -> pd.DataFrame:
    parts = []
    seasons = sorted(df.season.unique())
    for s in seasons:
        if s < TRAIN_START + 1:
            continue
        tr, te = df[(df.season < s) & (df.season >= TRAIN_START)], df[df.season == s].copy()
        if tr.empty or te.empty:
            continue
        base_name = "ridge" if name == "ridge_eb" else name
        obj = fit_base(base_name, tr)
        te["prediction"] = predict_base(base_name, obj, te)
        parts.append(te)
    oof = pd.concat(parts, ignore_index=True)
    oof["residual"] = oof.attempts - oof.prediction
    if name == "ridge_eb":
        # corrections for season S use OOF residuals from seasons < S *and* earlier weeks of S (as-of)
        oof = oof.sort_values("kickoff").reset_index(drop=True)
        adj = np.zeros(len(oof))
        hist = {}
        for idx, r in enumerate(oof.itertuples()):
            h = hist.get(r.player_id, [])
            if h:
                res = np.array([x[1] for x in h if x[0] + pd.Timedelta(hours=6) < r.kickoff])[-24:]
                if len(res):
                    w = 0.5 ** (np.arange(len(res))[::-1] / 8.0)
                    adj[idx] = (w * res).sum() / (w.sum() + EB_K)
            hist.setdefault(r.player_id, []).append((r.kickoff, r.residual))
        oof["prediction"] = (oof.prediction + adj).clip(10, 55)
        oof["residual"] = oof.attempts - oof.prediction
    oof["candidate"] = name
    return oof


def evaluate(oof: pd.DataFrame, seasons, use_scale=True) -> dict:
    """Score seasons with residual pools / scale models fit strictly on earlier seasons."""
    rows, brier, ll = [], [], []
    for s in seasons:
        prior = oof[oof.season < s]
        te = oof[oof.season == s]
        if len(prior) < 300 or te.empty:
            continue
        resid = prior.residual.to_numpy()
        sm = fit_scale(prior) if use_scale else None
        sf = scale_factor(sm, te)
        for i, r in enumerate(te.itertuples()):
            rows.append(dict(season=s, crps=crps_discrete(r.prediction, resid, r.attempts, sf[i]),
                             lo=r.prediction + sf[i] * np.quantile(resid, 0.1), hi=r.prediction + sf[i] * np.quantile(resid, 0.9),
                             y=r.attempts, p=r.prediction))
            for line in (24.5, 29.5, 34.5, 39.5, 44.5):
                po, pu, _ = probabilities(r.prediction, resid, line, sf[i])
                po = min(max(po, 1e-4), 1 - 1e-4)
                yo = int(r.attempts > line)
                brier.append((po - yo) ** 2)
                ll.append(-(yo * np.log(po) + (1 - yo) * np.log(1 - po)))
    d = pd.DataFrame(rows)
    out = point_metrics(d.y, d.p)
    out.update(crps=float(d.crps.mean()), coverage_80=float(((d.y >= d.lo) & (d.y <= d.hi)).mean()),
               fixed_threshold_brier=float(np.mean(brier)), fixed_threshold_log_loss=float(np.mean(ll)))
    return out


def train_validate(feats: pd.DataFrame, out_dir: Path = MODEL_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = feats[feats.season >= TRAIN_START].copy()
    oofs, summary = {}, {}
    for name in CANDIDATES:
        oof = expanding_oof(df, name)
        oofs[name] = oof
        summary[name] = dict(
            development=evaluate(oof, range(DEV_START, DEV_END + 1)),
            development_no_scale=evaluate(oof, range(DEV_START, DEV_END + 1), use_scale=False),
            by_year={str(s): evaluate(oof, [s]) for s in range(DEV_START, HOLDOUT + 1)},
        )
        print(f"{name:12s} dev CRPS {summary[name]['development']['crps']:.4f} (no-scale {summary[name]['development_no_scale']['crps']:.4f}) "
              f"MAE {summary[name]['development']['mae']:.4f}  | 2025 CRPS {summary[name]['by_year']['2025']['crps']:.4f} MAE {summary[name]['by_year']['2025']['mae']:.4f}", flush=True)
    chosen = min([c for c in CANDIDATES if c != "recent"], key=lambda n: summary[n]["development"]["crps"])
    use_scale = summary[chosen]["development"]["crps"] <= summary[chosen]["development_no_scale"]["crps"]
    holdout = evaluate(oofs[chosen], [HOLDOUT], use_scale=use_scale)
    baseline = evaluate(oofs["recent"], [HOLDOUT], use_scale=use_scale)
    # game-block bootstrap of the MAE difference vs recent baseline on the holdout
    a = oofs[chosen][oofs[chosen].season == HOLDOUT].set_index(["game_id", "player_id"]).prediction
    b = oofs["recent"][oofs["recent"].season == HOLDOUT].set_index(["game_id", "player_id"])
    diff = (b.attempts - a.reindex(b.index)).abs() - (b.attempts - b.prediction).abs()
    g = diff.groupby(level=0).mean().to_numpy()
    rng = np.random.default_rng(7)
    ci = np.quantile(rng.choice(g, (4000, len(g)), replace=True).mean(axis=1), [0.025, 0.975]).tolist()
    result = dict(version=VERSION, chosen=chosen, use_scale_model=bool(use_scale), selection_rule=f"lowest {DEV_START}-{DEV_END} expanding-season CRPS; {HOLDOUT} untouched holdout",
                  candidates=summary, holdout=holdout, holdout_baseline=baseline, holdout_mae_diff_95ci=ci, features=FEATURES,
                  training_rows=int(len(df)))
    # production artifact: base model fit on all rows; residual pool + scale from OOF; EB corrections if chosen
    base_name = "ridge" if chosen == "ridge_eb" else chosen
    oof = oofs[chosen]
    artifact = dict(version=VERSION, name=chosen, base_name=base_name, estimator=fit_base(base_name, df), residuals=oof.residual.to_numpy(),
                    scale=fit_scale(oof) if use_scale else None, eb=eb_corrections(oof) if chosen == "ridge_eb" else {}, metrics=result,
                    trained_through=int(df.season.max()))
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2))
    pd.concat(oofs.values())[["game_id", "season", "week", "kickoff", "player_id", "player", "team", "opponent", "attempts", "candidate", "prediction", "residual"]].to_csv(out_dir / "oof_predictions.csv", index=False)
    return artifact


def load_artifact(path: Path = MODEL_DIR / "model.pkl"):
    with path.open("rb") as f:
        return pickle.load(f)


def predict_rows(artifact, df: pd.DataFrame):
    """Mean + scale factor for live rows built by the same History.features() path."""
    mu = predict_base(artifact["base_name"], artifact["estimator"], df)
    if artifact["name"] == "ridge_eb":
        mu = apply_eb(mu, df, artifact["eb"])
    return mu, scale_factor(artifact["scale"], df)


def fit_only(feats: pd.DataFrame, out_dir: Path = MODEL_DIR):
    """Weekly production refit: keep the validated candidate + OOF residual pool, refit the base on all rows
    (new season's completed games included) and refresh EB corrections from the saved OOF file."""
    meta = json.loads((out_dir / "validation.json").read_text())
    oof_all = pd.read_csv(out_dir / "oof_predictions.csv", parse_dates=["kickoff"])
    oof = oof_all[oof_all.candidate == meta["chosen"]]
    df = feats[feats.season >= TRAIN_START].copy()
    base_name = "ridge" if meta["chosen"] == "ridge_eb" else meta["chosen"]
    # extend the OOF pool with in-season rows the frozen artifact has predicted (as-of) so EB corrections stay current
    prev = load_artifact(out_dir / "model.pkl")
    new = df[df.season > oof.season.max()] if not oof.empty else df.iloc[0:0]
    if not new.empty:
        mu = predict_base(prev["base_name"], prev["estimator"], new)
        add = new[["game_id", "season", "week", "kickoff", "player_id", "player", "team", "opponent", "attempts"]].copy()
        add["candidate"], add["prediction"] = meta["chosen"], mu
        add["residual"] = add.attempts - add.prediction
        oof = pd.concat([oof, add], ignore_index=True)
    feats_idx = feats.set_index(["game_id", "player_id"])
    oof_f = oof.merge(feats, on=["game_id", "player_id"], suffixes=("", "_f"))
    artifact = dict(version=VERSION, name=meta["chosen"], base_name=base_name, estimator=fit_base(base_name, df), residuals=oof.residual.to_numpy(),
                    scale=fit_scale(oof_f) if meta.get("use_scale_model", True) else None,
                    eb=eb_corrections(oof) if meta["chosen"] == "ridge_eb" else {}, metrics=meta, trained_through=int(df.season.max()),
                    refit_at=pd.Timestamp.now(tz="UTC").isoformat())
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    print(f"refit {meta['chosen']} on {len(df)} rows through season {artifact['trained_through']}; residual pool {len(oof)}")
    return artifact


if __name__ == "__main__":
    import sys
    feats = pd.read_parquet(DATA / "features.parquet")
    if "--fit-only" in sys.argv:
        fit_only(feats); sys.exit(0)
    art = train_validate(feats)
    print(json.dumps({k: art["metrics"][k] for k in ("chosen", "use_scale_model", "holdout", "holdout_baseline", "holdout_mae_diff_95ci")}, indent=1))
