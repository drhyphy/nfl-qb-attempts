"""Selected Claude engine. Function bodies copied verbatim from source/model.py.
Unused candidate training and LightGBM import are deliberately omitted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES
EB_K = 20.0
_G_CACHE = {}
_XGRID = np.arange(-70.0, 70.0, 0.02)

def fit_ridge(df):
    m = make_pipeline(StandardScaler(), Ridge(alpha=60.0))
    m.fit(df[FEATURES], df.attempts)
    return m

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

def predict_rows(artifact, df: pd.DataFrame):
    """Mean + scale factor for live rows built by the same History.features() path."""
    mu = predict_base(artifact["base_name"], artifact["estimator"], df)
    if artifact["name"] == "ridge_eb":
        mu = apply_eb(mu, df, artifact["eb"])
    return mu, scale_factor(artifact["scale"], df)
