"""Shared betting math + market-center inversion (used by the backtest and the live board)."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from .model import probabilities


def payout(odds):
    return odds / 100 if odds > 0 else 100 / -odds


def implied(odds):
    return 1 / (1 + payout(odds))


def devig(over, under):
    po, pu = implied(over), implied(under)
    return po / (po + pu)


def ev(p, odds, push=0.0):
    return p * payout(odds) - (1 - p - push)


# ------------------------------------------------------------------ market center via distribution inversion
def implied_center(line, p_over_cond, resid, sf):
    """Find mu such that P(X > line | X != line) == p_over_cond under the model's residual shape."""
    def f(mu):
        o, u, push = probabilities(mu, resid, line, sf)
        return o / max(1e-9, o + u) - p_over_cond
    try:
        return brentq(f, line - 25, line + 25, xtol=1e-3)
    except ValueError:
        return np.nan



def fair_odds(p):
    p = float(np.clip(p, 0.001, 0.999))
    return int(round(100 * (1 - p) / p if p <= 0.5 else -100 * p / (1 - p)))


def bet_to(p, push=0.0, target=0.02):
    """Worst American price at which the bet still returns `target` EV (rounded against the bettor)."""
    b = (1 - p - push + target) / max(p, 0.001)
    return math.ceil(100 * b) if b >= 1 else math.ceil(-100 / b)
