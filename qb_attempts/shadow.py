"""Frozen research-only probability calibration and conditional residual experiment.

No function here recommends bets or changes either production model's candidates.
Historical quotes must be joined by event time, opponents and player, never week.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import team_key
from .model import probabilities
from .scoring import name_key, no_vig

VERSION = "shadow-protocol-2026-09-14"
MIN_FIT_ROWS = 200
MIN_VALIDATION_ROWS = 100
REAL_BOOKS = {"fanduel", "betmgm", "betrivers", "draftkings", "thescorebet", "fanatics", "caesars"}
CONDITION_COLUMNS = ["expected_margin", "game_total", "prediction"]


def utc(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    return result.tz_convert("UTC")


def prepare_archive(offers, features, oof, now):
    """Reconstruct one real paired offer per QB independently of outcome/EV.

    Provider scheduled/updated fields are documented archive UTC strings. Require
    exact official scheduled time and both opponent identities; stale player-team
    labels in this provider are deliberately not used to identify a matchup.
    """
    now = utc(now)
    raw = offers.copy()
    raw["kickoff"] = pd.to_datetime(raw.scheduled, utc=True, errors="coerce")
    raw["quote_at"] = pd.to_datetime(raw.updated, utc=True, errors="coerce")
    raw["nk"] = raw.player.map(name_key)
    raw["matchup"] = ["|".join(sorted([team_key(h), team_key(a)])) for h, a in zip(raw.home, raw.visitor)]
    good = (raw.book.isin(REAL_BOOKS) & raw.side.isin(["over", "under"])
            & (raw.kickoff < now) & (raw.quote_at < raw.kickoff)
            & (raw.quote_at >= raw.kickoff - pd.Timedelta(hours=24))
            & (raw.season <= 2025) & raw.line.between(0, 90) & (raw.line * 2 % 1 == 0)
            & ((raw.cost <= -100) | (raw.cost >= 100)))
    raw = raw.loc[good].copy()
    identities = ["event_id", "nk", "book", "line", "kickoff", "matchup"]
    raw = raw.sort_values("quote_at").drop_duplicates(identities + ["side"], keep="last")
    over = raw[raw.side == "over"][identities + ["cost", "quote_at"]]
    under = raw[raw.side == "under"][identities + ["cost", "quote_at"]]
    paired = over.merge(under, on=identities, suffixes=("_over", "_under"), validate="one_to_one")
    # Both sides must describe approximately the same snapshot.
    paired = paired[(paired.quote_at_over - paired.quote_at_under).abs() <= pd.Timedelta(minutes=30)]
    f = features.copy()
    f["kickoff"] = pd.to_datetime(f.kickoff, utc=True)
    f["nk"] = f.player.map(name_key)
    f["matchup"] = ["|".join(sorted([team_key(t), team_key(o)])) for t, o in zip(f.team, f.opponent)]
    keys = ["kickoff", "nk", "matchup"]
    # Ambiguous player/event identities are not resolved by looking at outcomes.
    f = f[~f.duplicated(keys, keep=False)]
    joined = paired.merge(f[keys + ["game_id", "player_id", "player", "season"]], on=keys, validate="many_to_one")
    p = oof[oof.candidate == "market_ridge"].copy()
    joined = joined.merge(p[["game_id", "player_id", "attempts", "prediction", "expected_margin", "game_total"]],
                          on=["game_id", "player_id"], validate="many_to_one")
    joined["p_own"] = [no_vig(a, b) for a, b in zip(joined.cost_over, joined.cost_under)]
    rows = []
    for _, group in joined.groupby(["game_id", "player_id", "line"], sort=True):
        for index, row in group.iterrows():
            peers = group[group.book != row.book]
            if peers.empty:
                continue
            result = row.to_dict()
            result["p_market"] = float(peers.p_own.median())
            result["reference_books"] = sorted(peers.book.unique().tolist())
            rows.append(result)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["game_id", "player_id", "book", "line", "event_id"]).drop_duplicates(["game_id", "player_id"])
    diagnostics = {"raw_offer_rows": len(offers), "timestamp_and_book_eligible_sides": len(raw),
                   "paired_offers": len(paired), "exact_event_joined_offers": len(joined),
                   "selected_qb_games": len(frame), "excluded_unmatched_pairs": len(paired) - len(joined),
                   "join_policy": "exact_utc_scheduled_time_opponent_pair_normalized_player",
                   "quote_policy": "real_book_paired_sides_updated_strictly_before_kickoff_within_24h_other_book_same_line"}
    diagnostics["raw_sides_by_season"] = {str(k): int(v) for k, v in offers.groupby("season").size().items()}
    diagnostics["eligible_sides_by_season"] = {str(k): int(v) for k, v in raw.groupby("season").size().items()}
    return frame.reset_index(drop=True), diagnostics


def fit_weight(frame):
    """Closed-form Brier-optimal convex weight; no grid or ROI optimization."""
    delta = frame.p_model.to_numpy() - frame.p_market.to_numpy()
    target = frame.target.to_numpy() - frame.p_market.to_numpy()
    return float(np.clip(np.dot(delta, target) / np.dot(delta, delta), 0, 1)) if np.dot(delta, delta) > 0 else 0.


def metrics(frame, column):
    if frame.empty:
        return {"n": 0, "brier": None, "log_loss": None}
    p = np.clip(frame[column].to_numpy(), 1e-7, 1 - 1e-7)
    y = frame.target.to_numpy()
    return {"n": len(frame), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def conditional_probabilities(mean, line, margin, total, history):
    """Fixed conditional empirical tails, with 200 global pseudo-observations.

    Kernel bandwidths are seven points/attempts, specified before evaluation.
    This conditions on projected volume and game script, not actual game state.
    """
    x = history[CONDITION_COLUMNS].to_numpy(dtype=float)
    query = np.array([margin, total, mean])
    weights = np.exp(-.5 * np.sum(((x-query)/7.)**2, axis=1))
    weights += 200. / len(history)
    weights /= weights.sum()
    r = history.residual.to_numpy(dtype=float)
    def cdf(k):
        return 0. if k < 0 else float(np.dot(weights, norm.cdf(k+.5-mean-r)))
    under = cdf(int(np.ceil(line))-1)
    over = 1-cdf(int(np.floor(line)))
    return over, under, max(0., 1-under-over)


def train_shadow(archive, oof, now, diagnostics=None):
    now = utc(now)
    artifact = {"version": VERSION, "status": "insufficient_data", "generated_at": now.isoformat(),
                "shadow_only": True, "promotion_allowed": False, "weight": None,
                "fit_seasons": [2022, 2023], "validation_seasons": [2024],
                "exposed_benchmark_seasons": [2025], "live_results_used_for_training": False,
                "diagnostics": diagnostics or {}, "metrics": {}, "residual_history": []}
    p = oof[(oof.candidate == "market_ridge") & (oof.season < 2025)].copy()
    p = p[np.isfinite(p[["residual"] + CONDITION_COLUMNS]).all(axis=1)]
    rows = []
    for row in archive.to_dict("records"):
        # Function-level safety also applies when caller supplies a custom archive.
        kickoff = utc(row["kickoff"])
        if kickoff >= now or row["season"] not in [2022, 2023, 2024, 2025]:
            continue
        if utc(row["quote_at_over"]) >= kickoff or utc(row["quote_at_under"]) >= kickoff:
            continue
        history = p[p.season < row["season"]]
        if len(history) < MIN_FIT_ROWS or row["attempts"] == row["line"]:
            continue
        over, _, push = probabilities(row["prediction"], history.residual, row["line"])
        co, _, cp = conditional_probabilities(row["prediction"], row["line"], row["expected_margin"], row["game_total"], history)
        rows.append({**row, "p_model": over/(1-push), "p_conditional": co/(1-cp),
                     "target": int(row["attempts"] > row["line"])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        artifact["diagnostics"]["reason"] = "No causal paired historical quote/forecast rows"
        return artifact, frame
    fit = frame[frame.season.isin([2022, 2023])]
    validation = frame[frame.season == 2024]
    if len(fit) < MIN_FIT_ROWS or len(validation) < MIN_VALIDATION_ROWS:
        artifact["diagnostics"].update(reason="Insufficient fixed chronological development rows", fit_rows=len(fit), validation_rows=len(validation))
        return artifact, frame
    weight = fit_weight(fit)
    frame["p_calibrated"] = frame.p_market + weight*(frame.p_model-frame.p_market)
    artifact.update(status="ok", weight=weight, residual_history=p[["season", "residual"]+CONDITION_COLUMNS].to_dict("records"))
    for name, seasons in [("fit_2022_2023", [2022, 2023]), ("validation_2024", [2024]), ("exposed_benchmark_2025", [2025])]:
        segment = frame[frame.season.isin(seasons)]
        artifact["metrics"][name] = {key: metrics(segment, col) for key, col in
            [("market_only", "p_market"), ("independent_model", "p_model"),
             ("calibrated_blend", "p_calibrated"), ("conditional_distribution", "p_conditional")]}
    artifact["diagnostics"].update(fit_rows=len(fit), validation_rows=len(validation),
        limitations=["Retrospectively downloaded closing-reference quotes; no verified executable morning archive",
                     "2024 and 2025 have previously been inspected; neither is an untouched holdout",
                     "Game spread/total features use historical closing references",
                     "Conditional tails include observed exits but do not estimate a separate exit hazard",
                     "One fixed kernel experiment; not a full within-game play-volume simulator"])
    validation_metrics = artifact["metrics"]["validation_2024"]
    artifact["validation_improved_brier"] = (validation_metrics["calibrated_blend"]["brier"] < validation_metrics["market_only"]["brier"])
    return artifact, frame


def score_shadow(candidates, now, artifact_path):
    """Publish one same-offer QB forecast, independent of EV or qualification."""
    now = utc(now)
    result = {"version": VERSION, "generated_at": now.isoformat(), "status": "insufficient_data",
              "shadow_only": True, "promotion_allowed": False, "predictions": [], "metrics": {}, "diagnostics": {}}
    try:
        artifact = json.loads(Path(artifact_path).read_text())
        if artifact.get("version") != VERSION or utc(artifact["generated_at"]) > now:
            raise ValueError("Invalid protocol version or future training artifact")
        if artifact.get("status") != "ok":
            result["diagnostics"] = artifact.get("diagnostics", {})
            return result
        weight = float(artifact["weight"])
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Invalid frozen blend weight")
        history = pd.DataFrame(artifact["residual_history"])
        if history.empty or history.season.max() >= 2025 or not np.isfinite(history[["residual"]+CONDITION_COLUMNS]).all(axis=None):
            raise ValueError("Invalid or out-of-protocol residual history")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result.update(status="artifact_unavailable", diagnostics={"reason": str(exc)})
        return result
    selected = {}
    diagnostics = {"invalid_candidates": 0}
    for row in sorted(candidates or [], key=lambda r: tuple(str(r.get(k, "")) for k in ["game_id", "player_id", "book", "line", "side"])):
        try:
            if utc(row["kickoff"]) <= now:
                continue
            key = (row["game_id"], row["player_id"])
            if key in selected:
                continue
            push = float(row["p_push"])
            mean, line, margin, total = [float(row[k]) for k in ("mean", "line", "expected_margin", "game_total")]
            if not all(np.isfinite(v) for v in [mean, line, margin, total]) or line < 0 or line*2 % 1:
                raise ValueError("Invalid count or game context")
            for field in ("observed_at", "source_updated_at"):
                if row.get(field) is not None and utc(row[field]) > now:
                    raise ValueError("Future quote timestamp")
            pm, market = float(row["p_model"])/(1-push), float(row["p_market"])/(1-push)
            if row["side"].lower() == "under":
                pm, market = 1-pm, 1-market
            elif row["side"].lower() != "over":
                raise ValueError("Invalid side")
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in [push, pm, market]) or push == 1:
                raise ValueError("Invalid probabilities")
            co, _, cp = conditional_probabilities(float(row["mean"]), float(row["line"]),
                                                  float(row["expected_margin"]), float(row["game_total"]), history)
            selected[key] = {**{k: row[k] for k in ["game_id", "player_id", "player", "book", "line", "kickoff"]},
                "market_only_p_over": market, "independent_p_over": pm,
                "calibrated_p_over": market+weight*(pm-market), "conditional_p_over": co/(1-cp),
                "conditional_p_push": cp, "production_p_push": push, "model_weight": weight,
                "mean": float(row["mean"]), "quote_verification": row.get("quote_verification"),
                "observed_at": row.get("observed_at"), "source_updated_at": row.get("source_updated_at")}
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            diagnostics["invalid_candidates"] += 1
    result.update(status="ok" if selected else "no_odds", predictions=list(selected.values()),
                  metrics=artifact.get("metrics", {}), diagnostics={**diagnostics, "training": artifact.get("diagnostics", {})})
    return result
