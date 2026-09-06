"""Game-level, situation-conditioned offensive tendencies from nflverse PBP.

These are *postgame* aggregates, never pregame features by themselves. Consumers
must use only earlier games. Neutral means a pre-play margin within seven points
with more than 120 regulation seconds remaining. Leading/trailing mean margins
greater than seven / less than minus seven, respectively, at any game time.

Dropbacks include sacks and scrambles; pass_attempts excludes both. We exclude
kneels, spikes, two-point tries and no-plays from preference denominators, so the
attempt count is deliberately not an official box-score total. ``dropback_oe``
is mean(qb_dropback - xpass) on valid xpass observations, in fraction units.
nflfastR documents xpass as expected dropback probability. Its upstream fitting
vintage is unknown: using downloaded historical xpass is retrospective research,
not proof of a strictly vintage-correct expected-pass model. Neutral rates remain
available without xpass and do not depend on an upstream fitted model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .odds_sources import normalize_team

AGGREGATION_VERSION = 1
KEYS = ["game_id", "season", "week", "team", "opponent", "game_date"]
REQUIRED_COLUMNS = {
    "game_id", "season", "week", "posteam", "defteam", "game_date", "play_type",
    "qb_dropback", "pass_attempt", "rush_attempt", "sack", "qb_scramble",
    "qb_kneel", "qb_spike", "two_point_attempt", "score_differential",
    "game_seconds_remaining",
}
PBP_COLUMNS = sorted(REQUIRED_COLUMNS | {"season_type", "play_id", "xpass"})
COUNT_COLUMNS = ["plays", "dropbacks", "pass_attempts", "neutral_plays", "neutral_dropbacks",
                 "lead_plays", "lead_dropbacks", "trail_plays", "trail_dropbacks", "xpass_plays"]
RATE_COLUMNS = ["dropback_rate", "attempts_per_dropback", "neutral_dropback_rate",
                "lead_dropback_rate", "trail_dropback_rate", "dropback_oe"]


def aggregate_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return one game/team row; empty situational samples have NaN rates."""
    missing = REQUIRED_COLUMNS - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")
    if "play_id" in pbp and pbp.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate game_id/play_id in PBP")
    df = pbp.copy()
    numeric = REQUIRED_COLUMNS - {"game_id", "posteam", "defteam", "game_date", "play_type"}
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df.play_type.isin(["run", "pass"]) & df.posteam.notna() & df.defteam.notna()
    valid &= df.qb_dropback.isin([0, 1])
    valid &= df.qb_dropback.eq(1) | df.rush_attempt.eq(1) | df.pass_attempt.eq(1)
    for col in ["qb_kneel", "qb_spike", "two_point_attempt"]:
        valid &= ~df[col].eq(1)
    if "season_type" in df:
        valid &= df.season_type.isin(["REG", "POST"])
    df = df.loc[valid].copy()
    if df.empty:
        return pd.DataFrame(columns=KEYS + COUNT_COLUMNS + RATE_COLUMNS)
    df["team"] = df.posteam.map(normalize_team)
    df["opponent"] = df.defteam.map(normalize_team)
    if df[KEYS].isna().any().any():
        raise ValueError("Missing game/team identity in valid offensive plays")
    df["plays"] = 1
    df["dropbacks"] = df.qb_dropback.astype(int)
    df["pass_attempts"] = (df.pass_attempt.eq(1) & ~df.sack.eq(1) & ~df.qb_scramble.eq(1)).astype(int)
    masks = {"neutral": df.score_differential.abs().le(7) & df.game_seconds_remaining.gt(120),
             "lead": df.score_differential.gt(7), "trail": df.score_differential.lt(-7)}
    for label, mask in masks.items():
        df[f"{label}_plays"] = mask.astype(int)
        df[f"{label}_dropbacks"] = df.dropbacks * mask
    xpass = pd.to_numeric(df.get("xpass", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["xpass_plays"] = xpass.between(0, 1).astype(int)
    df["dropback_oe_sum"] = (df.qb_dropback - xpass).where(xpass.between(0, 1), 0)
    out = df.groupby(KEYS, as_index=False, dropna=False)[COUNT_COLUMNS + ["dropback_oe_sum"]].sum()
    out["dropback_rate"] = out.dropbacks / out.plays
    out["attempts_per_dropback"] = out.pass_attempts / out.dropbacks.replace(0, np.nan)
    for label in masks:
        out[f"{label}_dropback_rate"] = out[f"{label}_dropbacks"] / out[f"{label}_plays"].replace(0, np.nan)
    out["dropback_oe"] = out.pop("dropback_oe_sum") / out.xpass_plays.replace(0, np.nan)
    out[COUNT_COLUMNS + ["season", "week"]] = out[COUNT_COLUMNS + ["season", "week"]].astype(int)
    return out[KEYS + COUNT_COLUMNS + RATE_COLUMNS].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
