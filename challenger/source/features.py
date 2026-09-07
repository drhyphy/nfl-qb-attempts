"""Leakage-safe QB pass-attempt feature store built from nflverse play-by-play + schedules.

Design (lessons from earlier projects baked in):
- Label = official box-score pass attempts of the *listed starting QB* (nflverse
  home_qb_id/away_qb_id), including early exits (zero if no stat row). Never pick
  the QB with the most attempts (selection leakage).
- Every rolling feature is computed from games whose kickoff precedes the target
  game's kickoff (simultaneous games excluded). One shared builder serves both
  training rows and live rows, so train/serve mismatch is structurally impossible
  (the MLB NaN-at-inference and stale-merge bugs both came from separate paths).
- Game context (spread/total) comes from the schedule table: closing numbers for
  history, current numbers for upcoming games. The model card must state that
  historical context is closing, not 6:30 a.m.
- Rates are shrunk toward league priors with explicit pseudo-counts so a rookie
  or a new-team QB never inherits veteran certainty.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEAM_MAP = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def team_key(x):
    return TEAM_MAP.get(str(x), str(x))


PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
            "passer_player_id", "rusher_player_id", "pass_attempt", "sack", "qb_scramble", "qb_spike", "qb_kneel",
            "two_point_attempt", "play_type", "qb_dropback", "rush_attempt", "score_differential",
            "game_seconds_remaining", "qtr", "xpass", "no_huddle", "down", "epa", "yards_gained", "complete_pass",
            "incomplete_pass", "drive", "fixed_drive", "penalty", "first_down", "series"]


def load_pbp(seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        f = DATA / "pbp" / f"play_by_play_{s}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=PBP_COLS))
    p = pd.concat(frames, ignore_index=True)
    p = p[p.season_type == "REG"].copy()
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        p[c] = p[c].map(team_key)
    return p


def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "raw" / "games.csv")
    g = g[g.game_type == "REG"].copy()
    for c in ["home_team", "away_team"]:
        g[c] = g[c].map(team_key)
    g["kickoff"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g


def load_weekly_stats(seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        f = DATA / "raw" / f"stats_player_week_{s}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=["player_id", "player_display_name", "position", "season", "week", "season_type",
                                                      "game_id", "team", "opponent_team", "attempts", "completions", "sacks_suffered", "carries",
                                                      "passing_yards", "passing_epa"]))
    s = pd.concat(frames, ignore_index=True)
    s = s[s.season_type == "REG"].copy()
    s["team"] = s.team.map(team_key)
    return s.drop_duplicates(["game_id", "player_id"])


def team_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game/offense: volume, pass tendencies by game state, pace."""
    v = pbp[pbp.play_type.isin(["pass", "run", "qb_spike", "qb_kneel"]) & pbp.posteam.notna()].copy()
    v["is_att"] = ((v.pass_attempt == 1) & (v.sack == 0) & (v.two_point_attempt == 0)).astype(int)
    v["is_db"] = (v.qb_dropback == 1).astype(int)
    v["is_sack"] = (v.sack == 1).astype(int)
    v["is_scr"] = (v.qb_scramble == 1).astype(int)
    v["is_play"] = 1
    v["neutral"] = ((v.score_differential.abs() <= 7) & (v.game_seconds_remaining > 120) & (v.qtr <= 3)).astype(int)
    v["lead"] = (v.score_differential > 7).astype(int)
    v["trail"] = (v.score_differential < -7).astype(int)
    v["neutral_db"] = v.neutral * v.is_db
    v["lead_play"] = v.lead
    v["trail_play"] = v.trail
    v["lead_db"] = v.lead * v.is_db
    v["trail_db"] = v.trail * v.is_db
    xp = pd.to_numeric(v.xpass, errors="coerce")
    v["xpass_valid"] = xp.between(0, 1).astype(int)
    v["db_oe_sum"] = (v.is_db - xp).where(xp.between(0, 1), 0.0)
    v["neutral_xpass_valid"] = v.neutral * v.xpass_valid
    v["neutral_db_oe_sum"] = v.neutral * v.db_oe_sum
    v["nohuddle"] = (v.no_huddle == 1).astype(int)
    v["first_down_gained"] = pd.to_numeric(v.first_down, errors="coerce").fillna(0).astype(int)
    agg = v.groupby(["game_id", "posteam"]).agg(
        plays=("is_play", "sum"), dropbacks=("is_db", "sum"), team_att=("is_att", "sum"), sacks=("is_sack", "sum"),
        scrambles=("is_scr", "sum"), neutral_plays=("neutral", "sum"), neutral_db=("neutral_db", "sum"),
        lead_plays=("lead_play", "sum"), lead_db=("lead_db", "sum"), trail_plays=("trail_play", "sum"), trail_db=("trail_db", "sum"),
        xpass_valid=("xpass_valid", "sum"), db_oe_sum=("db_oe_sum", "sum"), neutral_xpass_valid=("neutral_xpass_valid", "sum"),
        neutral_db_oe_sum=("neutral_db_oe_sum", "sum"), nohuddle=("nohuddle", "sum"), first_downs=("first_down_gained", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    # Seconds per offensive play (pace): total game seconds the offense consumed is not observable here; use
    # plays as pace proxy plus drives count.
    drives = v.groupby(["game_id", "posteam"]).fixed_drive.nunique().rename("drives").reset_index().rename(columns={"posteam": "team"})
    agg = agg.merge(drives, on=["game_id", "team"], how="left")
    return agg


def game_records(stats: pd.DataFrame, games: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """One row per scheduled starting QB per completed game (label rows)."""
    tg_idx = tg.set_index(["game_id", "team"])
    st = stats.set_index(["game_id", "player_id"])
    rows = []
    for g in games.sort_values("kickoff").to_dict("records"):
        if pd.isna(g["home_score"]):
            continue
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = g[f"{side}_team"], g[f"{other}_team"]
            pid = g[f"{side}_qb_id"]
            if not isinstance(pid, str) or (g["game_id"], team) not in tg_idx.index:
                continue
            t = tg_idx.loc[(g["game_id"], team)]
            o = tg_idx.loc[(g["game_id"], opp)] if (g["game_id"], opp) in tg_idx.index else None
            s = st.loc[(g["game_id"], pid)] if (g["game_id"], pid) in st.index else None
            att = float(s["attempts"]) if s is not None else 0.0
            if att == 0.0 and (s is None or float(s["carries"]) == 0.0):
                # Listed starter with no attempt and no carry = did not play (scratch or schedule error).
                # A sportsbook voids that prop; it is neither a label nor QB information.
                continue
            qb_share = att / max(1.0, float(t.team_att))
            margin = float(g["result"]) if side == "home" else -float(g["result"])
            exp_margin = float(g["spread_line"]) if side == "home" else -float(g["spread_line"])
            rows.append(dict(
                game_id=g["game_id"], season=int(g["season"]), week=int(g["week"]), kickoff=g["kickoff"], team=team, opponent=opp,
                player_id=pid, player=g[f"{side}_qb_name"], home=int(side == "home" and g["location"] != "Neutral"),
                rest=float(g[f"{side}_rest"]), coach=g[f"{side}_coach"], opp_coach=g[f"{other}_coach"],
                dome=int(g["roof"] in ("dome", "closed")), div_game=int(g["div_game"]), exp_margin=exp_margin,
                total_line=float(g["total_line"]), actual_margin=margin, total_points=float(g["total"]),
                attempts=att, qb_share=qb_share, qb_full_game=int(qb_share >= 0.85),
                qb_carries=float(s["carries"]) if s is not None else 0.0, qb_sacks=float(s["sacks_suffered"]) if s is not None else 0.0,
                team_att=float(t.team_att), plays=float(t.plays), dropbacks=float(t.dropbacks), sacks=float(t.sacks), scrambles=float(t.scrambles),
                neutral_plays=float(t.neutral_plays), neutral_db=float(t.neutral_db), lead_plays=float(t.lead_plays), lead_db=float(t.lead_db),
                trail_plays=float(t.trail_plays), trail_db=float(t.trail_db), xpass_valid=float(t.xpass_valid), db_oe_sum=float(t.db_oe_sum),
                neutral_xpass_valid=float(t.neutral_xpass_valid), neutral_db_oe_sum=float(t.neutral_db_oe_sum), nohuddle=float(t.nohuddle),
                drives=float(t.drives), first_downs=float(t.first_downs),
                opp_plays=float(o.plays) if o is not None else np.nan, opp_att=float(o.team_att) if o is not None else np.nan,
                opp_dropbacks=float(o.dropbacks) if o is not None else np.nan, opp_drives=float(o.drives) if o is not None else np.nan,
                # what the defense allowed: computed from the offense rows of the opponent later via 'defenses' history
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- as-of features
PRIORS = dict(team_att=33.0, plays=63.0, dropbacks=37.0, db_rate=0.58, neutral_db_rate=0.55, sack_rate=0.065, scr_rate=0.05,
              qb_share=0.97, att_per_db=0.88, db_oe=0.0, drives=11.0)


def _wmean(vals, weights=None):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan
    if weights is None:
        return float(vals.mean())
    w = np.asarray(weights, dtype=float)
    return float((vals * w).sum() / w.sum())


def _decayed(hist, key, half_life, default, n_max=20):
    """Exponentially-decayed mean over the last n_max games (most recent weighted most)."""
    if not hist:
        return default, 0.0
    h = hist[-n_max:]
    ages = np.arange(len(h))[::-1]
    w = 0.5 ** (ages / half_life)
    vals = np.array([r[key] for r in h], dtype=float)
    return float((vals * w).sum() / w.sum()), float(w.sum())


def _ratio_decayed(hist, num, den, half_life, default, n_max=20, prior_n=3.0):
    if not hist:
        return default
    h = hist[-n_max:]
    ages = np.arange(len(h))[::-1]
    w = 0.5 ** (ages / half_life)
    n = sum(wi * r[num] for wi, r in zip(w, h))
    d = sum(wi * r[den] for wi, r in zip(w, h))
    # shrink with pseudo-count expressed in denominator units (e.g. 3 games of plays)
    avg_den = d / max(w.sum(), 1e-9)
    return float((n + default * prior_n * avg_den) / (d + prior_n * avg_den)) if d > 0 else default


FEATURES = [
    # QB own history
    "qb_att_d3", "qb_att_d8", "qb_att_mean", "qb_att_sd", "qb_games", "qb_season_games", "qb_share_d8", "qb_carries_d8",
    "qb_scr_rate", "qb_sack_rate", "qb_att_per_db", "qb_exit_rate",
    # team offense (current team, includes games with other QBs)
    "tm_att_d3", "tm_att_d8", "tm_plays_d3", "tm_plays_d8", "tm_db_rate_d8", "tm_neutral_db_rate_d8", "tm_db_oe_d8",
    "tm_neutral_db_oe_d8", "tm_lead_db_rate", "tm_trail_db_rate", "tm_nohuddle_rate", "tm_drives_d8", "tm_first_downs_d8",
    "tm_sack_rate_d8", "tm_scr_rate_d8",
    # opponent defense faced
    "opp_att_allowed_d8", "opp_plays_allowed_d8", "opp_db_rate_allowed_d8", "opp_db_oe_allowed_d8", "opp_drives_allowed_d8",
    "opp_first_downs_allowed_d8", "opp_sack_rate_d8",
    # opponent offense (dictates game pace / possession count)
    "opp_off_plays_d8", "opp_off_db_rate_d8",
    # game context
    "exp_margin", "abs_exp_margin", "total_line", "implied_team_pts", "implied_opp_pts", "home", "rest", "week", "dome", "div_game",
    "coach_change", "qb_team_change", "team_coach_games", "season_frac",
]


def vector(row, players, teams, defenses, league):
    ph_all = players[row["player_id"]]
    # Level features use full games only: an early exit (injury/benching/blowout pull) says nothing
    # about how many passes a healthy starter throws. Exit frequency is its own feature.
    ph = [r for r in ph_all if r["qb_share"] >= 0.85]
    th = teams[row["team"]]
    dh = defenses[row["opponent"]]        # opponent's defense: offense rows of teams that played the opponent
    oh = teams[row["opponent"]]           # opponent's own offense
    f = {}
    # ---- QB
    n_qb = len(ph)
    lg_att = league.get("att", PRIORS["team_att"])
    tm_att8, _ = _decayed(th, "team_att", 4.0, lg_att, 12)
    qb_d3, w3 = _decayed(ph, "attempts", 1.5, tm_att8, 6)
    qb_d8, w8 = _decayed(ph, "attempts", 4.0, tm_att8, 16)
    # explicit shrinkage toward the team's recent volume: rookies / new starters carry little own weight
    k = 3.0
    f["qb_att_d8"] = (w8 * qb_d8 + k * tm_att8) / (w8 + k)
    f["qb_att_d3"] = (w3 * qb_d3 + k * f["qb_att_d8"]) / (w3 + k)
    f["qb_att_mean"] = _wmean([r["attempts"] for r in ph[-20:]]) if ph else tm_att8
    f["qb_att_sd"] = float(np.std([r["attempts"] for r in ph[-16:]])) if n_qb >= 4 else 8.0
    f["qb_games"] = min(len(ph_all), 40)
    f["qb_season_games"] = sum(r["season"] == row["season"] for r in ph_all)
    recent_all = ph_all[-20:]
    f["qb_exit_rate"] = (sum(r["qb_share"] < 0.85 for r in recent_all) + 0.08 * 5) / (len(recent_all) + 5)
    f["qb_share_d8"] = _ratio_decayed(ph, "attempts", "team_att", 4.0, PRIORS["qb_share"], 16, prior_n=2.0)
    f["qb_carries_d8"], _ = _decayed(ph, "qb_carries", 4.0, 3.0, 16)
    f["qb_scr_rate"] = _ratio_decayed(ph, "scrambles", "dropbacks", 6.0, PRIORS["scr_rate"], 20, prior_n=4.0)
    f["qb_sack_rate"] = _ratio_decayed(ph, "qb_sacks", "dropbacks", 6.0, PRIORS["sack_rate"], 20, prior_n=4.0)
    f["qb_att_per_db"] = _ratio_decayed(ph, "attempts", "dropbacks", 6.0, PRIORS["att_per_db"], 20, prior_n=4.0)
    # ---- team offense
    f["tm_att_d3"], _ = _decayed(th, "team_att", 1.5, lg_att, 6)
    f["tm_att_d8"] = tm_att8
    f["tm_plays_d3"], _ = _decayed(th, "plays", 1.5, league.get("plays", PRIORS["plays"]), 6)
    f["tm_plays_d8"], _ = _decayed(th, "plays", 4.0, league.get("plays", PRIORS["plays"]), 12)
    f["tm_db_rate_d8"] = _ratio_decayed(th, "dropbacks", "plays", 4.0, league.get("db_rate", PRIORS["db_rate"]), 12)
    f["tm_neutral_db_rate_d8"] = _ratio_decayed(th, "neutral_db", "neutral_plays", 4.0, league.get("neutral_db_rate", PRIORS["neutral_db_rate"]), 12)
    f["tm_db_oe_d8"] = _ratio_decayed(th, "db_oe_sum", "xpass_valid", 4.0, 0.0, 12)
    f["tm_neutral_db_oe_d8"] = _ratio_decayed(th, "neutral_db_oe_sum", "neutral_xpass_valid", 4.0, 0.0, 12)
    f["tm_lead_db_rate"] = _ratio_decayed(th, "lead_db", "lead_plays", 8.0, 0.45, 20, prior_n=4.0)
    f["tm_trail_db_rate"] = _ratio_decayed(th, "trail_db", "trail_plays", 8.0, 0.72, 20, prior_n=4.0)
    f["tm_nohuddle_rate"] = _ratio_decayed(th, "nohuddle", "plays", 4.0, 0.08, 12)
    f["tm_drives_d8"], _ = _decayed(th, "drives", 4.0, PRIORS["drives"], 12)
    f["tm_first_downs_d8"], _ = _decayed(th, "first_downs", 4.0, 19.0, 12)
    f["tm_sack_rate_d8"] = _ratio_decayed(th, "sacks", "dropbacks", 4.0, PRIORS["sack_rate"], 12)
    f["tm_scr_rate_d8"] = _ratio_decayed(th, "scrambles", "dropbacks", 4.0, PRIORS["scr_rate"], 12)
    # ---- opponent defense (what offenses did against them)
    f["opp_att_allowed_d8"], _ = _decayed(dh, "team_att", 4.0, lg_att, 12)
    f["opp_plays_allowed_d8"], _ = _decayed(dh, "plays", 4.0, league.get("plays", PRIORS["plays"]), 12)
    f["opp_db_rate_allowed_d8"] = _ratio_decayed(dh, "dropbacks", "plays", 4.0, league.get("db_rate", PRIORS["db_rate"]), 12)
    f["opp_db_oe_allowed_d8"] = _ratio_decayed(dh, "db_oe_sum", "xpass_valid", 4.0, 0.0, 12)
    f["opp_drives_allowed_d8"], _ = _decayed(dh, "drives", 4.0, PRIORS["drives"], 12)
    f["opp_first_downs_allowed_d8"], _ = _decayed(dh, "first_downs", 4.0, 19.0, 12)
    f["opp_sack_rate_d8"] = _ratio_decayed(dh, "sacks", "dropbacks", 4.0, PRIORS["sack_rate"], 12)
    # ---- opponent offense
    f["opp_off_plays_d8"], _ = _decayed(oh, "plays", 4.0, league.get("plays", PRIORS["plays"]), 12)
    f["opp_off_db_rate_d8"] = _ratio_decayed(oh, "dropbacks", "plays", 4.0, league.get("db_rate", PRIORS["db_rate"]), 12)
    # ---- context
    f["exp_margin"] = float(row["exp_margin"])
    f["abs_exp_margin"] = abs(float(row["exp_margin"]))
    f["total_line"] = float(row["total_line"])
    f["implied_team_pts"] = (f["total_line"] + f["exp_margin"]) / 2
    f["implied_opp_pts"] = (f["total_line"] - f["exp_margin"]) / 2
    f["home"] = int(row["home"])
    f["rest"] = min(float(row["rest"]), 21.0)
    f["week"] = int(row["week"])
    f["dome"] = int(row["dome"])
    f["div_game"] = int(row["div_game"])
    f["coach_change"] = int(bool(th) and th[-1]["coach"] != row["coach"])
    f["qb_team_change"] = int(bool(ph) and ph[-1]["team"] != row["team"])
    f["team_coach_games"] = sum(r["coach"] == row["coach"] for r in th[-17:])
    f["season_frac"] = (int(row["week"]) - 1) / 17.0
    return f


class History:
    """Replays completed games in kickoff order and emits features strictly as-of each request."""

    def __init__(self, records: pd.DataFrame):
        self.records = records

    def features(self, requests: list[dict]) -> pd.DataFrame:
        players, teams, defenses = defaultdict(list), defaultdict(list), defaultdict(list)
        league_hist = []
        events = sorted(self.records.to_dict("records"), key=lambda r: r["kickoff"])
        out, i = [], 0
        for row in sorted(requests, key=lambda r: min(r["kickoff"], r.get("as_of", r["kickoff"]))):
            cutoff = min(row["kickoff"], row.get("as_of", row["kickoff"]))
            while i < len(events) and events[i]["kickoff"] + pd.Timedelta(hours=6) < cutoff:
                r = events[i]
                players[r["player_id"]].append(r)
                teams[r["team"]].append(r)
                defenses[r["opponent"]].append(r)   # r is an offense row vs. r['opponent'] defense
                league_hist.append(r)
                i += 1
            lh = league_hist[-320:]  # ~ last 10 weeks league-wide
            league = {}
            if lh:
                league = dict(att=float(np.mean([r["team_att"] for r in lh])), plays=float(np.mean([r["plays"] for r in lh])),
                              db_rate=float(np.sum([r["dropbacks"] for r in lh]) / np.sum([r["plays"] for r in lh])),
                              neutral_db_rate=float(np.sum([r["neutral_db"] for r in lh]) / max(1, np.sum([r["neutral_plays"] for r in lh]))))
            out.append({**row, **vector(row, players, teams, defenses, league)})
        return pd.DataFrame(out)


def build_dataset(seasons=range(2017, 2026)):
    pbp = load_pbp(seasons)
    games = load_games()
    games = games[games.season.isin(list(seasons))]
    stats = load_weekly_stats(seasons)
    tg = team_game_table(pbp)
    records = game_records(stats, games, tg)
    feats = History(records).features(records.to_dict("records"))
    feats = feats.sort_values(["kickoff", "game_id", "team"]).reset_index(drop=True)  # deterministic order (seed-noise lesson)
    return feats, records


if __name__ == "__main__":
    feats, records = build_dataset()
    out = DATA / "features.parquet"
    feats.to_parquet(out, index=False)
    records.to_parquet(DATA / "records.parquet", index=False)
    print(feats.shape, "->", out)
    print(feats[FEATURES].describe().T.to_string())
