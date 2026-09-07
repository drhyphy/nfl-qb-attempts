"""Fair prospective, shared-offer forecast comparison.

A prediction pair is published before results exist and is selected independently
of betting recommendations. The first valid shared publication is immutable for
scoring. Binary probability scores condition on no push and omit observed pushes;
all settled attempts still contribute to MAE. These are descriptive scores, not
independent-game significance estimates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .tracking import _number, _time


def _forecast(candidate):
    mean = _number(candidate.get("mean"))
    push = _number(candidate.get("p_push"))
    probability = _number(candidate.get("p_final"))
    side = str(candidate.get("side", "")).lower()
    if (mean is None or mean < 0 or push is None or not 0 <= push < 1
            or probability is None or not 0 <= probability <= 1 - push + 1e-10
            or side not in {"over", "under"}):
        return None
    conditional = min(1., probability / (1 - push))
    return {"mean": mean, "p_over": conditional if side == "over" else 1 - conditional,
            "p_push": push}


def _identity(candidate):
    if not isinstance(candidate, dict):
        return None
    ids = [candidate.get(field) for field in ("game_id", "player_id", "book")]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        return None
    line = _number(candidate.get("line"))
    kickoff = _time(candidate.get("kickoff"))
    if line is None or line < 0 or line * 2 != int(line * 2) or kickoff is None:
        return None
    return (*ids, line), kickoff


def _candidate_map(candidates):
    grouped = {}
    for candidate in candidates or []:
        identity = _identity(candidate)
        if identity is None:
            continue
        key, kickoff = identity
        forecast = _forecast(candidate)
        if forecast is not None:
            grouped.setdefault(key, []).append((kickoff, forecast, candidate))
    result = {}
    for key, rows in grouped.items():
        first = rows[0]
        # Over/under candidate duplicates must describe the same distribution.
        # Conflicting duplicates cannot be resolved using the favorable result.
        if all(row[0] == first[0] and all(math.isclose(row[1][field], first[1][field],
                    rel_tol=1e-9, abs_tol=1e-9) for field in ("mean", "p_over", "p_push"))
               for row in rows):
            result[key] = first
    return result


def paired_forecasts(champion_candidates: list[dict], challenger_candidates: list[dict]) -> list[dict]:
    """Pick one common book/line per QB, independent of EV and qualification.

    Inputs are full scored candidate arrays, not recommendation cards. Standard
    p_final is unconditional side-win mass; returned p_over conditions on no push.
    Both models must refer to the same kickoff and the exact offered book/line.
    """
    champion = _candidate_map(champion_candidates)
    challenger = _candidate_map(challenger_candidates)
    selected = {}
    for key in sorted(champion.keys() & challenger.keys()):
        game_id, player_id, book, line = key
        left, right = champion[key], challenger[key]
        if left[0] != right[0] or (game_id, player_id) in selected:
            continue
        selected[(game_id, player_id)] = {
            "game_id": game_id, "player_id": player_id,
            "player": left[2].get("player", right[2].get("player")),
            "team": left[2].get("team", right[2].get("team")),
            "book": book, "line": line, "kickoff": left[0].isoformat(),
            "champion": left[1], "challenger": right[1],
        }
    return list(selected.values())


def _valid_pair(row):
    if _identity(row) is None:
        return False
    for model in ("champion", "challenger"):
        output = row.get(model)
        if not isinstance(output, dict):
            return False
        mean, probability, push = [_number(output.get(key)) for key in ("mean", "p_over", "p_push")]
        if (mean is None or mean < 0 or probability is None or not 0 <= probability <= 1
                or push is None or not 0 <= push < 1):
            return False
    return True


def _scores(results, model):
    settled = [row for row in results if row["actual"] is not None]
    binary = [row for row in settled if row["actual"] != row["line"]]
    return {
        "mae": sum(abs(row[model]["mean"] - row["actual"]) for row in settled) / len(settled) if settled else None,
        "brier": sum((row[model]["p_over"] - (row["actual"] > row["line"])) ** 2 for row in binary) / len(binary) if binary else None,
        "log_loss": sum(-math.log(max(1e-15, min(1 - 1e-15,
            row[model]["p_over"] if row["actual"] > row["line"] else 1 - row[model]["p_over"])))
            for row in binary) / len(binary) if binary else None,
    }


def grade_comparison(history_dir: Path, stats: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Grade the first valid pregame shared snapshot per game/player.

    Requires one official schedule row with both final scores and one consistent
    nonnegative integer attempts value. Repeated identical stat rows are harmless;
    missing/conflicting stats remain unresolved. Future publications and postgame
    publications disguised with a later kickoff are excluded.
    """
    now = pd.Timestamp.now(tz="UTC")
    diagnostics, boards = [], []
    for path in sorted(Path(history_dir).glob("*.json")):
        try:
            board = json.loads(path.read_text())
            generated = _time(board.get("generated_at"))
            comparison = board.get("comparison", {})
            pairs = comparison.get("paired_forecasts", [])
            if generated is None or generated > now or not isinstance(pairs, list):
                raise ValueError("invalid/future generation time or paired forecast list")
            boards.append((generated, path.name, pairs))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            diagnostics.append({"file": path.name, "reason": str(exc)})
    game_map, stat_map = {}, {}
    for row in games.to_dict("records"):
        game_map.setdefault(row.get("game_id"), []).append(row)
    for row in stats.to_dict("records"):
        stat_map.setdefault((row.get("game_id"), row.get("player_id")), []).append(row)
    frozen = {}
    for generated, name, pairs in sorted(boards, key=lambda item: (item[0], item[1])):
        # A board with multiple offers for a QB is still resolved by identity,
        # never by performance. Normal publications have already selected one.
        valid = [row for row in pairs if isinstance(row, dict) and _valid_pair(row)]
        for row in sorted(valid, key=lambda r: (r["game_id"], r["player_id"], r["book"], float(r["line"]))):
            key = (row["game_id"], row["player_id"])
            kickoff = _time(row["kickoff"])
            official_starts = [_time(g.get("kickoff")) for g in game_map.get(key[0], [])]
            if generated >= kickoff or any(start is not None and generated >= start for start in official_starts):
                continue
            if key not in frozen:
                frozen[key] = {**row, "line": float(row["line"]), "kickoff": kickoff.isoformat(),
                               "generated_at": generated.isoformat(), "snapshot": name,
                               **{model: {field: float(row[model][field])
                                   for field in ("mean", "p_over", "p_push")}
                                  for model in ("champion", "challenger")}}
    results = []
    for key, pair in frozen.items():
        row = {**pair, "actual": None, "settlement_reason": "official_final_score_unavailable",
               "probability_scored": False}
        official = game_map.get(key[0], [])
        official_start = _time(official[0].get("kickoff")) if len(official) == 1 else None
        if (len(official) == 1 and official_start is not None and official_start <= now
                and _time(pair["kickoff"]) <= now
                and _number(official[0].get("home_score")) is not None
                and _number(official[0].get("away_score")) is not None):
            actuals = {_number(stat.get("attempts")) for stat in stat_map.get(key, [])}
            if (len(actuals) == 1 and None not in actuals and next(iter(actuals)) >= 0
                    and next(iter(actuals)).is_integer()):
                actual = next(iter(actuals))
                row.update(actual=actual, settlement_reason="official_player_stat",
                           probability_scored=actual != row["line"])
            else:
                row["settlement_reason"] = "player_stat_missing_or_conflicting"
        results.append(row)
    champion, challenger = _scores(results, "champion"), _scores(results, "challenger")
    return {
        "n_total": len(results), "n_settled": sum(row["actual"] is not None for row in results),
        "n_probability_scored": sum(row["probability_scored"] for row in results),
        "champion": champion, "challenger": challenger,
        "differences": {metric: challenger[metric] - champion[metric] if champion[metric] is not None else None
                        for metric in champion},
        "difference_definition": "challenger_minus_champion; negative favors challenger",
        "record_policy": "first_valid_pregame_shared_forecast_per_game_player",
        "offer_policy": "lexicographic_exact_book_then_line_independent_of_ev_or_qualification",
        "probability_definition": "conditional_over_probability; observed_pushes_excluded_from_binary_scores",
        "uncertainty_note": "Descriptive paired scores; correlated quarterbacks and repeated teams are not independent evidence.",
        "results": results, "diagnostics": diagnostics,
    }
