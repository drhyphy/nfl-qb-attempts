"""Grade frozen pregame publications; never reselect history using today's policy.

No files are mutated. Missing player rows are unresolved, not invented zeros or
automatic voids. Games with final scores are treated as officially completed in
the supplied nflverse schedule; callers must supply the finalized data source.
Closing comparisons require the exact event and a paired quote in the last hour.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median

import pandas as pd


def _time(value):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            return None
        return result.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value):
    try:
        result = float(value)
        return result if not isinstance(value, bool) and math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _decimal(value):
    odds = _number(value)
    if odds is None or abs(odds) < 100:
        return None
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def _summary(rows):
    settled = [r for r in rows if r["result"] in {"win", "loss", "push"}]
    units = sum(r["profit_1u"] for r in settled)
    closing = [r["clv"] for r in rows if r["clv"] is not None]
    return {
        "bets": len(rows), "settled": len(settled),
        "wins": sum(r["result"] == "win" for r in settled),
        "losses": sum(r["result"] == "loss" for r in settled),
        "pushes": sum(r["result"] == "push" for r in settled),
        "unresolved": len(rows) - len(settled), "units": units,
        "roi": units / len(settled) if settled else None,
        "clv_bets": len(closing),
        "mean_clv": sum(closing) / len(closing) if closing else None,
    }


def _closing(bet, quotes):
    kickoff = _time(bet["kickoff"])
    latest = {}
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        if (quote.get("game_id") != bet["game_id"]
                or quote.get("player_id") != bet["player_id"]
                or _number(quote.get("line")) != bet["line"]):
            continue
        observed = _time(quote.get("observed_at"))
        if observed is None or not kickoff - pd.Timedelta(minutes=60) <= observed <= kickoff:
            continue
        # An explicitly stale provider update cannot become fresh by re-fetching.
        if quote.get("source_updated_at") is not None:
            updated = _time(quote["source_updated_at"])
            if updated is None or not kickoff - pd.Timedelta(minutes=60) <= updated <= observed:
                continue
        if quote.get("is_live") or quote.get("line_status", "normal") != "normal":
            continue
        book = quote.get("book")
        over, under = _decimal(quote.get("over_odds")), _decimal(quote.get("under_odds"))
        if not isinstance(book, str) or not book or over is None or under is None:
            continue
        fair_over = (1 / over) / (1 / over + 1 / under)
        if book not in latest or observed > latest[book][0]:
            latest[book] = (observed, fair_over)
    if not latest:
        return {"clv": None, "closing_probability": None, "closing_books": 0,
                "closing_observed_at": None}
    p_over = median(v[1] for v in latest.values())
    p_side = p_over if bet["side"] == "over" else 1 - p_over
    return {"clv": p_side * _decimal(bet["odds"]) - 1,
            "closing_probability": p_side, "closing_books": len(latest),
            "closing_observed_at": max(v[0] for v in latest.values()).isoformat()}


def grade_history(history_dir: Path, stats: pd.DataFrame, games: pd.DataFrame,
                  quotes: list[dict] | None = None) -> dict:
    """Return flat-one-unit prospective results and conditional price CLV.

    The earliest valid pregame publication wins per (game_id, player_id), even
    if a later run changes sides, price, line, model, or eligibility. Empty later
    boards do not erase earlier recommendations. Kickoff is checked against both
    the frozen recommendation and current official schedule when available.
    CLV is conditional on no push at integer lines, not unconditional model EV.
    """
    diagnostics = []
    boards = []
    for path in sorted(Path(history_dir).glob("*.json")):
        try:
            board = json.loads(path.read_text())
            generated = _time(board.get("generated_at"))
            if generated is None or not isinstance(board.get("recommendations"), list):
                raise ValueError("missing UTC generation time or recommendations list")
            boards.append((generated, path.name, board))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            diagnostics.append({"file": path.name, "reason": str(exc)})

    game_map = {}
    for row in games.to_dict("records"):
        key = row.get("game_id")
        if key is not None:
            game_map.setdefault(key, []).append(row)
    stat_map = {}
    for row in stats.to_dict("records"):
        stat_map.setdefault((row.get("game_id"), row.get("player_id")), []).append(row)

    frozen = {}
    for generated, name, board in sorted(boards, key=lambda row: (row[0], row[1])):
        for rec in board["recommendations"]:
            if not isinstance(rec, dict):
                continue
            game_id, player_id = rec.get("game_id"), rec.get("player_id")
            if not isinstance(game_id, str) or not game_id or not isinstance(player_id, str) or not player_id:
                continue
            key = (game_id, player_id)
            kickoff = _time(rec.get("kickoff"))
            line, dec = _number(rec.get("line")), _decimal(rec.get("odds"))
            side = str(rec.get("side", "")).lower()
            if (kickoff is None or generated >= kickoff or line is None or line < 0
                    or line * 2 != int(line * 2) or dec is None or side not in {"over", "under"}):
                continue
            official = game_map.get(game_id, [])
            official_starts = [_time(g.get("kickoff")) for g in official]
            if any(start is not None and generated >= start for start in official_starts):
                continue
            if key not in frozen:
                frozen[key] = {**rec, "side": side, "line": line,
                               "odds": float(rec["odds"]), "kickoff": kickoff.isoformat(),
                               "generated_at": generated.isoformat(), "snapshot": name,
                               "model_version": board.get("model_version", "unknown")}

    results = []
    for key, bet in frozen.items():
        row = {**bet, "actual": None, "result": "unresolved", "profit_1u": None,
               "settlement_reason": "official_final_score_unavailable"}
        official = game_map.get(key[0], [])
        game = official[0] if len(official) == 1 else None
        complete = (game is not None and _number(game.get("home_score")) is not None
                    and ("away_score" not in game or _number(game.get("away_score")) is not None))
        if complete:
            player_rows = stat_map.get(key, [])
            actuals = {_number(s.get("attempts")) for s in player_rows}
            if len(actuals) == 1 and None not in actuals and next(iter(actuals)) >= 0 and next(iter(actuals)).is_integer():
                actual = actuals.pop()
                win = actual > bet["line"] if bet["side"] == "over" else actual < bet["line"]
                result = "push" if actual == bet["line"] else "win" if win else "loss"
                row.update(actual=actual, result=result, settlement_reason="official_player_stat",
                           profit_1u=0. if result == "push" else _decimal(bet["odds"]) - 1 if win else -1.)
            else:
                row["settlement_reason"] = "player_stat_missing_or_conflicting"
        row.update(_closing(bet, quotes or []))
        results.append(row)
    versions = sorted({str(r["model_version"]) for r in results})
    return {**_summary(results), "results": results, "diagnostics": diagnostics,
            "record_policy": "first_pregame_recommendation_per_game_player",
            "stake_policy": "flat_one_unit_per_recommendation",
            "clv_definition": "median_paired_fair_probability_times_bet_decimal_minus_one_conditional_on_no_push",
            "by_model_version": {v: _summary([r for r in results if str(r["model_version"]) == v]) for v in versions}}
