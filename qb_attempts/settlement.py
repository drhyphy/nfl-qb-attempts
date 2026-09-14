"""Evidence-backed paper grading, separate from immutable prediction history.

Missing passing rows never imply zero attempts or nonparticipation. A confirmed
DNP is a provisional paper void; actual sportsbook settlement remains unknown.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import unicodedata

import requests


def normalized_name(value):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def normalize_espn_summary(payload, game_id, observed_at):
    """Normalize a retained ESPN summary without treating scores as final flags."""
    if timestamp(observed_at) is None:
        raise ValueError("observed_at must include timezone")
    header = payload.get("header") or {}
    competitions = header.get("competitions") or []
    competition = competitions[0] if len(competitions) == 1 else {}
    status = (competition.get("status") or {}).get("type") or {}
    rows = []
    for team in (payload.get("boxscore") or {}).get("players", []):
        for group in team.get("statistics", []):
            if group.get("name") != "passing":
                continue
            labels = group.get("labels") or []
            keys = group.get("keys") or []
            for item in group.get("athletes", []):
                athlete = item.get("athlete") or {}
                stats = item.get("stats") or []
                values = dict(zip(labels, stats))
                values.update(zip(keys, stats))
                value = values.get("C/ATT", values.get("completions/passingAttempts"))
                attempts = None
                if value is not None and re.fullmatch(r"\d+\s*/\s*\d+", str(value)):
                    attempts = int(str(value).split("/")[1])
                elif "passingAttempts" in values and str(values["passingAttempts"]).isdigit():
                    attempts = int(values["passingAttempts"])
                # A passing row with positive attempts proves participation.
                # A zero row alone does not establish that the player played.
                rows.append({"espn_id": str(athlete.get("id") or ""),
                             "player": athlete.get("displayName"),
                             "name_key": normalized_name(athlete.get("displayName")),
                             "attempts": attempts,
                             "participated": True if attempts is not None and attempts > 0 else None,
                             "confirmed_dnp": False})
    return {"game_id": str(game_id), "event_id": str(header.get("id") or competition.get("id") or ""),
            "final": status.get("completed") is True and status.get("state") == "post",
            "status": status.get("name"), "observed_at": observed_at,
            "source": "espn_summary", "players": rows}


def fetch_espn_evidence(event_id, game_id, raw_dir, *, session=None):
    """Read-only network fetch; retain response and provenance for later audit."""
    event_id = str(event_id)
    if not event_id.isdigit():
        raise ValueError("ESPN event_id must be numeric")
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
    response = (session or requests).get(url, timeout=30)
    now = datetime.now(timezone.utc)
    directory = Path(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{event_id}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    (directory / f"{stem}.body.json").write_bytes(response.content)
    (directory / f"{stem}.meta.json").write_text(json.dumps({"url": url,
        "observed_at": now.isoformat(), "status_code": response.status_code}, indent=2))
    response.raise_for_status()
    normalized = normalize_espn_summary(response.json(), game_id, now.isoformat())
    normalized["source_url"] = url
    normalized["raw_path"] = str(directory / f"{stem}.body.json")
    return normalized


def settle_entry(entry, game_evidence, *, as_of=None):
    """Grade one frozen paper price using final and participation evidence."""
    answer = {"result": "pending", "actual": None, "profit_1u": None,
              "settlement_reason": "final_game_evidence_unavailable", "actual_wager": False}
    evidence = game_evidence or {}
    observed = timestamp(evidence.get("observed_at"))
    now = timestamp(as_of) if as_of is not None else datetime.now(timezone.utc)
    entry_time = timestamp(entry.get("entry_at", entry.get("generated_at")))
    if (evidence.get("game_id") != entry.get("game_id") or evidence.get("final") is not True
            or observed is None or now is None or observed > now
            or (entry_time is not None and observed < entry_time)):
        return answer
    rows = evidence.get("players") or []
    # Match an available ID first. Name matching must be unique and is retained.
    matches = [r for r in rows if entry.get("espn_id") and str(r.get("espn_id")) == str(entry["espn_id"])]
    if not matches:
        matches = [r for r in rows if entry.get("player_id") and r.get("player_id") == entry["player_id"]]
    if not matches:
        key = normalized_name(entry.get("player"))
        matches = [r for r in rows if key and normalized_name(r.get("player")) == key]
    answer.update(evidence_observed_at=evidence.get("observed_at"), evidence_source=evidence.get("source_url", evidence.get("source")))
    if len(matches) != 1:
        answer["settlement_reason"] = "player_stat_missing_or_ambiguous"
        return answer
    row = matches[0]
    if row.get("confirmed_dnp") is True and row.get("participated") is False:
        if not row.get("dnp_evidence"):
            answer["settlement_reason"] = "dnp_evidence_missing"
            return answer
        answer.update(result="void", profit_1u=0., settlement_reason="confirmed_dnp_provisional_paper_void",
                      sportsbook_rules_confirmed=False, dnp_evidence=row["dnp_evidence"])
        return answer
    actual = row.get("attempts")
    if (isinstance(actual, bool) or not isinstance(actual, (float, int))
            or not math.isfinite(actual) or actual < 0 or int(actual) != actual
            or row.get("participated") is not True):
        answer["settlement_reason"] = "participation_or_attempts_unconfirmed"
        return answer
    line, odds, side = entry.get("line"), entry.get("odds"), str(entry.get("side", "")).lower()
    if (isinstance(line, bool) or not isinstance(line, (int, float)) or not math.isfinite(line)
            or isinstance(odds, bool) or not isinstance(odds, (int, float)) or not math.isfinite(odds)
            or abs(odds) < 100 or side not in {"over", "under"}):
        answer["settlement_reason"] = "invalid_frozen_quote"
        return answer
    win = actual > line if side == "over" else actual < line
    result = "push" if actual == line else "win" if win else "loss"
    answer.update(result=result, actual=actual, settlement_reason="final_participating_player_stat",
                  profit_1u=0. if result == "push" else (odds / 100 if odds > 0 else 100 / -odds) if win else -1.)
    return answer


def summarize_results(rows):
    settled = [r for r in rows if r.get("result") in {"win", "loss", "push"}]
    profit = sum(r.get("profit_1u") or 0. for r in settled)
    return {"bets": len(rows), "settled": len(settled), "wins": sum(r.get("result") == "win" for r in rows),
            "losses": sum(r.get("result") == "loss" for r in rows), "pushes": sum(r.get("result") == "push" for r in rows),
            "voids": sum(r.get("result") == "void" for r in rows),
            "unresolved": sum(r.get("result") in {"pending", "unresolved"} for r in rows),
            "units": profit, "roi": profit / len(settled) if settled else None}


def augment_results(research_summary, evidence_by_game, *, as_of=None):
    """Reconcile a derived summary, leaving every frozen publication untouched."""
    result = deepcopy(research_summary)
    rows = result.get("results", [])
    for row in rows:
        evidence = evidence_by_game.get(row.get("game_id"))
        if evidence is not None:
            row.update(settle_entry(row, evidence, as_of=as_of))
    result.update(summarize_results(rows))
    versions = sorted({str(r.get("model_version", "unknown")) for r in rows})
    result["by_model_version"] = {v: summarize_results([r for r in rows if str(r.get("model_version", "unknown")) == v]) for v in versions}
    result["settlement_policy"] = "final_participation_evidence_v1; DNP paper void pending sportsbook rules"
    return result


def fetch_pending_evidence(research_summaries, ledger, games, raw_dir, *, now=None, session=None):
    """Fetch ended/past-kickoff unresolved games only, preserving each raw reply.

    games accepts a dataframe or list of canonical schedule dictionaries; espn
    is the nflverse ESPN event identifier. Returns (evidence_by_game, errors).
    """
    current = timestamp(now) if now is not None else datetime.now(timezone.utc)
    if current is None:
        raise ValueError("timezone-aware now required")
    schedules = games.to_dict("records") if hasattr(games, "to_dict") else games
    unresolved = {row.get("game_id") for summary in research_summaries for row in summary.get("results", [])
                  if row.get("result") in {"pending", "unresolved"}}
    unresolved.update(e["game_id"] for e in (ledger or {}).get("entries", [])
                      if e.get("settlement", {}).get("result", "pending") == "pending")
    evidence, errors = {}, []
    for game in schedules:
        game_id = game.get("game_id")
        kickoff = timestamp(game.get("kickoff"))
        if game_id not in unresolved or kickoff is None or kickoff >= current:
            continue
        event_id = str(game.get("espn") or game.get("event_id") or "")
        if event_id.endswith(".0"):
            event_id = event_id[:-2]
        if not event_id.isdigit():
            errors.append(f"{game_id}: missing ESPN event id")
            continue
        try:
            evidence[game_id] = fetch_espn_evidence(event_id, game_id, raw_dir, session=session)
        except (ValueError, OSError, requests.RequestException) as exc:
            errors.append(f"{game_id}: settlement evidence unavailable: {exc}")
    return evidence, errors
