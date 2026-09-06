"""Public ESPN depth-chart context, with conservative starter identification.

``available`` means context was parsed, not that a player is medically cleared.
The feed timestamp is its payload timestamp, not a clinical report update time.
An empty injury list means no injury listed by this source; it is not clearance.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from .odds_sources import NFL_TEAMS, normalize_team

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _iso(value) -> str | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat() if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def _injuries(athlete: dict) -> list[dict]:
    rows = []
    for injury in athlete.get("injuries", []) or []:
        if not isinstance(injury, dict):
            rows.append({"status": "Unspecified injury", "date": None})
            continue
        status = injury.get("status") or (injury.get("type") or {}).get("description") or "Unspecified injury"
        rows.append({"status": str(status), "date": _iso(injury.get("date"))})
    return rows


def parse_depthchart(payload: dict, team: str, *, observed_at: str, source_url: str, expected_season: int | None = None) -> dict:
    observed = _iso(observed_at)
    now = datetime.fromisoformat(observed) if observed else datetime.now(timezone.utc)
    # January/February are still the prior NFL season.
    season = expected_season if expected_season is not None else now.year - (now.month <= 2)
    result = {"starter": None, "espn_id": None, "injuries": [], "offense_injuries": [],
              "observed_at": observed_at, "source_url": source_url,
              "source_updated_at": _iso(payload.get("timestamp")), "available": False,
              "season": (payload.get("season") or {}).get("year"), "issues": []}
    if not observed or payload.get("status") != "success":
        result["issues"].append("Missing observation time or unsuccessful source response")
        return result
    if str((payload.get("season") or {}).get("year")) != str(season):
        result["issues"].append(f"Source season does not match expected {season}")
        return result
    if team not in NFL_TEAMS or normalize_team((payload.get("team") or {}).get("abbreviation")) != team:
        result["issues"].append("Source team does not match requested team")
        return result
    if result["source_updated_at"] is None:
        result["issues"].append("Source payload timestamp missing or invalid")
        return result
    age_hours = (now - datetime.fromisoformat(result["source_updated_at"])).total_seconds() / 3600
    if age_hours > 48 or age_hours < -1:
        result["issues"].append("Source payload timestamp is stale or in the future")
        return result
    candidates, offense = {}, {}
    invalid_qb_chart = False
    for chart in payload.get("depthchart", []) or []:
        if not isinstance(chart, dict):
            continue
        for position, role in (chart.get("positions") or {}).items():
            if not isinstance(role, dict):
                continue
            position = position.lower()
            athletes = role.get("athletes") or []
            if position == "qb":
                if not isinstance(athletes, list) or not athletes or not isinstance(athletes[0], dict):
                    invalid_qb_chart = True
                    continue
                first = athletes[0]
                identity = str(first.get("id") or "")
                name = str(first.get("displayName") or "").strip()
                if not identity or not name:
                    invalid_qb_chart = True
                    continue
                candidates[identity] = first
            if not re.fullmatch(r"(?:wr\d*|te\d*|lt|lg|c|rg|rt|ol|rb\d*|fb)", position):
                continue
            for depth, athlete in enumerate(athletes):
                if not isinstance(athlete, dict):
                    continue
                # Keep all listed offense injuries with depth so downstream can
                # distinguish a starting tackle from a reserve on long-term IR.
                for injury in _injuries(athlete):
                    name = str(athlete.get("displayName") or "Unknown player")
                    key = (str(athlete.get("id")), position, injury["status"])
                    offense[key] = {"player": name, "position": position.upper(), "depth": depth + 1, **injury}
    result["offense_injuries"] = list(offense.values())
    if invalid_qb_chart or len(candidates) != 1:
        result["issues"].append("No unambiguous first-listed QB across depth charts")
        return result
    starter = next(iter(candidates.values()))
    result.update(starter=starter["displayName"], espn_id=str(starter["id"]),
                  injuries=sorted({i["status"] for i in _injuries(starter)}), available=True)
    return result


def _fetch_team(team: str, directory: Path) -> dict:
    espn_team = {"LA": "LAR", "WAS": "WSH"}.get(team, team)
    url = f"{BASE_URL}/{espn_team}/depthcharts"
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            observed = datetime.now(timezone.utc).isoformat()
            with (directory / f"{team}.{attempt}.body.json").open("xb") as handle:
                handle.write(response.content)
            with (directory / f"{team}.{attempt}.meta.json").open("x") as handle:
                json.dump({"url": response.url, "observed_at": observed, "status_code": response.status_code}, handle, indent=2)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("unexpected depth-chart schema")
            return parse_depthchart(payload, team, observed_at=observed, source_url=url)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def fetch_context(teams: list[str], output_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """Read ESPN's current depth chart; failures remain unavailable, never healthy."""
    snapshot = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    snapshot.mkdir(parents=True, exist_ok=False)
    contexts, errors = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for team in sorted({normalize_team(t) for t in teams}):
            if team not in NFL_TEAMS:
                errors.append(f"ESPN {team}: unknown NFL team")
                continue
            futures[pool.submit(_fetch_team, team, snapshot)] = team
        for future in as_completed(futures):
            team = futures[future]
            try:
                contexts[team] = future.result()
                if not contexts[team]["available"]:
                    errors.append(f"ESPN {team}: " + "; ".join(contexts[team]["issues"]))
            except (RuntimeError, requests.RequestException, ValueError, TypeError) as exc:
                errors.append(f"ESPN {team}: {exc}")
                contexts[team] = {"starter": None, "espn_id": None, "injuries": [], "offense_injuries": [],
                                  "available": False, "source_updated_at": None,
                                  "observed_at": datetime.now(timezone.utc).isoformat(),
                                  "source_url": f"{BASE_URL}/{ {'LA': 'LAR', 'WAS': 'WSH'}.get(team, team) }/depthcharts",
                                  "issues": ["Context fetch failed"]}
    with (snapshot / "context.json").open("x") as handle:
        json.dump(contexts, handle, indent=2)
    with (snapshot / "errors.json").open("x") as handle:
        json.dump(sorted(errors), handle, indent=2)
    return contexts, sorted(errors)
