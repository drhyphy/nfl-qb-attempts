"""Public paired, full-game, pregame QB passing-attempts sportsbook quotes.

ScoresAndOdds exposes the comparison JSON used by its public odds table. The
table slug is ``pass-attempts`` but the comparison stat is ``pass attempts``.
The public schedule's JSON-LD supplies dates absent from comparison responses.
Raw HTTP bodies and request metadata are retained in a unique snapshot folder.
No sportsbook source is guaranteed current merely because it was just fetched.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

BOARD_URL = "https://www.scoresandodds.com/nfl/props/pass-attempts"
SCHEDULE_URL = "https://www.scoresandodds.com/nfl"
COMPARISON_URL = "https://rga51lus77.execute-api.us-east-1.amazonaws.com/prod/market-comparison"
MARKET = "pass attempts"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "Accept": "text/html,application/json,text/plain,*/*",
    "Referer": BOARD_URL,
}
NFL_TEAMS = set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split())
NON_SPORTSBOOKS = {"prizepicks", "sleeper", "underdog", "pick6", "draftkingspick6", "consensus", "open", "underdogfantasy"}


def normalize_team(value: Any) -> str:
    key = str(value or "").upper().strip()
    return {"LAR": "LA", "STL": "LA", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "OAK": "LV"}.get(key, key)


def _name_key(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonical_book(value: Any) -> str:
    key = _name_key(value)
    return {"riverscasino": "betrivers", "williamhill": "caesars", "williamhillus": "caesars", "hardrockbet": "hardrock", "espnbet": "thescorebet"}.get(key, key)


def _timestamp(value: Any) -> str | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            stamp = datetime.fromtimestamp(value, timezone.utc)
        else:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                return None
        return stamp.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _odds(value: Any) -> int | None:
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number != int(number) or abs(number) < 100:
            return None
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class BoardEntry:
    event: str
    player: str


def parse_board_html(html: str) -> list[BoardEntry]:
    entries = {}
    soup = BeautifulSoup(html, "html.parser")
    for row in soup.select("li.border[data-name][data-state]"):
        if str(row.get("data-state", "")).lower() != "pregame":
            continue
        chassis = row.select_one('[data-role="chassis"][data-event][data-market="pass attempts"]')
        if chassis is None:
            continue
        event, player = str(chassis["data-event"]), str(chassis.get("data-filter", "")).strip()
        if not re.fullmatch(r"nfl/\d+", event) or not player:
            continue
        entries[(event, _name_key(player))] = BoardEntry(event, player)
    return list(entries.values())


def parse_schedule_html(html: str) -> dict[str, dict]:
    events = {}
    for script in BeautifulSoup(html, "html.parser").select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.get_text())
        except (ValueError, TypeError):
            continue
        candidates = payload if isinstance(payload, list) else payload.get("@graph", [payload]) if isinstance(payload, dict) else []
        for event in candidates:
            if not isinstance(event, dict) or event.get("@type") != "SportsEvent":
                continue
            if event.get("eventStatus") != "https://schema.org/EventScheduled":
                continue
            kickoff = _timestamp(event.get("startDate"))
            identifier = str(event.get("identifier", ""))
            home = normalize_team(str((event.get("homeTeam") or {}).get("name", "")).split(" ")[0])
            away = normalize_team(str((event.get("awayTeam") or {}).get("name", "")).split(" ")[0])
            if kickoff and identifier.isdigit() and home in NFL_TEAMS and away in NFL_TEAMS and home != away:
                events[f"nfl/{identifier}"] = {"kickoff": kickoff, "home_team": home, "away_team": away, "event_url": event.get("url")}
    return events


def parse_comparison_payload(payload: dict, entry: BoardEntry, schedule: dict[str, dict], *, observed_at: str) -> list[dict]:
    """Reject uncertain identity, dates, partial/live markets and unpaired prices.

    ``market.date`` is used only for its named primary sportsbook. Applying it
    to other books would falsely assert freshness for their independent quotes.
    """
    context = schedule.get(entry.event)
    now = _timestamp(observed_at)
    if not context or not now or context["kickoff"] <= now:
        return []
    event = payload.get("event") or {}
    home = normalize_team((event.get("home") or {}).get("key"))
    away = normalize_team((event.get("away") or {}).get("key"))
    if event.get("sport") != "nfl" or home != context["home_team"] or away != context["away_team"]:
        return []
    rows = []
    for market in payload.get("markets", []) or []:
        if not isinstance(market, dict) or market.get("stat") != MARKET or market.get("available") is not True:
            continue
        # The provider's period component 0 means full game.
        prefix = entry.event.replace("/", ".") + ".0."
        if not str(market.get("id", "")).startswith(prefix) or market.get("is_live") or market.get("line_status", "normal") != "normal":
            continue
        player = market.get("player") or {}
        name = " ".join(str(player.get(k) or "").strip() for k in ("first_name", "last_name")).strip()
        team = normalize_team((player.get("team") or {}).get("key"))
        if _name_key(name) != _name_key(entry.player) or team not in (home, away):
            continue
        for raw_book, quote in (market.get("comparison") or {}).items():
            book = canonical_book(raw_book)
            if not book or book in NON_SPORTSBOOKS or not isinstance(quote, dict) or quote.get("available") is not True:
                continue
            if quote.get("is_live") or quote.get("line_status", "normal") != "normal":
                continue
            over, under = _odds(quote.get("over")), _odds(quote.get("under"))
            try:
                line = float(quote.get("value"))
            except (TypeError, ValueError):
                continue
            if over is None or under is None or isinstance(quote.get("value"), bool) or not math.isfinite(line) or not 0 < line < 100 or line * 2 != int(line * 2):
                continue
            updated = _timestamp(quote.get("date"))
            if updated is None and quote.get("sportsbook") is not None and quote.get("sportsbook") == market.get("sportsbook"):
                updated = _timestamp(market.get("date"))
            rows.append({
                "player": name, "team": team, "opponent": away if team == home else home,
                "home_team": home, "away_team": away, "line": line, "over_odds": over, "under_odds": under,
                "book": book, "source": "scoresandodds", "source_url": BOARD_URL,
                "observed_at": now, "source_updated_at": updated,
                "event_id": entry.event, "kickoff": context["kickoff"],
            })
    return rows


def _fetch_saved(url: str, directory: Path, name: str, *, params: dict | None = None) -> tuple[str, str]:
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            observed = datetime.now(timezone.utc).isoformat()
            # Preserve unsuccessful HTTP bodies too. Exclusive creation prevents overwrite.
            with (directory / f"{name}.{attempt}.body").open("xb") as handle:
                handle.write(response.content)
            with (directory / f"{name}.{attempt}.meta.json").open("x") as handle:
                json.dump({"url": response.url, "observed_at": observed, "status_code": response.status_code,
                           "http_date": response.headers.get("Date"), "content_type": response.headers.get("Content-Type")}, handle, indent=2)
            response.raise_for_status()
            return response.text, observed
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def fetch_quotes(output_dir: Path) -> tuple[list[dict], list[str]]:
    """Fetch current publicly listed quotes; never fall back to a stale snapshot."""
    snapshot = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    snapshot.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    try:
        board, _ = _fetch_saved(BOARD_URL, snapshot, "board")
        schedule_html, _ = _fetch_saved(SCHEDULE_URL, snapshot, "schedule")
        entries, schedule = parse_board_html(board), parse_schedule_html(schedule_html)
    except (requests.RequestException, RuntimeError) as exc:
        return [], [f"ScoresAndOdds public board/schedule failed: {exc}"]
    if not entries or not schedule:
        return [], ["ScoresAndOdds contained no pregame pass-attempts players or dated scheduled events"]
    quotes = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for entry in entries:
            if entry.event not in schedule:
                errors.append(f"ScoresAndOdds {entry.player}: no dated event matching {entry.event}")
                continue
            name = entry.event.replace("/", "_") + "_" + _name_key(entry.player)
            future = pool.submit(_fetch_saved, COMPARISON_URL, snapshot, name,
                                 params={"event": entry.event, "market": MARKET, "filter": entry.player})
            futures[future] = entry
        for future in as_completed(futures):
            entry = futures[future]
            try:
                body, observed = future.result()
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
                    raise ValueError("unexpected comparison response schema")
                rows = parse_comparison_payload(payload, entry, schedule, observed_at=observed)
                quotes.extend(rows)
                if not rows:
                    errors.append(f"ScoresAndOdds {entry.player}: no valid paired pregame sportsbook quotes")
            except (requests.RequestException, RuntimeError, ValueError, TypeError) as exc:
                errors.append(f"ScoresAndOdds {entry.player}: {exc}")
    unique = {(q["event_id"], q["player"], q["book"], q["line"]): q for q in quotes}
    quotes = sorted(unique.values(), key=lambda q: (q["kickoff"], q["player"], q["book"], q["line"]))
    with (snapshot / "quotes.json").open("x") as handle:
        json.dump(quotes, handle, indent=2)
    with (snapshot / "errors.json").open("x") as handle:
        json.dump(sorted(errors), handle, indent=2)
    return quotes, sorted(errors)
