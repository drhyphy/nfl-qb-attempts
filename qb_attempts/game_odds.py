"""Timestamped public main-game spreads/totals for prospective game-state inputs.

ESPN's public scoreboard labels its current quote ``close`` even before kickoff.
We use that field only for explicitly scheduled, future games, never its ``open``
field or nflverse's historical closing lines. Both priced sides must be present.
The spread sign comes from the home handicap: -3.5 becomes a +3.5 expected home
margin. The top-level ESPN ``spread`` alone does not identify the favored team.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import uuid4

import pandas as pd
import requests

from .odds_sources import NFL_TEAMS, NON_SPORTSBOOKS, _odds, _timestamp, canonical_book, normalize_team

SCOREBOARD_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
HORIZON = pd.Timedelta(days=9)


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _stamp(value) -> pd.Timestamp | None:
    parsed = _timestamp(value)
    return pd.Timestamp(parsed) if parsed else None


def _kickoff(game: dict) -> pd.Timestamp | None:
    if game.get("kickoff") is not None:
        stamp = _stamp(game["kickoff"])
        if stamp is not None:
            return stamp
    try:
        if pd.isna(game.get("gameday")) or pd.isna(game.get("gametime")):
            return None
        return pd.Timestamp(f"{game['gameday']} {game['gametime']}", tz="America/New_York").tz_convert("UTC")
    except (KeyError, TypeError, ValueError):
        return None


def _eligible_games(games: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    eligible = []
    for game in games.to_dict("records"):
        kickoff = _kickoff(game)
        home, away = normalize_team(game.get("home_team")), normalize_team(game.get("away_team"))
        if kickoff is None or not now < kickoff <= now + HORIZON:
            continue
        # Do not use games already recorded as played, even if a date is wrong.
        if any(pd.notna(game.get(field)) for field in ("home_score", "away_score", "result")):
            continue
        if not game.get("game_id") or home not in NFL_TEAMS or away not in NFL_TEAMS or home == away:
            continue
        eligible.append(dict(game, home_team=home, away_team=away, kickoff=kickoff))
    return eligible


def _line(value, prefix: str = "") -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if prefix:
        if not text.lower().startswith(prefix):
            return None
        text = text[1:]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    number = float(text)
    return number if math.isfinite(number) and number * 2 == int(number * 2) else None


def _book_quote(odds: dict, home: str, away: str) -> dict | None:
    if not isinstance(odds, dict) or odds.get("isLive") or odds.get("is_live") or odds.get("available") is False:
        return None
    provider = _mapping(odds.get("provider"))
    name = provider.get("name") or provider.get("displayName")
    if not isinstance(name, str) or not name.strip():
        return None
    book = canonical_book(name)
    if not book or book in NON_SPORTSBOOKS:
        return None
    for field, team in (("homeTeamOdds", home), ("awayTeamOdds", away)):
        identity = _mapping(_mapping(odds.get(field)).get("team")).get("abbreviation")
        if identity and normalize_team(identity) != team:
            return None
    spread, total = odds.get("pointSpread") or {}, odds.get("total") or {}
    try:
        home_quote, away_quote = spread["home"]["close"], spread["away"]["close"]
        over_quote, under_quote = total["over"]["close"], total["under"]["close"]
        home_line, away_line = _line(home_quote.get("line")), _line(away_quote.get("line"))
        over_line, under_line = _line(over_quote.get("line"), "o"), _line(under_quote.get("line"), "u")
        prices = [_odds(q.get("odds")) for q in (home_quote, away_quote, over_quote, under_quote)]
    except (KeyError, TypeError, AttributeError):
        return None
    if any(value is None for value in (home_line, away_line, over_line, under_line, *prices)):
        return None
    if home_line != -away_line or over_line != under_line or abs(home_line) > 35 or not 15 <= over_line <= 90:
        return None
    # Reject contradictions when provider also identifies a favorite. Do not
    # require those optional flags, because signed paired handicaps suffice.
    for field, favored in (("homeTeamOdds", home_line < 0), ("awayTeamOdds", away_line < 0)):
        flag = _mapping(odds.get(field)).get("favorite")
        if home_line != 0 and isinstance(flag, bool) and flag != favored:
            return None
    return {"book": book, "home_expected_margin": -home_line, "total_line": over_line,
            "home_spread_odds": prices[0], "away_spread_odds": prices[1],
            "over_odds": prices[2], "under_odds": prices[3],
            "source_updated_at": _timestamp(odds.get("lastUpdated") or odds.get("lastUpdatedAt"))}


def parse_scoreboard(payload: dict, games: pd.DataFrame, *, observed_at: str,
                     source_url: str = SCOREBOARD_URL, now: pd.Timestamp | None = None) -> tuple[dict[str, dict], list[str]]:
    """Match provider ID, ordered teams and kickoff to the canonical schedule.

    Team/time matching is permitted when nflverse has no ESPN identifier, but
    an existing identifier may never be silently ignored. Ambiguity fails closed.
    Observation time is an HTTP retrieval time, not an asserted book-update time.
    """
    observed = _stamp(observed_at)
    if observed is None:
        return {}, ["ESPN game markets: missing or timezone-naive observation timestamp"]
    effective_now = max(observed, _stamp(now) or observed)
    eligible = _eligible_games(games, effective_now)
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return {}, ["ESPN game markets: unexpected scoreboard schema"]
    contexts, errors, ambiguous = {}, [], set()
    for event in payload["events"]:
        if not isinstance(event, dict):
            continue
        for competition in event.get("competitions", []) or []:
            if not isinstance(competition, dict):
                continue
            status = _mapping(_mapping(competition.get("status") or event.get("status")).get("type"))
            if status.get("state") != "pre" or status.get("completed") is not False or status.get("name") != "STATUS_SCHEDULED":
                continue
            kickoff = _stamp(competition.get("date") or event.get("date"))
            if kickoff is None or not effective_now < kickoff <= effective_now + HORIZON or competition.get("timeValid") is False:
                continue
            competitors = competition.get("competitors") or []
            teams = {side: [normalize_team(_mapping(c.get("team")).get("abbreviation")) for c in competitors
                            if isinstance(c, dict) and c.get("homeAway") == side] for side in ("home", "away")}
            if len(teams["home"]) != 1 or len(teams["away"]) != 1:
                continue
            home, away = teams["home"][0], teams["away"][0]
            matches = []
            for game in eligible:
                if game["home_team"] != home or game["away_team"] != away or abs(game["kickoff"] - kickoff) > pd.Timedelta(minutes=5):
                    continue
                espn_id = game.get("espn")
                if pd.notna(espn_id) and str(espn_id).strip() and str(espn_id) != str(event.get("id")):
                    continue
                matches.append(game)
            if len(matches) != 1:
                continue
            game_id = str(matches[0]["game_id"])
            quotes = {}
            conflicting_books = set()
            for odds in competition.get("odds", []) or []:
                quote = _book_quote(odds, home, away)
                if quote:
                    book = quote["book"]
                    if book in quotes and quote != quotes[book]:
                        conflicting_books.add(book)
                    quotes[book] = quote
            quotes = {book: q for book, q in quotes.items() if book not in conflicting_books}
            if not quotes:
                continue
            rows = [quotes[book] for book in sorted(quotes)]
            updates = [q["source_updated_at"] for q in rows]
            record = {"game_id": game_id, "event_id": str(event.get("id")), "home_team": home, "away_team": away,
                      "kickoff": kickoff.isoformat(), "home_expected_margin": float(median(q["home_expected_margin"] for q in rows)),
                      "total_line": float(median(q["total_line"] for q in rows)), "books": sorted(quotes), "book_quotes": rows,
                      "spread_observed_at": observed.isoformat(), "total_observed_at": observed.isoformat(),
                      "observed_at": observed.isoformat(), "source_url": source_url, "source": "espn",
                      "source_type": "public_pregame_sportsbook", "source_updated_at": min(updates) if all(updates) else None}
            if game_id in contexts and record != contexts[game_id]:
                ambiguous.add(game_id)
            contexts[game_id] = record
    for game_id in ambiguous:
        contexts.pop(game_id, None)
    for game in eligible:
        if str(game["game_id"]) not in contexts:
            errors.append(f"ESPN {game['game_id']}: no unambiguous paired pregame spread and total matched to schedule")
    return contexts, sorted(set(errors))


def fetch_game_markets(games: pd.DataFrame, output_dir: Path, now: pd.Timestamp | None = None) -> tuple[dict[str, dict], list[str]]:
    """Fetch the next nine days, retaining immutable source bodies and metadata."""
    current = _stamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if current is None:
        return {}, ["ESPN game markets: now must include a timezone"]
    if not _eligible_games(games, current):
        return {}, []
    snapshot = Path(output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    snapshot.mkdir(parents=True, exist_ok=False)
    params = {"dates": f"{current:%Y%m%d}-{current + HORIZON:%Y%m%d}", "limit": 100}
    contexts, errors = {}, []
    for attempt in range(2):
        try:
            response = requests.get(SCOREBOARD_URL, params=params, headers=HEADERS, timeout=30)
            observed = datetime.now(timezone.utc).isoformat()
            with (snapshot / f"scoreboard.{attempt}.body.json").open("xb") as handle:
                handle.write(response.content)
            with (snapshot / f"scoreboard.{attempt}.meta.json").open("x") as handle:
                json.dump({"url": response.url, "observed_at": observed, "status_code": response.status_code,
                           "http_date": response.headers.get("Date"), "content_type": response.headers.get("Content-Type")}, handle, indent=2)
            response.raise_for_status()
            contexts, errors = parse_scoreboard(response.json(), games, observed_at=observed, source_url=response.url, now=current)
            break
        except (requests.RequestException, ValueError, TypeError) as exc:
            errors = [f"ESPN game markets fetch failed: {exc}"]
            if isinstance(exc, requests.HTTPError) and response.status_code in (400, 401, 403, 404):
                break
    with (snapshot / "game_markets.json").open("x") as handle:
        json.dump(contexts, handle, indent=2)
    with (snapshot / "errors.json").open("x") as handle:
        json.dump(errors, handle, indent=2)
    return contexts, errors
