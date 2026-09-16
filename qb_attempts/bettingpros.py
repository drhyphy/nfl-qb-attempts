"""Timestamped NY comparison quotes from BettingPros' public web API.

This is comparison evidence, not an independent direct sportsbook check. The
provider's naive scheduled/updated fields use UTC (the response identifies its
clock as ``utc`` alongside Unix ``ts``); convert that provider format explicitly.
Provider offer/line IDs are retained with their namespace, never represented as
native sportsbook identifiers. Missing availability flags are not affirmative.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import requests

from .odds_sources import normalize_team

BASE = "https://api.bettingpros.com/v3"
MARKET = 333
PUBLIC_WEB_KEY = "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "x-api-key": PUBLIC_WEB_KEY,
           "Referer": "https://www.bettingpros.com/", "Cache-Control": "no-cache"}
BOOKS = {12: "draftkings", 10: "fanduel", 19: "betmgm", 13: "caesars", 33: "thescorebet",
         14: "fanatics", 18: "betrivers", 67: "ballybet"}


def _time(value, *, provider_utc=False):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            if not provider_utc:
                return None
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _number(value):
    try:
        result = float(value)
        return result if not isinstance(value, bool) and math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def match_events(payload, games, now):
    """Map provider events only by REG season/week, ordered teams and exact start."""
    current = _time(now)
    if current is None:
        raise ValueError("now must be timezone-aware")
    records = games.to_dict("records") if hasattr(games, "to_dict") else games
    matched, errors, seen_ids = {}, [], set()
    for event in payload.get("events", []):
        if event.get("sport") != "NFL" or event.get("status") != "scheduled" or event.get("season_type") != "REG":
            continue
        kickoff = _time(event.get("scheduled"), provider_utc=True)
        if kickoff is None or not current < kickoff <= current + timedelta(days=9):
            continue
        candidates = [game for game in records
                      if game.get("game_type", game.get("season_type")) == "REG"
                      and str(game.get("season")) == str(event.get("season"))
                      and str(game.get("week")) == str(event.get("week"))
                      and normalize_team(game.get("home_team")) == normalize_team(event.get("home"))
                      and normalize_team(game.get("away_team")) == normalize_team(event.get("visitor"))
                      and _time(game.get("kickoff")) == kickoff]
        if len(candidates) != 1 or not str(event.get("id") or "").isdigit():
            errors.append(f"BettingPros {event.get('id')}: no unique exact official schedule match")
            continue
        event_id = str(event["id"])
        if event_id in seen_ids:
            matched.pop(event_id, None)
            errors.append(f"BettingPros {event_id}: duplicate provider event")
            continue
        seen_ids.add(event_id)
        matched[event_id] = {**candidates[0], "provider_event_id": event_id,
                             "kickoff": kickoff.isoformat()}
    return matched, errors


def parse_bettingpros_offers(payload, event, observed_at, *, proof_path=None, proof_sha256=None, source_url=None):
    """Return exact paired main quotes and side-specific verification evidence."""
    observed = _time(observed_at)
    kickoff = _time(event.get("kickoff"))
    params = payload.get("_parameters") or {}
    if (observed is None or kickoff is None or kickoff <= observed or params.get("location") != "NY"
            or str(params.get("event_id")) != str(event.get("provider_event_id"))
            or str(params.get("market_id")) != str(MARKET) or params.get("sport") != "NFL"
            or params.get("live") not in (False, "false")):
        return [], [], ["BettingPros offer response scope or observation invalid"]
    quotes, evidence, errors = [], [], []
    for offer in payload.get("offers", []):
        participants = offer.get("participants") or []
        if (offer.get("active") is not True or offer.get("market_id") != MARKET
                or str(offer.get("event_id")) != str(event["provider_event_id"])
                or not offer.get("id") or len(participants) != 1):
            continue
        part = participants[0]
        player = part.get("player") or {}
        team = normalize_team(player.get("team"))
        name = str(part.get("name") or "").strip()
        if (player.get("position") != "QB" or team not in (event["home_team"], event["away_team"]) or not name
                or not str(part.get("id") or "").isdigit()
                or (offer.get("player_id") is not None and str(offer["player_id"]) != str(part["id"]))):
            continue
        common = {"player": name, "team": team, "opponent": event["away_team"] if team == event["home_team"] else event["home_team"],
                  "home_team": event["home_team"], "away_team": event["away_team"], "kickoff": kickoff.isoformat(),
                  "game_id": event["game_id"], "event_id": str(event["provider_event_id"]),
                  "provider_event_id": str(event["provider_event_id"]), "provider_player_id": str(part.get("id") or ""),
                  "source": "bettingpros", "source_url": source_url or f"{BASE}/offers?event_id={event['provider_event_id']}&market_id=333&location=NY&sport=NFL",
                  "observed_at": observed.isoformat(), "proof_path": str(proof_path) if proof_path else None,
                  "proof_sha256": proof_sha256, "market_id": str(offer["id"]),
                  "id_namespace": "bettingpros_provider", "evidence_level": "timestamped_comparison", "jurisdiction": "US-NY"}
        by_book = {}
        invalid_books = set()
        for selection in offer.get("selections", []):
            side = selection.get("selection")
            if side not in {"over", "under"}:
                continue
            for raw_book in selection.get("books", []):
                book = BOOKS.get(raw_book.get("id"))
                if book is None:
                    continue
                # Every main observation is checked, including inactive copies;
                # an active/inactive conflict cannot be resolved by first match.
                main_rows = [line for line in raw_book.get("lines", []) if line.get("main") is True]
                if selection.get("active") is not True or not selection.get("id") or len(main_rows) != 1:
                    invalid_books.add(book)
                    continue
                line = main_rows[0]
                value, price = _number(line.get("line")), _number(line.get("cost"))
                updated = _time(line.get("updated"), provider_utc=True)
                if (line.get("active") is not True or line.get("is_off") is not False or not line.get("id")
                        or value is None or value < 0 or value * 2 != int(value * 2)
                        or price is None or abs(price) < 100 or int(price) != price):
                    invalid_books.add(book)
                    continue
                row = {**common, "book": book, "line": value, "side": side, "odds": int(price),
                       "available": True, "market_status": "active", "runner_status": "active",
                       "selection_id": str(line["id"]), "provider_selection_id": str(selection["id"]),
                       "book_line_id": str(line["id"]), "sportsbook_url": line.get("link"),
                       "source_updated_at": updated.isoformat() if updated else None,
                       "source_updated_at_raw": line.get("updated"), "timestamp_timezone": "UTC"}
                sides = by_book.setdefault(book, {})
                if side in sides:
                    invalid_books.add(book)
                sides[side] = row
        for book, sides in by_book.items():
            if book in invalid_books or set(sides) != {"over", "under"}:
                continue
            over, under = sides["over"], sides["under"]
            if over["line"] != under["line"] or over["selection_id"] == under["selection_id"]:
                continue
            timestamps = [r["source_updated_at"] for r in (over, under)]
            quote = {**common, "book": book, "line": over["line"], "over_odds": over["odds"], "under_odds": under["odds"],
                     "source_updated_at": min(timestamps) if all(timestamps) else None,
                     "source_side_updated_at": {side: row["source_updated_at"] for side, row in sides.items()},
                     "sportsbook_urls": {side: row["sportsbook_url"] for side, row in sides.items()},
                     "ny_legal": True, "is_live": False, "line_status": "normal"}
            # Future/missing/old ticks stay research-only; the shared verification
            # layer rejects each side independently against its decision time.
            quotes.append(quote)
            evidence.extend((over, under))
    # Multiple offer IDs for one player/book main market are ambiguous too.
    keys = [(q["game_id"], q["provider_player_id"], q["book"]) for q in quotes]
    conflicts = {key for key in keys if keys.count(key) > 1}
    if conflicts:
        errors.append(f"BettingPros: rejected {len(conflicts)} conflicting player/book main markets")
        quotes = [q for q in quotes if (q["game_id"], q["provider_player_id"], q["book"]) not in conflicts]
        evidence = [q for q in evidence if (q["game_id"], q["provider_player_id"], q["book"]) not in conflicts]
    return quotes, evidence, errors


def _fetch(endpoint, params, directory, label):
    requested = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.get(f"{BASE}/{endpoint}", params=params, headers=HEADERS, timeout=25)
    except requests.RequestException as exc:
        (directory / f"{label}.meta.json").write_text(json.dumps({"url": f"{BASE}/{endpoint}", "params": params,
            "requested_at": requested, "failed_at": datetime.now(timezone.utc).isoformat(), "error": str(exc)}, indent=2))
        raise
    observed = datetime.now(timezone.utc).isoformat()
    body = response.content
    path = directory / f"{label}.json"
    path.write_bytes(body)
    meta = {"url": response.url, "requested_at": requested, "observed_at": observed,
            "status_code": response.status_code, "sha256": hashlib.sha256(body).hexdigest(),
            "source": "bettingpros", "params": params}
    (directory / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    response.raise_for_status()
    return response.json(), meta, path


def fetch_bettingpros_quotes(games, output_dir, now=None):
    """Fetch official eligible games within nine days and retain all raw evidence."""
    current = _time(now) if now is not None else datetime.now(timezone.utc)
    if current is None:
        raise ValueError("now must be timezone-aware")
    records = games.to_dict("records") if hasattr(games, "to_dict") else list(games)
    eligible = [g for g in records if g.get("game_type", g.get("season_type")) == "REG"
                and _time(g.get("kickoff")) is not None and current < _time(g["kickoff"]) <= current + timedelta(days=9)]
    directory = Path(output_dir) / ("bettingpros-" + uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    matches, quotes, evidence, errors = {}, [], [], []
    for season, week in sorted({(int(g["season"]), int(g["week"])) for g in eligible}):
        try:
            payload, _, _ = _fetch("events", {"sport": "NFL", "season": season, "week": week, "season_type": "REG"}, directory, f"events-{season}-{week}")
            rows, issues = match_events(payload, eligible, current)
            matches.update(rows)
            errors.extend(issues)
        except (requests.RequestException, ValueError, OSError) as exc:
            errors.append(f"BettingPros events {season}/{week}: {exc}")
    def one(event_id, event):
        output, proofs, issues = [], [], []
        page = 1
        while True:
            payload, meta, path = _fetch("offers", {"sport": "NFL", "market_id": MARKET, "event_id": event_id,
                                       "location": "NY", "live": "false", "page": page}, directory, f"offers-{event_id}-{page}")
            q, p, errs = parse_bettingpros_offers(payload, event, meta["observed_at"], proof_path=path,
                                                proof_sha256=meta["sha256"], source_url=meta["url"])
            output.extend(q); proofs.extend(p); issues.extend(errs)
            pages = (payload.get("_pagination") or {}).get("total_pages", 1)
            if not isinstance(pages, int) or pages > 20:
                raise ValueError("unexpected offer pagination")
            if page >= pages:
                return output, proofs, issues
            page += 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, event_id, event): event_id for event_id, event in matches.items()}
        for future in as_completed(futures):
            try:
                q, p, errs = future.result()
                quotes.extend(q); evidence.extend(p); errors.extend(errs)
            except (requests.RequestException, ValueError, OSError, TypeError) as exc:
                errors.append(f"BettingPros offers {futures[future]}: {exc}")
    # Conflicts can straddle API pages; reject across the complete capture too.
    counts = {}
    for q in quotes:
        key = (q["game_id"], q["provider_player_id"], q["book"])
        counts[key] = counts.get(key, 0) + 1
    ambiguous = {key for key, count in counts.items() if count > 1}
    if ambiguous:
        errors.append(f"BettingPros: rejected {len(ambiguous)} repeated main markets across response pages")
        quotes = [q for q in quotes if (q["game_id"], q["provider_player_id"], q["book"]) not in ambiguous]
        evidence = [q for q in evidence if (q["game_id"], q["provider_player_id"], q["book"]) not in ambiguous]
    return quotes, evidence, sorted(set(errors))
