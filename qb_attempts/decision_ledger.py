"""Append-only early-entry paper signal cohort, never an execution ledger.

Verified recommendations are an explicit input from the source-verification
layer. Unknown/questionable roles are warnings, never a kickoff-time gate.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from statistics import median

from .settlement import settle_entry, summarize_results, timestamp
from .odds_sources import canonical_book

VERSION = "verified_paper_entry_v1"


def empty_ledger():
    return {"version": VERSION, "cohort": "verified_published_paper_signals",
            "actual_wagers": [], "entries": [], "events": [], "processed_boards": [],
            "note": "Flat one-unit paper signals at first verified published prices; no wagers executed."}


def _key(model, row):
    return (str(model), str(row.get("game_id") or ""), str(row.get("player_id") or row.get("player") or ""))


def _event(ledger, model, row, at, status, reason):
    key = _key(model, row)
    event = {"model": key[0], "game_id": key[1], "player_id": key[2],
             "at": at, "status": status, "reason": reason, "version": VERSION}
    # Exact reruns are idempotent; later observations remain auditable.
    if event not in ledger["events"]:
        ledger["events"].append(event)


def _withdrawal(row):
    context = row.get("role_evidence") or {}
    return (context.get("reliable") is True and context.get("status") in {"out", "inactive", "benched"})


def update_ledger(ledger, boards, now, *, max_open_per_model=5):
    """Return updated ledger. Only explicit verified_recommendations can enter.

    Capacity counts unsettled entries across daily cards, per model. A game can
    create only one entry per model even after settlement. Withdrawals affect
    current suggestions, never the historical price, exposure or eventual grade.
    """
    current = timestamp(now)
    if current is None or not isinstance(max_open_per_model, int) or max_open_per_model < 1:
        raise ValueError("timezone-aware now and positive max_open_per_model required")
    result = deepcopy(ledger) if ledger is not None else empty_ledger()
    if result.get("version") != VERSION:
        raise ValueError("unsupported ledger version")
    previous = timestamp(result.get("updated_at"))
    if previous is not None and current < previous:
        raise ValueError("ledger updates cannot travel backwards in time")
    for model, board in boards.items():
        generated = timestamp(board.get("generated_at"))
        if generated is None or generated > current or (previous is not None and generated < previous):
            continue
        at = generated.isoformat()
        observation_id = f"{model}:{at}"
        if observation_id in result.setdefault("processed_boards", []):
            continue
        result["processed_boards"].append(observation_id)
        for row in board.get("recommendations", []) + board.get("watchlist", []):
            _event(result, model, row, at, "withdrawn" if _withdrawal(row) else "observed",
                   "explicit_reliable_unavailability" if _withdrawal(row) else "research_observation")
        if board.get("status") not in {"ok", "no_odds"}:
            continue
        for row in board.get("verified_recommendations", []):
            if _withdrawal(row):
                _event(result, model, row, at, "withdrawn", "explicit_reliable_unavailability")
                continue
            kickoff = timestamp(row.get("kickoff"))
            key = _key(model, row)
            line, odds = row.get("line"), row.get("odds")
            quote_at = timestamp(row.get("observed_at"))
            source_at = timestamp(row.get("source_updated_at"))
            if (not all(key) or kickoff is None or generated >= kickoff or current >= kickoff
                    or quote_at is None or quote_at > generated or (source_at is not None and source_at > quote_at)
                    or isinstance(line, bool) or not isinstance(line, (int, float)) or not math.isfinite(line)
                    or line < 0 or line * 2 != int(line * 2)
                    or isinstance(odds, bool) or not isinstance(odds, (int, float)) or not math.isfinite(odds)
                    or abs(odds) < 100 or str(row.get("side", "")).lower() not in {"over", "under"}):
                _event(result, model, row, at, "observed", "invalid_or_noncausal_entry")
                continue
            _event(result, model, row, at, "eligible", "verified_price_early_entry_allowed")
            entries = [e for e in result["entries"] if e["model"] == model]
            if any(e["game_id"] == row["game_id"] for e in entries):
                _event(result, model, row, at, "observed", "existing_game_exposure")
                continue
            if sum(e.get("settlement", {}).get("result", "pending") == "pending" for e in entries) >= max_open_per_model:
                _event(result, model, row, at, "observed", "open_exposure_cap")
                continue
            identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
            result["entries"].append({**deepcopy(row), "entry_id": identity, "model": model,
                "model_version": board.get("model_version", "unknown"), "entry_at": at,
                "cohort": VERSION, "actual_wager": False, "stake_units": 1.,
                "settlement": {"result": "pending", "actual": None, "profit_1u": None},
                "settlement_history": [],
                "warnings": list(row.get("warnings") or []) + (["role_not_confirmed_at_entry"]
                    if not (row.get("role_evidence") or {}).get("confirmed") else [])})
            _event(result, model, row, at, "paper_entry", "first_verified_published_quote")
    result["updated_at"] = current.isoformat()
    return result


def reconcile_ledger(ledger, evidence_by_game, now=None):
    """Add versioned grades without changing any frozen entry quote."""
    now = now or datetime.now(timezone.utc).isoformat()
    current = timestamp(now)
    if current is None:
        raise ValueError("timezone-aware now required")
    result = deepcopy(ledger)
    for entry in result["entries"]:
        if entry["game_id"] not in evidence_by_game:
            continue
        grade = settle_entry(entry, evidence_by_game[entry["game_id"]], as_of=now)
        if grade["result"] == "pending":
            continue  # A transient source failure must not erase a final grade.
        if grade != entry.get("settlement"):
            entry.setdefault("settlement_history", []).append({"at": current.isoformat(), "grade": deepcopy(grade)})
            entry["settlement"] = grade
            _event(result, entry["model"], entry, current.isoformat(), grade["result"], grade["settlement_reason"])
    result["performance"] = {model: summarize_results([{**e, **e["settlement"]} for e in result["entries"] if e["model"] == model])
                             for model in sorted({e["model"] for e in result["entries"]})}
    return result


def attach_verified_clv(ledger, closings, now):
    """Attach strict verified paired closing-price evidence to paper entries.

    Revalidate retained direct/comparison evidence as of its capture, require the
    last pregame hour, and exclude observations after a reliable withdrawal.
    This is conditional-on-no-push price CLV; no executable-profit claim.
    """
    from .quote_verification import verify_against_offers

    current = timestamp(now)
    if current is None:
        raise ValueError("timezone-aware now required")
    result = deepcopy(ledger)
    for entry in result["entries"]:
        kickoff = timestamp(entry.get("kickoff"))
        latest = {}
        entry_key = _key(entry["model"], entry)
        withdrawals = [timestamp(e["at"]) for e in result["events"]
                       if _key(e["model"], e) == entry_key and e["status"] == "withdrawn"]
        if entry.get('settlement', {}).get('result') == 'void':
            entry['verified_clv'] = {'value': None, 'books': 0, 'reason': 'void_entry'}
            continue
        for quote in closings:
            if (quote.get("game_id") != entry.get("game_id") or quote.get("player_id") != entry.get("player_id")
                    or quote.get("line") != entry.get("line") or _withdrawal(quote)):
                continue
            proof = quote.get("quote_verification") or {}
            checked = timestamp(proof.get("checked_at"))
            observed = timestamp(proof.get("observed_at"))
            if (proof.get("status") != "verified" or kickoff is None or checked is None or observed is None
                    or checked > current or not 0 <= (kickoff-checked).total_seconds() <= 3600
                    or not 0 <= (kickoff-observed).total_seconds() <= 3600
                    or any(t is not None and t <= checked for t in withdrawals)):
                continue
            sides = proof.get("sides") or {}
            # The shared verifier checks exact line/prices, availability, event,
            # market and selection IDs, jurisdiction and original source age.
            offers = list(sides.values())
            if verify_against_offers([quote], offers, checked)[0]["quote_verification"]["status"] != "verified":
                continue
            try:
                odds = [float(quote[s + "_odds"]) for s in ("over", "under")]
                if any(not math.isfinite(o) or abs(o) < 100 for o in odds):
                    continue
                decimal = [1 + (o / 100 if o > 0 else 100 / -o) for o in odds]
            except (KeyError, ValueError, TypeError):
                continue
            probability = (1 / decimal[0]) / (1 / decimal[0] + 1 / decimal[1])
            book = canonical_book(quote.get("book"))
            if book and (book not in latest or checked > latest[book][0]):
                latest[book] = (checked, probability, deepcopy(proof))
        clv = {"value": None, "books": 0, "definition": "verified_paired_probability_times_entry_decimal_minus_one_conditional_on_no_push"}
        if latest:
            p_over = median(v[1] for v in latest.values())
            probability = p_over if str(entry["side"]).lower() == "over" else 1 - p_over
            odds = entry["odds"]
            decimal = 1 + (odds / 100 if odds > 0 else 100 / -odds)
            clv.update(value=probability * decimal - 1, books=len(latest), probability=probability,
                       observed_at=max(v[0] for v in latest.values()).isoformat(),
                       evidence={book: v[2] for book, v in latest.items()})
        entry["verified_clv"] = clv
    return result
