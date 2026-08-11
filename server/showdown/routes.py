"""HTTP surface for SHOWDOWN: POST /move and GET /health.

The coordinator never retries a /move call, and five bad replies in a row
forfeit the match, so the contract this module keeps is that *something
legal always comes back*. Any exception at all - a protocol change, a bug in
the strategy, malformed JSON - falls through to a legal action rather than a
500. That fallback matters more than the strategy does: a strategy bug costs
chips, an exception costs the match.
"""

import json

from starlette.responses import JSONResponse

from server.showdown.model import MatchMemory
from server.showdown.policy import decide

MEMORY = MatchMemory()


def _legal(request: dict) -> tuple:
    legal = request.get("legal_actions")
    return tuple(legal) if isinstance(legal, list) else ()


def safe_action(request: dict) -> dict:
    """The cheapest legal action available - what we fall back to."""
    legal = _legal(request)
    for action in ("check", "fold", "call"):
        if action in legal:
            return {"action": action}
    return {"action": legal[0]} if legal else {"action": "check"}


def sanitize(reply: dict, request: dict) -> dict:
    """Force a reply into something the coordinator will accept.

    `legal_actions` is authoritative, and an out-of-range amount is not
    clamped for us - it counts as an illegal move - so both are enforced here
    rather than trusted from the strategy.
    """
    legal = _legal(request)
    action = reply.get("action")
    if action not in legal:
        return safe_action(request)

    if action not in ("bet", "raise"):
        return {"action": action}

    low, high = request.get("min_raise_to"), request.get("max_raise_to")
    if not isinstance(low, int) or not isinstance(high, int) or high < low:
        return safe_action(request)

    amount = reply.get("amount")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return safe_action(request)

    return {"action": action, "amount": int(max(low, min(high, round(amount))))}


def play(request: dict) -> dict:
    """Full decision for one turn, including the never-fail guarantee."""
    try:
        seat = request.get("your_seat")
        stats = MEMORY.stats_for(request.get("match_id", "unknown"))
        opponent_seat = next(
            (
                p.get("seat")
                for p in (request.get("players") or [])
                if isinstance(p, dict) and p.get("seat") != seat
            ),
            None,
        )
        if opponent_seat is not None:
            stats.observe_recent(request.get("recent_hands"), opponent_seat)
        return sanitize(decide(request, stats), request)
    except Exception as error:  # noqa: BLE001 - a raise here forfeits hands
        print(f"[showdown] falling back after {type(error).__name__}: {error}", flush=True)
        try:
            return safe_action(request)
        except Exception:
            return {"action": "check"}


async def move(http_request):
    # An OPTIONS probe against /move is the warm-up fallback when /health is
    # not found, so it must answer 200 without being treated as a turn.
    if http_request.method == "OPTIONS":
        return JSONResponse({"ok": True})
    try:
        body = await http_request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    return JSONResponse(play(body))


async def health(http_request):
    return JSONResponse({"ok": True})
