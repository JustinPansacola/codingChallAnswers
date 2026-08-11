"""Stage 3 ("Working Life"): find the earliest meeting window that works for
everyone, reconciling other people's calendars with your own inbox.

Cheap stage, same rule as stage 1: httpx is imported inside the fetch
helpers rather than at module scope, so stages 1 and 2 never pay for it.

The whole stage is one decisive tool. The grader's agent parses a question,
calls `find_earliest_free_window`, and relays the result to the grader's own
`_submit_slot` - it does no arithmetic of its own, because the two ways to
lose this stage are both arithmetic:

  * `DECLINED` invitations are not commitments. Four of the ten invitations
    in the inbox each day are declined, and counting them as busy pushes the
    answer far too late.
  * Obstacles end at untidy times (11:45, 14:15, 16:45) but meetings start
    only on the hour or the half hour. An obstacle ending at 14:45 means the
    next candidate is 15:00 - never 14:45, and never 14:30. Snapping every
    candidate onto the :00/:30 grid makes that fall out for free instead of
    being a special case somebody has to remember.

The inbox (~14KB, 50 messages) is far over the 1,500-token response ceiling
and is never returned to the agent; only the computed window is. Nothing is
cached: the answer is three HTTP calls and some integer arithmetic, well
inside the 10s budget, and a long-lived process that cached the inbox would
keep serving it after the grader moved on.
"""

import json
import os
import re

from mcp.server.fastmcp import FastMCP

from server.transport import PUBLIC_TRANSPORT_SECURITY

mcp = FastMCP("stage3", transport_security=PUBLIC_TRANSPORT_SECURITY)

DEFAULT_API_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"

DEFAULT_DURATION_MINUTES = 60
DEFAULT_WINDOW_START = "08:00"
DEFAULT_WINDOW_END = "18:00"

# Meetings begin on the hour or the half hour, so candidate starts step
# across a 30-minute grid regardless of how untidy the obstacles are.
SLOT_GRID_MINUTES = 30

# Ways the agent might name the inbox owner in a `people` list. They are not
# people to look up - "you" is the inbox, and is always included anyway.
SELF_ALIASES = frozenset({"you", "me", "myself", "self", "i", "us", "we", "your", "yourself"})


def _api_base_url() -> str:
    return os.environ.get("STAGE3_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


# --- times ----------------------------------------------------------------


_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _to_minutes(value: str) -> int:
    """Parse "HH:MM" into minutes since midnight.

    Accepts a non-padded hour on the way in (the calendar API and inbox are
    consistent, but the agent may not be); output is always padded.
    """
    match = _TIME_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"could not read {value!r} as a HH:MM time")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if not (0 <= hours <= 24 and 0 <= minutes < 60):
        raise ValueError(f"{value!r} is not a valid time of day")
    return hours * 60 + minutes


def _to_hhmm(minutes: int) -> str:
    """Format minutes since midnight as a zero-padded 24-hour "HH:MM"."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _snap_up(minutes: int, grid: int) -> int:
    """Round up to the next point on the grid (a no-op if already on it)."""
    return -(-minutes // grid) * grid


# --- the question ---------------------------------------------------------


_DAY_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DURATION_RE = re.compile(r"\b(\d+)\s*[-\s]?\s*(?:minute|min\b|hour|hr\b)", re.I)
_HOUR_UNIT_RE = re.compile(r"\b\d+\s*[-\s]?\s*(?:hour|hr)", re.I)
_WINDOW_RE = re.compile(r"\bbetween\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})", re.I)
_PEOPLE_RE = re.compile(r"\bwhen\s+(.+?)\s+(?:are|is)\s+(?:all\s+)?free", re.I)


def _clean_people(names) -> list[str]:
    """Normalise a people argument into a list of other people's names.

    Accepts a real list or a bare "ada, bram" string, and drops any way the
    agent might have written the inbox owner into it.
    """
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    cleaned = []
    for entry in names:
        for name in re.split(r",|\band\b|&", str(entry), flags=re.I):
            name = name.strip().strip(".").lower()
            if name and name not in SELF_ALIASES and name not in cleaned:
                cleaned.append(name)
    return cleaned


def _parse_question(text: str) -> dict:
    """Pull whatever the canonical question sentence states into arguments.

    Only used to fill gaps the caller left; keys are omitted when the
    sentence does not state them, so an explicit argument always wins.
    """
    found: dict = {}
    if not text:
        return found

    day = _DAY_RE.search(text)
    if day:
        found["day"] = day.group(1)

    duration = _DURATION_RE.search(text)
    if duration:
        amount = int(duration.group(1))
        found["duration_minutes"] = amount * 60 if _HOUR_UNIT_RE.search(duration.group(0)) else amount

    window = _WINDOW_RE.search(text)
    if window:
        found["between_start"] = window.group(1)
        found["between_end"] = window.group(2)

    people = _PEOPLE_RE.search(text)
    if people:
        names = _clean_people(people.group(1))
        if names:
            found["people"] = names

    return found


# --- busy time ------------------------------------------------------------


_RESPONSE_RE = re.compile(r"^Response:\s*(\S+)", re.M)
_WHEN_RE = re.compile(
    r"^When:\s*(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", re.M
)


def _accepted_intervals(emails: list, day: str) -> list[tuple[int, int]]:
    """Your own commitments on `day`, read out of the inbox.

    Every message is an invitation you replied to, so only the ones you
    ACCEPTED actually occupy your calendar.
    """
    intervals = []
    for email in emails:
        body = email.get("body", "")
        response = _RESPONSE_RE.search(body)
        when = _WHEN_RE.search(body)
        if not response or not when:
            continue
        if response.group(1).strip().upper() != "ACCEPTED":
            continue
        if when.group(1) != day:
            continue
        intervals.append((_to_minutes(when.group(2)), _to_minutes(when.group(3))))
    return intervals


def _fetch_my_busy(day: str) -> list[tuple[int, int]]:
    import httpx

    resp = httpx.get(f"{_api_base_url()}/emails", timeout=8.0)
    resp.raise_for_status()
    return _accepted_intervals(resp.json().get("emails", []), day)


def _fetch_person_busy(person: str, day: str) -> list[tuple[int, int]]:
    import httpx

    resp = httpx.get(f"{_api_base_url()}/schedule/{person}/{day}", timeout=8.0)
    if resp.status_code == 404:
        # Answering anyway would quietly ignore this person and hand back a
        # confidently too-early slot. A raised error is attributed to the
        # grader and the question is re-attempted instead.
        raise ValueError(f"no schedule published for {person!r} on {day}")
    resp.raise_for_status()
    return [(_to_minutes(start), _to_minutes(end)) for start, end in resp.json().get("busy", [])]


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and union intervals into non-overlapping blocks."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _earliest_slot(
    busy: list[tuple[int, int]], window_start: int, window_end: int, duration: int
) -> int | None:
    """First grid-aligned start whose whole meeting is free, or None.

    Overlap is tested strictly, so touching does not count: a commitment
    ending at 09:00 blocks neither a meeting ending at 09:00 nor one
    starting there.
    """
    start = _snap_up(window_start, SLOT_GRID_MINUTES)
    while start + duration <= window_end:
        finish = start + duration
        if not any(b_start < finish and start < b_end for b_start, b_end in busy):
            return start
        start += SLOT_GRID_MINUTES
    return None


# --- tools ----------------------------------------------------------------


@mcp.tool()
def find_earliest_free_window(
    question: str = "",
    day: str = "",
    people: list[str] | str | None = None,
    duration_minutes: int | None = None,
    between_start: str = "",
    between_end: str = "",
) -> str:
    """Find the earliest window on a given day when you and the named people
    are all free, and return it ready to submit.

    This is the answer, not the ingredients: it reads your inbox and each
    person's calendar itself, discards invitations you declined, and returns
    the one correct window. Do not adjust the times it gives you - submit
    them exactly as returned, including the zero padding.

    Pass the question verbatim as `question` and everything else is optional;
    the day, people, duration and window bounds are read out of it. Any
    argument you do supply overrides what the sentence says. Times are
    zero-padded 24-hour "HH:MM" strings, and the window returned starts on
    the hour or the half hour.

    Returns a JSON object with `start` and `end`, plus the exact
    `_submit_slot` call to make.
    """
    from_question = _parse_question(question)

    day = (day or "").strip() or from_question.get("day", "")
    if not day:
        raise ValueError("no day given: pass `day` as YYYY-MM-DD, or the question verbatim")

    names = _clean_people(people) or from_question.get("people", [])

    duration = duration_minutes or from_question.get("duration_minutes") or DEFAULT_DURATION_MINUTES
    duration = int(duration)
    if duration <= 0:
        raise ValueError(f"meeting length must be positive, got {duration}")

    start_bound = _to_minutes(
        (between_start or "").strip() or from_question.get("between_start", DEFAULT_WINDOW_START)
    )
    end_bound = _to_minutes(
        (between_end or "").strip() or from_question.get("between_end", DEFAULT_WINDOW_END)
    )
    if start_bound >= end_bound:
        raise ValueError(f"window {_to_hhmm(start_bound)}-{_to_hhmm(end_bound)} is empty")

    busy = _fetch_my_busy(day)
    for name in names:
        busy.extend(_fetch_person_busy(name, day))

    slot = _earliest_slot(_merge(busy), start_bound, end_bound, duration)
    if slot is None:
        raise ValueError(
            f"no {duration}-minute window on {day} between {_to_hhmm(start_bound)} and "
            f"{_to_hhmm(end_bound)} is free for all of: you, {', '.join(names) or '(nobody else)'}"
        )

    start_time, end_time = _to_hhmm(slot), _to_hhmm(slot + duration)
    return json.dumps(
        {
            "start": start_time,
            "end": end_time,
            "day": day,
            "duration_minutes": duration,
            "people": ["you", *names],
            "submit": f'_submit_slot(start="{start_time}", end="{end_time}")',
        }
    )
