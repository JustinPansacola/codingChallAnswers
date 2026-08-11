"""Stage 3 tests. Plain asserts, no test-runner dependency:

    python -m tests.test_stage3          # pure functions only
    python -m tests.test_stage3 --live   # also hits the grader host

The pure functions carry the whole answer, so they are tested against
fixtures captured from the real endpoints rather than against the network.
"""

import json
import sys

from server.stages import stage3 as s3

QUESTION = (
    "Find the earliest 60-minute window on 2026-09-08 between 08:00 and 18:00 "
    "when you and ada, bram are all free, for lunch. Times are HH:MM, 24-hour."
)

# One day of the real inbox: six accepted, four declined. Captured from
# GET /emails, trimmed to the fields the parser reads.
INBOX_DAY = [
    ("2026-09-08 09:00-09:15", "ACCEPTED"),
    ("2026-09-08 09:30-10:30", "DECLINED"),
    ("2026-09-08 10:30-11:45", "ACCEPTED"),
    ("2026-09-08 12:00-13:00", "DECLINED"),
    ("2026-09-08 13:00-14:15", "ACCEPTED"),
    ("2026-09-08 14:30-15:30", "DECLINED"),
    ("2026-09-08 15:30-16:45", "ACCEPTED"),
    ("2026-09-08 17:00-18:00", "DECLINED"),
    ("2026-09-08 18:00-18:30", "ACCEPTED"),
    ("2026-09-08 19:00-19:30", "ACCEPTED"),
    ("2026-09-09 08:00-17:00", "ACCEPTED"),  # a different day, must be ignored
]


def _emails(rows=INBOX_DAY):
    return [
        {
            "id": f"e{i:03d}",
            "body": (
                "From: Marek Sould <m.sould@kesterline.example>\n"
                "Sent: 2026-09-01 08:12\n"
                "Subject: Invitation — Quarterly budget review\n"
                f"Response: {response}\n"
                f"When: {when}\n\n"
                "I've put it in my calendar.\n"
            ),
        }
        for i, (when, response) in enumerate(rows, 1)
    ]


def _hhmm(intervals):
    return [(s3._to_hhmm(a), s3._to_hhmm(b)) for a, b in intervals]


def _slot(busy, window=("08:00", "18:00"), duration=60):
    found = s3._earliest_slot(
        s3._merge([(s3._to_minutes(a), s3._to_minutes(b)) for a, b in busy]),
        s3._to_minutes(window[0]),
        s3._to_minutes(window[1]),
        duration,
    )
    return None if found is None else (s3._to_hhmm(found), s3._to_hhmm(found + duration))


# --- times ----------------------------------------------------------------


def test_times_round_trip_zero_padded():
    assert s3._to_minutes("09:00") == 540
    assert s3._to_minutes("14:45") == 885
    assert s3._to_hhmm(540) == "09:00", "9:00 would be rejected by the grader"
    assert s3._to_hhmm(885) == "14:45"
    assert s3._to_hhmm(1080) == "18:00"
    # Tolerant on input, strict on output.
    assert s3._to_hhmm(s3._to_minutes("9:00")) == "09:00"

    for bad in ("9am", "", "25:00", "09:60", "0900"):
        try:
            s3._to_minutes(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse as a time")


def test_snap_up_leaves_grid_points_alone():
    assert s3._snap_up(s3._to_minutes("08:00"), 30) == s3._to_minutes("08:00")
    assert s3._snap_up(s3._to_minutes("08:01"), 30) == s3._to_minutes("08:30")
    assert s3._snap_up(s3._to_minutes("14:45"), 30) == s3._to_minutes("15:00")


# --- question parsing -----------------------------------------------------


def test_parses_the_canonical_question():
    assert s3._parse_question(QUESTION) == {
        "day": "2026-09-08",
        "duration_minutes": 60,
        "between_start": "08:00",
        "between_end": "18:00",
        "people": ["ada", "bram"],
    }


def test_parsing_handles_the_other_shapes_a_question_can_take():
    # "you and ada and bram", a 90-minute meeting, a tighter window.
    parsed = s3._parse_question(
        "Find the earliest 90 minute window on 2026-09-10 between 9:30 and 17:00 "
        "when you and ada and bram are all free."
    )
    assert parsed["duration_minutes"] == 90
    assert parsed["people"] == ["ada", "bram"]
    assert (parsed["between_start"], parsed["between_end"]) == ("9:30", "17:00")

    # Hours rather than minutes, and a single guest.
    parsed = s3._parse_question("earliest 1-hour window on 2026-09-08 when you and iris are free")
    assert parsed["duration_minutes"] == 60
    assert parsed["people"] == ["iris"]

    # Nothing stated is nothing claimed, so real arguments always win.
    assert s3._parse_question("") == {}
    assert "day" not in s3._parse_question("when you and ada are all free")


def test_self_is_never_looked_up_as_a_person():
    assert s3._clean_people(["you", "ada", "bram"]) == ["ada", "bram"]
    assert s3._clean_people("ada, bram") == ["ada", "bram"], "a bare string must work too"
    assert s3._clean_people("You and Ada") == ["ada"]
    assert s3._clean_people(["ada", "ada"]) == ["ada"]
    assert s3._clean_people(None) == []


# --- inbox ----------------------------------------------------------------


def test_declined_invitations_are_not_commitments():
    busy = _hhmm(s3._accepted_intervals(_emails(), "2026-09-08"))
    assert busy == [
        ("09:00", "09:15"),
        ("10:30", "11:45"),
        ("13:00", "14:15"),
        ("15:30", "16:45"),
        ("18:00", "18:30"),
        ("19:00", "19:30"),
    ]
    assert ("09:30", "10:30") not in busy, "a DECLINED invitation is free time"
    assert ("08:00", "17:00") not in busy, "another day's invitation must not leak in"


# --- merging and scanning -------------------------------------------------


def test_merge_unions_overlapping_and_adjacent_blocks():
    assert _hhmm(s3._merge([])) == []
    assert _hhmm(
        s3._merge(
            [
                (s3._to_minutes("13:00"), s3._to_minutes("14:15")),
                (s3._to_minutes("09:00"), s3._to_minutes("09:15")),
                (s3._to_minutes("13:30"), s3._to_minutes("15:00")),  # overlaps
                (s3._to_minutes("15:00"), s3._to_minutes("15:30")),  # adjacent
            ]
        )
    ) == [("09:00", "09:15"), ("13:00", "15:30")]


def test_touching_a_commitment_is_not_a_clash():
    # Free until 09:00 exactly: an hour ending as the commitment starts fits.
    assert _slot([("09:00", "10:00")]) == ("08:00", "09:00")
    # And one starting the moment a commitment ends fits too.
    assert _slot([("08:00", "09:00")]) == ("09:00", "10:00")


def test_an_untidy_end_time_is_not_rounded_down():
    # The trap: busy until 14:45, so 15:00 is next - not 14:45, not 14:30.
    assert _slot([("08:00", "14:45")]) == ("15:00", "16:00")
    assert _slot([("08:00", "14:15")]) == ("14:30", "15:30")


def test_the_window_bounds_are_respected():
    # A 60-minute meeting must finish by 18:00, so 17:00 is the last start.
    assert _slot([("08:00", "17:00")]) == ("17:00", "18:00")
    assert _slot([("08:00", "17:30")]) is None, "17:30-18:30 would overrun the window"
    # The scan starts at the window, not at midnight.
    assert _slot([], window=("13:00", "18:00")) == ("13:00", "14:00")
    # An off-grid bound snaps up rather than answering 08:15.
    assert _slot([], window=("08:15", "18:00")) == ("08:30", "09:30")


def test_the_example_question_against_the_real_data():
    """Inbox for 2026-09-08 plus ada 16:00-16:30 and bram 13:30-14:00."""
    busy = _hhmm(s3._accepted_intervals(_emails(), "2026-09-08"))
    busy += [("16:00", "16:30"), ("13:30", "14:00")]
    assert _slot(busy) == ("08:00", "09:00")

    # And with the cheat sheet's illustrative ada, who is busy until 11:30:
    # 11:45 is when the inbox frees up, so the grid gives 12:00.
    illustrative = _hhmm(s3._accepted_intervals(_emails(), "2026-09-08"))
    illustrative += [("08:00", "11:30"), ("16:30", "17:00"), ("13:30", "14:00")]
    assert _slot(illustrative) == ("12:00", "13:00")


# --- the tool itself ------------------------------------------------------


def test_tool_rejects_a_question_with_no_day():
    try:
        s3.find_earliest_free_window(question="Find the earliest 60-minute window")
    except ValueError as e:
        assert "day" in str(e)
        return
    raise AssertionError("a question with no day should raise, not guess a day")


def test_tool_rejects_an_empty_window():
    try:
        s3.find_earliest_free_window(day="2026-09-08", between_start="18:00", between_end="08:00")
    except ValueError as e:
        assert "empty" in str(e)
        return
    raise AssertionError("an inverted window should raise")


def test_tool_answers_the_example_question_live():
    """Hits the real grader host. Only run under --live."""
    answer = json.loads(s3.find_earliest_free_window(question=QUESTION))
    assert answer["start"] == "08:00" and answer["end"] == "09:00", answer
    assert answer["day"] == "2026-09-08"
    assert answer["people"] == ["you", "ada", "bram"]
    assert answer["submit"] == '_submit_slot(start="08:00", end="09:00")'

    # Structured arguments must reach the same answer as the sentence.
    assert json.loads(
        s3.find_earliest_free_window(
            day="2026-09-08",
            people=["ada", "bram"],
            duration_minutes=60,
            between_start="08:00",
            between_end="18:00",
        )
    ) == answer

    # A person with no published schedule must raise rather than be skipped.
    try:
        s3.find_earliest_free_window(day="2026-09-08", people=["cleo"])
    except ValueError as e:
        assert "cleo" in str(e)
    else:
        raise AssertionError("an unknown person should raise, not be treated as free")


def main() -> int:
    live = "--live" in sys.argv
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        if test.__name__.endswith("_live") and not live:
            print(f"SKIP {test.__name__} (pass --live to run)")
            continue
        try:
            test()
        except Exception as e:
            failures += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
