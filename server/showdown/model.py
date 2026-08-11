"""What we know about the opponent, accumulated across legs and attempts.

`recent_hands` only carries the last 20 completed hands and resets at the
start of every leg, so this keeps its own running totals and folds each
finished hand in once. The current hand is deliberately *not* accumulated: it
is still in progress and would be counted twice when it later appears in
`recent_hands`.

Two things make the keying non-obvious:

  * A leg is a fresh match with a fresh `match_id`, but phase 2 states that
    the opponent/rule pairing is identical on every retry. So stats are keyed
    by leg number and table rule, not by `match_id` - that is what makes a
    read survive into the next attempt, which the phase explicitly invites.
  * Hand numbers restart at 1 each leg, so deduplicating on `hand_number`
    alone would silently discard the first hands of every later attempt.
    The dedupe key is (match_id, hand_number).

Two numbers do the exploiting:

  * `fold_to_bet` - how often they give up when bet at. It is the entire
    justification for bluffing, and its breakeven point is pure arithmetic.
  * `bet_quantile` - how wide their betting range is, which is what turns raw
    equity into equity against the hand they are actually representing.

Both are prior-weighted, so an early read cannot swing us on three hands of
evidence, and both degrade to neutral when we have seen nothing.
"""

import json
import os
from dataclasses import dataclass, field

# Roughly "pretend we have already seen this many neutral observations".
# Eight is about one leg of hands - enough that a couple of early folds do
# not read as a bot that folds everything.
PRIOR_WEIGHT = 8.0
PRIOR_AGGRESSION = 0.30
PRIOR_FOLD_TO_BET = 0.40

# Below this many showdowns the number they show when betting is noise.
MIN_SHOWDOWNS_FOR_RANGE = 6

AGGRESSIVE_ACTIONS = frozenset({"bet", "raise"})
# A fold or a call can only happen facing a bet, and `raise` (as opposed to
# `bet`) only exists when there is something to raise over.
FACING_BET_ACTIONS = frozenset({"fold", "call", "raise"})

# Keep the persisted dedupe set from growing without bound across attempts.
MAX_REMEMBERED_HANDS = 4000


@dataclass
class OpponentStats:
    """Running totals for one opponent under one table rule."""

    seen_hands: set = field(default_factory=set)
    actions: int = 0
    aggressive_actions: int = 0
    times_facing_bet: int = 0
    folds: int = 0
    showdowns_when_aggressive: list = field(default_factory=list)
    showdowns_when_passive: list = field(default_factory=list)

    @property
    def aggression_freq(self) -> float:
        """Share of their decisions that were a bet or a raise."""
        return (self.aggressive_actions + PRIOR_AGGRESSION * PRIOR_WEIGHT) / (
            self.actions + PRIOR_WEIGHT
        )

    @property
    def fold_to_bet(self) -> float:
        """Share of the time they fold when someone bets at them."""
        return (self.folds + PRIOR_FOLD_TO_BET * PRIOR_WEIGHT) / (
            self.times_facing_bet + PRIOR_WEIGHT
        )

    @property
    def bet_quantile(self) -> float:
        """How wide their betting range looks, as a fraction of all hands.

        How often they bet is the sturdier signal, but the numbers they show
        down after betting are the more direct one: a top-k range averages
        (27 - k) / 2 in *rank order*, so an average shown strength inverts
        back to a width. A bot showing its best hands is betting a narrow
        range; one showing middling hands is betting everything.
        """
        from_frequency = self.aggression_freq

        shown = self.showdowns_when_aggressive
        if len(shown) < MIN_SHOWDOWNS_FOR_RANGE:
            return _clamp(from_frequency, 0.15, 1.0)

        mean_shown = sum(shown) / len(shown)
        implied_width = _clamp((27.0 - 2.0 * mean_shown) / 13.0, 0.15, 1.0)

        # Let the showdown read take over gradually as it earns the sample.
        trust = min(1.0, len(shown) / 20.0)
        return _clamp(trust * implied_width + (1.0 - trust) * from_frequency, 0.15, 1.0)

    @property
    def hands_recorded(self) -> int:
        return len(self.seen_hands)

    def observe_hand(self, hand: dict, opponent_seat: int, scope: str, rule: str) -> None:
        """Fold one completed hand into the totals, at most once.

        `scope` is the match id: hand numbers restart every leg, so the pair
        is what makes the dedupe correct across attempts.
        """
        number = hand.get("hand_number")
        if number is None:
            return
        key = f"{scope}#{number}"
        if key in self.seen_hands:
            return
        self.seen_hands.add(key)

        was_aggressive = False
        for entry in hand.get("actions") or []:
            if entry.get("seat") != opponent_seat:
                continue
            action = entry.get("action")
            self.actions += 1
            if action in AGGRESSIVE_ACTIONS:
                self.aggressive_actions += 1
                was_aggressive = True
            if action in FACING_BET_ACTIONS:
                self.times_facing_bet += 1
                if action == "fold":
                    self.folds += 1

        shown = _shown_number(hand.get("shown_numbers"), opponent_seat)
        if shown is not None:
            # Record where the number sat in *rank order under this rule*, not
            # the number itself. A 2 shown under low_ball is a monster; the
            # same 2 under standard is nothing, and averaging them raw would
            # read a tight opponent as a loose one.
            bucket = (
                self.showdowns_when_aggressive if was_aggressive else self.showdowns_when_passive
            )
            bucket.append(_strength_rank(shown, hand.get("community_number"), rule))

    def observe_recent(self, recent_hands, opponent_seat: int, scope: str, rule: str) -> None:
        for hand in recent_hands or []:
            if isinstance(hand, dict):
                self.observe_hand(hand, opponent_seat, scope, rule)

    # --- persistence ------------------------------------------------------

    def as_json(self) -> dict:
        remembered = list(self.seen_hands)[-MAX_REMEMBERED_HANDS:]
        return {
            "seen_hands": remembered,
            "actions": self.actions,
            "aggressive_actions": self.aggressive_actions,
            "times_facing_bet": self.times_facing_bet,
            "folds": self.folds,
            "showdowns_when_aggressive": self.showdowns_when_aggressive[-200:],
            "showdowns_when_passive": self.showdowns_when_passive[-200:],
        }

    @classmethod
    def from_json(cls, blob: dict) -> "OpponentStats":
        return cls(
            seen_hands=set(blob.get("seen_hands") or []),
            actions=int(blob.get("actions") or 0),
            aggressive_actions=int(blob.get("aggressive_actions") or 0),
            times_facing_bet=int(blob.get("times_facing_bet") or 0),
            folds=int(blob.get("folds") or 0),
            showdowns_when_aggressive=list(blob.get("showdowns_when_aggressive") or []),
            showdowns_when_passive=list(blob.get("showdowns_when_passive") or []),
        )


def _strength_rank(number: int, community, rule: str) -> float:
    """Where a shown number sits on a 1-13 strength scale under this rule.

    Returned on the same scale as a raw number so `bet_quantile`'s inversion
    keeps working, but ordered by actual strength rather than by face value.
    """
    from server.showdown import equity as eq

    rule = eq.normalize_rule(rule)
    order = eq.strength_order(community if isinstance(community, int) else None, rule)
    position = order.index(number) if number in order else len(order) // 2
    return float(len(order) - position)


def _shown_number(shown_numbers, seat: int):
    """Read one seat out of `shown_numbers`, whose keys arrive as strings."""
    if not isinstance(shown_numbers, dict):
        return None
    for key in (seat, str(seat)):
        if key in shown_numbers:
            value = shown_numbers[key]
            return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Memory:
    """Opponent stats, keyed so a read survives the things that reset.

    Phase 2 fixes the opponent/rule pairing across retries, so keying on leg
    and rule lets an attempt start with what the last one learned. Persisting
    to disk is best-effort: the file is a cache, never a dependency, and any
    IO failure silently degrades to memory-only.
    """

    def __init__(self, capacity: int = 32, path: str | None = None):
        self.capacity = capacity
        self.path = path if path is not None else os.environ.get("SHOWDOWN_MEMORY_PATH")
        self._entries: dict[str, OpponentStats] = {}
        self._load()

    @staticmethod
    def key_for(request: dict) -> str:
        leg = request.get("leg_number")
        rule = request.get("table_rule") or "standard"
        if isinstance(leg, int) and not isinstance(leg, bool):
            # Stable across attempts, which is exactly what phase 2 promises.
            return f"leg{leg}:{rule}"
        return f"match:{request.get('match_id', 'unknown')}"

    def stats_for(self, request: dict) -> OpponentStats:
        key = self.key_for(request)
        stats = self._entries.get(key)
        if stats is None:
            if len(self._entries) >= self.capacity:
                self._entries.pop(next(iter(self._entries)))
            stats = OpponentStats()
            self._entries[key] = stats
        return stats

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as handle:
                blob = json.load(handle)
            self._entries = {
                key: OpponentStats.from_json(value) for key, value in blob.items()
            }
        except Exception as error:  # noqa: BLE001 - a cache must never be fatal
            print(f"[showdown] ignoring unreadable memory: {error}", flush=True)
            self._entries = {}

    def save(self) -> None:
        if not self.path:
            return
        try:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w") as handle:
                json.dump({k: v.as_json() for k, v in self._entries.items()}, handle)
            os.replace(tmp, self.path)
        except Exception as error:  # noqa: BLE001
            print(f"[showdown] could not persist memory: {error}", flush=True)
