"""What we know about the opponent, accumulated across a match.

`recent_hands` only carries the last 20 completed hands, so this keeps its
own running totals keyed by `match_id` and folds in each finished hand once,
deduplicated by `hand_number`. The current hand is deliberately *not*
accumulated: it is still in progress and would be counted a second time when
it later shows up in `recent_hands`.

Two numbers do the exploiting:

  * `fold_to_bet` - how often they give up when bet at. It is the entire
    justification for bluffing, and its breakeven point is pure arithmetic:
    a half-pot bluff needs them to fold a third of the time.
  * `bet_quantile` - how wide their betting range is, which is what turns raw
    equity into equity against the hand they are actually representing.

Both are prior-weighted, so an early read cannot swing us on three hands of
evidence, and both degrade to neutral when we have seen nothing.
"""

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


@dataclass
class OpponentStats:
    """Running totals for one opponent in one match."""

    hands_folded_in: set = field(default_factory=set)
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
        (27 - k) / 2, so an average shown number inverts straight back to a
        width. A bot showing 11s is betting its top few hands; a bot showing
        7s is betting everything.
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
        return len(self.hands_folded_in)

    def observe_hand(self, hand: dict, opponent_seat: int) -> None:
        """Fold one completed hand into the totals, at most once."""
        number = hand.get("hand_number")
        if number is None or number in self.hands_folded_in:
            return
        self.hands_folded_in.add(number)

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
            bucket = self.showdowns_when_aggressive if was_aggressive else self.showdowns_when_passive
            bucket.append(shown)

    def observe_recent(self, recent_hands, opponent_seat: int) -> None:
        for hand in recent_hands or []:
            if isinstance(hand, dict):
                self.observe_hand(hand, opponent_seat)


def _shown_number(shown_numbers, seat: int):
    """Read one seat out of `shown_numbers`, whose keys arrive as strings."""
    if not isinstance(shown_numbers, dict):
        return None
    for key in (seat, str(seat)):
        if key in shown_numbers:
            value = shown_numbers[key]
            return value if isinstance(value, int) else None
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MatchMemory:
    """Per-match opponent stats, bounded so a long-lived process cannot grow
    without limit. Matches are short and finish, so evicting the oldest is
    enough - there is nothing to persist once a match is done.
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._matches: dict[str, OpponentStats] = {}

    def stats_for(self, match_id: str) -> OpponentStats:
        key = str(match_id)
        stats = self._matches.get(key)
        if stats is None:
            if len(self._matches) >= self.capacity:
                self._matches.pop(next(iter(self._matches)))
            stats = OpponentStats()
            self._matches[key] = stats
        return stats
