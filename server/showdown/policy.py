"""The strategy: a pure function from (request, opponent stats) to an action.

Nothing here touches the network or mutates anything, which is what lets the
whole thing be played out against a simulated table in tests/showdown_sim.py
rather than only reasoned about.

The shape of it:

  * Score the hand as equity against the range the opponent is actually
    representing, not against a random number (see equity.py).
  * Bet when that beats a value threshold, call when it beats the pot odds,
    fold otherwise, and bluff only when their measured fold rate says a bluff
    shows a profit.
  * Never risk the match on a hand that can lose. A pair cannot lose, so it
    is exempt from the exposure cap; everything else is capped.
  * Steer variance by the scoreboard. Clearing needs +10, and busting costs a
    flat -200, so the goal is not the most chips - it is the highest chance of
    finishing above the line. Being ahead late is worth protecting; being
    behind late is worth gambling on, because a bigger loss scores the same
    as a small one.
"""

import hashlib
from dataclasses import dataclass

from server.showdown import equity as eq

# --- tuning ---------------------------------------------------------------
# Equity needed to put chips in voluntarily. Post-reveal thresholds sit
# higher because there is no longer a community number to come that could
# rescue a bad number.
VALUE_BET_PRE = 0.720
VALUE_BET_POST = 0.727
VALUE_RAISE_PRE = 0.880
VALUE_RAISE_POST = 0.673

# Cushion over raw pot odds before calling, to cover the chips a worse hand
# will lose on later streets. Acting last is worth roughly this much back.
CALL_MARGIN = 0.051
POSITION_CREDIT = 0.049

# Bet sizes as a fraction of the pot.
BLUFF_SIZE = 0.480
BLUFF_RAISE_SIZE = 0.997
VALUE_SIZE = 0.540
PAIR_SIZE_VS_STATION = 1.870
PAIR_SIZE_VS_NORMAL = 0.754
PAIR_SIZE_VS_FOLDER = 0.915

# A bluff risks some chips to win the pot, so it breaks even exactly when
# they fold risk / (risk + pot) of the time. That is only the immediate
# arithmetic though - it says nothing about what the hand costs once they
# call, which is where a naive bluffing bot bleeds out. Three guards cover
# the gap: a fat cushion over breakeven, a floor on the equity we bluff with
# so a called bluff still wins sometimes, and giving up once called.
BLUFF_EDGE = 0.200
BLUFF_FREQUENCY = 0.959
BLUFF_MIN_EQUITY = 0.494

# Fraction of the stack we began the hand with that we will voluntarily
# commit across the whole hand without a pair. Counted per hand, not per
# action: three raises each capped at a third of the stack is not a third of
# the stack. A pair cannot lose and is exempt.
RISK_CAP = 0.525
# Equity that justifies calling past the cap anyway.
CAP_OVERRIDE_EQUITY = 0.668

# Short-stacked play: below this many chips the cap stops meaning anything
# and marginal spots just bleed us out.
SHORT_STACK = 20

# The scoreboard only steers the last quarter of the match. Earlier than
# that, folding to protect a lead simply donates it back through the blinds.
# Expressed as fractions because a phase-2 leg is 40 hands, not 100.
ENDGAME_FRACTION = 0.25
DESPERATE_FRACTION = 0.05

# What counts as clearing. Phase 1 needs +10 over one 100-hand match; a
# phase-2 leg needs +40, or +60 on the toughest, over 40 hands.
TARGET_DELTA_SINGLE = 10
TARGET_DELTA_LEG = 40
# Only protect a lead once it is clear of the *hardest* threshold we might be
# facing - we are never told which leg is the +60 one, so treating +40 as
# done would leave that leg's points on the table.
PROTECT_MARGIN = 20


@dataclass(frozen=True)
class Mode:
    name: str
    value_shift: float
    bluff_multiplier: float
    risk_cap: float


GRIND = Mode("grind", 0.0, 1.0, RISK_CAP)
PROTECT = Mode("protect", 0.05, 0.4, 0.22)
PUSH = Mode("push", -0.06, 1.5, 0.5)
DESPERATE = Mode("desperate", -0.12, 2.0, 0.9)


@dataclass
class Situation:
    number: int
    community: int | None
    rule: str
    pot: int
    to_call: int
    stack: int
    bet_this_round: int
    min_raise_to: int | None
    max_raise_to: int | None
    legal: tuple
    is_post_reveal: bool
    in_position: bool
    opponent_was_aggressive: bool
    opponent_has_contested: bool
    hands_left: int
    chip_delta: int
    stack_at_hand_start: int
    target_delta: int
    total_hands: int

    @property
    def locked(self) -> bool:
        """This holding cannot lose - it wins or ties against everything.

        A pair under `standard`, the lowest non-pair under `low_ball`, a 7
        under `wild_seven`. Chips in with it are free, so it is exempt from
        the exposure cap.
        """
        return eq.is_locked(self.number, self.community, self.rule)

    @property
    def holds_pair(self) -> bool:
        return eq.holds_pair(self.number, self.community, self.rule)

    @property
    def bounty(self) -> int:
        """Chips the house pays for winning a showdown holding a pair.

        Only under `pair_bounty`, and only at showdown - a hand won by a fold
        pays nothing, so this raises the value of *calling*, never of betting
        big enough to make them fold.
        """
        if self.rule == eq.PAIR_BOUNTY and self.holds_pair:
            return eq.PAIR_BOUNTY_CHIPS
        return 0

    @property
    def current_bet(self) -> int:
        return self.bet_this_round + self.to_call

    @property
    def committed_this_hand(self) -> int:
        """Chips already in from this hand, blinds included.

        `chip_delta` is frozen at the start of the hand, which is exactly what
        makes this recoverable: stack-at-hand-start minus stack-now.
        """
        return max(0, self.stack_at_hand_start - self.stack)

    def budget_left(self, risk_cap: float) -> int:
        """Chips we are still willing to put in voluntarily this hand."""
        return max(0, int(risk_cap * self.stack_at_hand_start) - self.committed_this_hand)


def _as_int(value, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def read_situation(request: dict, stats) -> Situation:
    """Pull the request apart defensively - every field gets a fallback, so a
    protocol addition or a missing key degrades instead of raising."""
    your_seat = _as_int(request.get("your_seat"))
    button_seat = _as_int(request.get("button_seat"))
    round_name = request.get("round")
    is_post = round_name == "post_reveal"

    players = request.get("players")
    players = players if isinstance(players, list) else []
    me = next(
        (p for p in players if isinstance(p, dict) and p.get("seat") == your_seat),
        {},
    )

    actions = request.get("current_hand_actions")
    actions = actions if isinstance(actions, list) else []
    opponent_aggressive = any(
        isinstance(a, dict)
        and a.get("seat") != your_seat
        and a.get("round") == round_name
        and a.get("action") in ("bet", "raise")
        for a in actions
    )
    # Anywhere in this hand, have they put chips in rather than passed? Once
    # they have, they are not folding to the next bullet either, so a second
    # bluff is throwing good chips after bad.
    opponent_contested = any(
        isinstance(a, dict)
        and a.get("seat") != your_seat
        and a.get("action") in ("call", "bet", "raise")
        for a in actions
    )

    total_hands = _as_int(request.get("total_hands"), 100)
    hand_number = _as_int(request.get("hand_number"), 1)
    starting_stack = _as_int(request.get("starting_stack"), 200)
    stack = _as_int(request.get("your_stack"), starting_stack)

    delta = me.get("chip_delta")
    if not isinstance(delta, int) or isinstance(delta, bool):
        delta = request.get("chip_delta")
    if not isinstance(delta, int) or isinstance(delta, bool):
        delta = stack - starting_stack

    legal = request.get("legal_actions")
    legal = tuple(legal) if isinstance(legal, list) else ()

    # The button acts first before the reveal and last after it. Acting last
    # is the advantage, so position flips between the two rounds.
    in_position = (your_seat == button_seat) if is_post else (your_seat != button_seat)

    return Situation(
        number=_as_int(request.get("your_number"), 7),
        community=request.get("community_number") if is_post else None,
        pot=_as_int(request.get("pot")),
        to_call=_as_int(request.get("to_call")),
        stack=stack,
        bet_this_round=_as_int(me.get("bet_this_round")),
        min_raise_to=request.get("min_raise_to"),
        max_raise_to=request.get("max_raise_to"),
        legal=legal,
        is_post_reveal=is_post,
        in_position=in_position,
        opponent_was_aggressive=opponent_aggressive,
        opponent_has_contested=opponent_contested,
        hands_left=max(0, total_hands - hand_number),
        chip_delta=delta,
        stack_at_hand_start=max(stack, starting_stack + delta),
    )


def pick_mode(situation: Situation) -> Mode:
    """Steer variance by where we stand against the +10 line."""
    if situation.hands_left > ENDGAME_HANDS:
        return GRIND
    if situation.chip_delta >= PROTECT_DELTA:
        return PROTECT
    if situation.chip_delta < TARGET_DELTA:
        return DESPERATE if situation.hands_left <= DESPERATE_HANDS else PUSH
    return GRIND


def hand_equity(situation: Situation, stats) -> float:
    """Equity against the range the opponent is representing right now.

    Only aggression narrows a range. The big blind is a forced bet, not a
    statement about their number, so an unraised pot is scored against a
    uniform opponent.
    """
    quantile = stats.bet_quantile if situation.opponent_was_aggressive else 1.0
    return eq.equity(situation.number, situation.community, quantile)


def _jitter(*parts) -> float:
    """Deterministic pseudo-random in [0, 1).

    Deterministic so a replay of the same spot gives the same answer and the
    simulator is reproducible; hashed over the hand state so our bluffs do
    not land in a pattern an opponent could key off.
    """
    seed = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(seed, digest_size=8).digest(), "big") / 2**64


def _bet_verb(legal) -> str | None:
    if "raise" in legal:
        return "raise"
    if "bet" in legal:
        return "bet"
    return None


def _raise_to(situation: Situation, size_fraction: float, mode: Mode) -> int | None:
    """Total for this round after raising, or None if no legal raise fits.

    Returns None rather than a clamped-up amount when the exposure cap cannot
    cover even the minimum raise - being forced to raise bigger than intended
    is how a cap turns into a bust.
    """
    low, high = situation.min_raise_to, situation.max_raise_to
    if not isinstance(low, int) or not isinstance(high, int) or high < low:
        return None

    target = situation.current_bet + max(1, round(size_fraction * max(situation.pot, 1)))

    # A pair cannot lose, so it is exempt - that is where the money is made.
    if not situation.has_pair:
        ceiling = situation.bet_this_round + situation.budget_left(mode.risk_cap)
        if ceiling < low:
            return None
        target = min(target, ceiling)

    target = max(low, min(high, target))
    return target if low <= target <= high else None


def _pair_size(stats) -> float:
    """Size a pair to whatever they will actually pay off."""
    folds = stats.fold_to_bet
    if folds < 0.30:
        return PAIR_SIZE_VS_STATION
    if folds < 0.50:
        return PAIR_SIZE_VS_NORMAL
    return PAIR_SIZE_VS_FOLDER


def _first_legal(legal, *preferred) -> dict:
    for action in preferred:
        if action in legal:
            return {"action": action}
    return {"action": legal[0]} if legal else {"action": "check"}


def decide(request: dict, stats) -> dict:
    situation = read_situation(request, stats)
    mode = pick_mode(situation)
    equity_now = hand_equity(situation, stats)
    legal = situation.legal

    if situation.is_post_reveal:
        value_bet = VALUE_BET_POST + mode.value_shift
        value_raise = VALUE_RAISE_POST + mode.value_shift
    else:
        value_bet = VALUE_BET_PRE + mode.value_shift
        value_raise = VALUE_RAISE_PRE + mode.value_shift

    verb = _bet_verb(legal)
    facing_bet = situation.to_call > 0
    value_threshold = value_raise if facing_bet else value_bet

    # --- put chips in for value -------------------------------------------
    if verb and equity_now >= value_threshold:
        size = _pair_size(stats) if situation.has_pair else VALUE_SIZE
        amount = _raise_to(situation, size, mode)
        if amount is not None:
            return {"action": verb, "amount": amount}

    # --- or as a bluff -----------------------------------------------------
    # This runs whether or not there is a bet to face. Pre-reveal on the
    # button `to_call` is always 1 for the blind, so a bluff that only fired
    # when checked to could never steal there - which is the single most
    # profitable spot against a bot that folds too much.
    if verb and not situation.has_pair and situation.stack > SHORT_STACK:
        size = BLUFF_RAISE_SIZE if facing_bet else BLUFF_SIZE
        amount = _raise_to(situation, size, mode)
        if amount is not None and _bluff_pays(situation, stats, mode, amount, equity_now, request):
            return {"action": verb, "amount": amount}

    if not facing_bet:
        return _first_legal(legal, "check", "call")

    # --- call or fold ------------------------------------------------------
    pot_odds = situation.to_call / max(1, situation.pot + situation.to_call)
    needed = pot_odds + CALL_MARGIN - (POSITION_CREDIT if situation.in_position else 0.0)

    if equity_now >= needed and "call" in legal:
        within_budget = situation.to_call <= situation.budget_left(mode.risk_cap)
        short = situation.stack <= SHORT_STACK
        if situation.has_pair or within_budget or short or equity_now >= CAP_OVERRIDE_EQUITY:
            return {"action": "call"}

    return _first_legal(legal, "fold", "check", "call")


def _bluff_pays(situation: Situation, stats, mode: Mode, amount: int, equity_now: float, request: dict) -> bool:
    """Whether their own fold rate covers what this bluff risks.

    Risking `risk` to win the pot breaks even exactly when they fold
    risk / (risk + pot) of the time. Deriving it from the real amount, rather
    than assuming a size, is what lets the same test cover an opening bet and
    a raise. A bot that never folds turns this off on its own.
    """
    if equity_now < BLUFF_MIN_EQUITY or situation.opponent_has_contested:
        return False

    risk = amount - situation.bet_this_round
    if risk <= 0:
        return False

    breakeven = risk / (risk + max(1, situation.pot))
    if stats.fold_to_bet < breakeven + BLUFF_EDGE:
        return False

    frequency = min(1.0, BLUFF_FREQUENCY * mode.bluff_multiplier)
    roll = _jitter(
        request.get("match_id"),
        request.get("hand_number"),
        request.get("round"),
        situation.number,
        len(request.get("current_hand_actions") or []),
    )
    return roll < frequency
