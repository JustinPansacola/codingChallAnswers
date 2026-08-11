"""A playable SHOWDOWN table, so the strategy can be measured instead of argued about.

This implements the published rules end to end - alternating forced bets, the
acting-order reversal between the two betting rounds, no-limit sizing with the
minimum-raise rule, all-in for less with the uncalled portion returned,
showdown, and busting - and serves each bot a request in exactly the wire
shape the real coordinator sends.

The engine is deliberately strict: any illegal action raises IllegalAction
rather than being quietly substituted the way the real coordinator would.
A substituted move looks like a small loss of chips; a raised one looks like
a bug, which is what it is.

    python -m tests.showdown_sim              # headline table
    python -m tests.showdown_sim --matches 400
"""

import argparse
import random
import statistics
from dataclasses import dataclass, field

from server.showdown import equity as eq
from server.showdown import policy
from server.showdown.model import OpponentStats

RANKS = (1, 13)
SMALL_BLIND = 1
BIG_BLIND = 2
STARTING_STACK = 200
HANDS_PER_MATCH = 100
RECENT_WINDOW = 20


class IllegalAction(Exception):
    pass


def hand_rank(number: int, community: int) -> tuple:
    """Sort key at showdown: any pair beats any non-pair, then higher number."""
    return (1 if number == community else 0, number)


@dataclass
class HandLog:
    hand_number: int
    community_number: int | None = None
    winners: list = field(default_factory=list)
    pot: int = 0
    shown_numbers: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)

    def as_wire(self) -> dict:
        return {
            "hand_number": self.hand_number,
            "community_number": self.community_number,
            "winners": list(self.winners),
            "pot": self.pot,
            "shown_numbers": {str(k): v for k, v in self.shown_numbers.items()},
            "actions": list(self.actions),
        }


class Table:
    """One heads-up match between two bots."""

    def __init__(self, bots, seed=0, hands=HANDS_PER_MATCH, match_id="sim"):
        self.bots = bots
        self.rng = random.Random(seed)
        self.hands = hands
        self.match_id = match_id
        self.stacks = [STARTING_STACK, STARTING_STACK]
        self.history: list[HandLog] = []
        self.hands_played = 0
        self.busted = False

    # --- the match ---------------------------------------------------------

    def play(self) -> list[int]:
        for index in range(self.hands):
            if min(self.stacks) <= 0:
                self.busted = True
                break
            self.play_hand(index + 1, button=index % 2)
            self.hands_played += 1
        if min(self.stacks) <= 0:
            self.busted = True
        return [s - STARTING_STACK for s in self.stacks]

    def play_hand(self, hand_number: int, button: int) -> None:
        self.hand_number = hand_number
        self.button = button
        self.numbers = [self.rng.randint(*RANKS), self.rng.randint(*RANKS)]
        self.community = self.rng.randint(*RANKS)
        self.folded = [False, False]
        self.all_in = [False, False]
        self.contributed = [0, 0]
        self.stack_at_start = list(self.stacks)
        self.log = HandLog(hand_number=hand_number)

        # The button pays the small blind and acts first before the reveal.
        other = 1 - button
        self.bet_this_round = [0, 0]
        self._commit(button, min(SMALL_BLIND, self.stacks[button]))
        self._commit(other, min(BIG_BLIND, self.stacks[other]))

        self.round = "pre_reveal"
        self._betting_round(first_to_act=button, opening_bet=BIG_BLIND)

        if not any(self.folded):
            self.round = "post_reveal"
            self.log.community_number = self.community
            self.bet_this_round = [0, 0]
            # After the reveal the order flips: the button now acts last.
            self._betting_round(first_to_act=other, opening_bet=0)

        self._settle()
        self.history.append(self.log)

    # --- betting -----------------------------------------------------------

    def _commit(self, seat: int, amount: int) -> None:
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.bet_this_round[seat] += amount
        self.contributed[seat] += amount
        if self.stacks[seat] == 0:
            self.all_in[seat] = True

    def _round_complete(self, acted, current_bet) -> bool:
        live = [s for s in (0, 1) if not self.folded[s]]
        if len(live) <= 1:
            return True
        actable = [s for s in live if not self.all_in[s]]
        if not actable:
            return True
        return all(acted[s] and self.bet_this_round[s] == current_bet for s in actable)

    def _betting_round(self, first_to_act: int, opening_bet: int) -> None:
        current_bet = max(self.bet_this_round)
        last_raise_size = max(opening_bet, BIG_BLIND)
        acted = [False, False]
        seat = first_to_act

        # Every raise lifts the current bet by at least the big blind and the
        # minimum only grows, so a legal round is bounded by the stack - but
        # bound the loop anyway so a strategy bug shows up as an error rather
        # than a hang.
        for _ in range(1000):
            if self._round_complete(acted, current_bet):
                return
            if self.folded[seat] or self.all_in[seat]:
                seat = 1 - seat
                continue

            opponent = 1 - seat
            to_call = min(current_bet - self.bet_this_round[seat], self.stacks[seat])
            can_escalate = self.stacks[seat] > to_call and not self.all_in[opponent]

            if to_call <= 0:
                legal = ["check"] + (["bet"] if can_escalate else [])
                min_raise_to = min(max(BIG_BLIND, 1), self.bet_this_round[seat] + self.stacks[seat])
            else:
                legal = ["fold", "call"] + (["raise"] if can_escalate else [])
                min_raise_to = current_bet + last_raise_size
            max_raise_to = self.bet_this_round[seat] + self.stacks[seat]
            if not can_escalate:
                min_raise_to = max_raise_to = None
            else:
                min_raise_to = min(min_raise_to, max_raise_to)

            reply = self.bots[seat](
                self._request(seat, to_call, legal, min_raise_to, max_raise_to)
            )
            action = reply.get("action")
            if action not in legal:
                raise IllegalAction(f"seat {seat} played {action!r}, legal was {legal}")

            if action == "fold":
                self.folded[seat] = True
                self.log.actions.append({"round": self.round, "seat": seat, "action": "fold"})
            elif action == "check":
                self.log.actions.append({"round": self.round, "seat": seat, "action": "check"})
            elif action == "call":
                self._commit(seat, to_call)
                self.log.actions.append(
                    {"round": self.round, "seat": seat, "action": "call",
                     "amount": self.bet_this_round[seat]}
                )
            else:  # bet or raise
                amount = reply.get("amount")
                if not isinstance(amount, int) or not (min_raise_to <= amount <= max_raise_to):
                    raise IllegalAction(
                        f"seat {seat} {action} to {amount!r}, legal range "
                        f"[{min_raise_to}, {max_raise_to}]"
                    )
                last_raise_size = max(amount - current_bet, last_raise_size)
                self._commit(seat, amount - self.bet_this_round[seat])
                current_bet = max(current_bet, self.bet_this_round[seat])
                acted = [False, False]
                self.log.actions.append(
                    {"round": self.round, "seat": seat, "action": action,
                     "amount": self.bet_this_round[seat]}
                )

            acted[seat] = True
            seat = 1 - seat
        raise IllegalAction("betting round failed to terminate")

    # --- payout ------------------------------------------------------------

    def _settle(self) -> None:
        # The uncalled part of a bet always comes back, which is also what
        # makes a fold cost the folder only what they had already put in.
        matched = min(self.contributed)
        for seat in (0, 1):
            if self.contributed[seat] > matched:
                self.stacks[seat] += self.contributed[seat] - matched
        pot = matched * 2
        self.log.pot = pot

        live = [s for s in (0, 1) if not self.folded[s]]
        if len(live) == 1:
            winners = live
        else:
            ranks = {s: hand_rank(self.numbers[s], self.community) for s in live}
            best = max(ranks.values())
            winners = [s for s in live if ranks[s] == best]
            self.log.shown_numbers = {s: self.numbers[s] for s in live}

        share, remainder = divmod(pot, len(winners))
        for i, seat in enumerate(winners):
            self.stacks[seat] += share + (remainder if i == 0 else 0)
        self.log.winners = winners

    # --- the wire ----------------------------------------------------------

    def _request(self, seat, to_call, legal, min_raise_to, max_raise_to) -> dict:
        pot = sum(self.contributed)
        return {
            "protocol_version": 2,
            "match_id": self.match_id,
            "phase": 1,
            "table_rule": "standard",
            "small_blind": SMALL_BLIND,
            "big_blind": BIG_BLIND,
            "starting_stack": STARTING_STACK,
            "your_stack": self.stacks[seat],
            "hand_number": self.hand_number,
            "total_hands": self.hands,
            "round": self.round,
            "your_number": self.numbers[seat],
            "community_number": self.community if self.round == "post_reveal" else None,
            "your_seat": seat,
            "button_seat": self.button,
            "pot": pot,
            "to_call": to_call,
            "min_raise_to": min_raise_to,
            "max_raise_to": max_raise_to,
            "legal_actions": list(legal),
            "players": [
                {
                    "seat": s,
                    "name": "you" if s == seat else "Gaston",
                    "folded": self.folded[s],
                    "chip_delta": self.stack_at_start[s] - STARTING_STACK,
                    "bet_this_round": self.bet_this_round[s],
                    "stack": self.stacks[s],
                    "all_in": self.all_in[s],
                    "busted": self.stacks[s] == 0 and not self.all_in[s],
                }
                for s in (0, 1)
            ],
            "current_hand_actions": list(self.log.actions),
            "recent_hands": [h.as_wire() for h in self.history[-RECENT_WINDOW:]],
        }


# --- our bot ---------------------------------------------------------------


def our_bot():
    """The real strategy, with per-match memory like the live server keeps."""
    stats = OpponentStats()

    def play(request):
        seat = request["your_seat"]
        stats.observe_recent(request.get("recent_hands"), 1 - seat)
        return policy.decide(request, stats)

    return play


# --- house-bot archetypes --------------------------------------------------
# Not attempts to guess the real opponent, but the corners of the space: the
# bot that never folds, the one that always folds, the one that never stops
# betting, and a competent one. A strategy that beats all four is not merely
# tuned to one leak.


def _clamp_raise(request, target):
    low, high = request["min_raise_to"], request["max_raise_to"]
    return int(max(low, min(high, round(target))))


def calling_station(request):
    """Never folds, never raises. Punishes bluffing, pays off value."""
    legal = request["legal_actions"]
    return {"action": "check"} if "check" in legal else {"action": "call"}


def over_folder(request):
    """Folds to any bet without a strong number. Should be bluffed relentlessly."""
    legal = request["legal_actions"]
    strong = _strength(request) >= 0.70
    if request["to_call"] > 0:
        if strong:
            return {"action": "call"}
        return {"action": "fold"} if "fold" in legal else {"action": "check"}
    if strong and "bet" in legal:
        return {"action": "bet", "amount": _clamp_raise(request, request["pot"] * 0.6)}
    return {"action": "check"}


def nit(request):
    """Plays only premium numbers, but bets them hard."""
    legal = request["legal_actions"]
    strength = _strength(request)
    if request["to_call"] > 0:
        if strength >= 0.85 and "raise" in legal:
            return {"action": "raise", "amount": _clamp_raise(request, request["pot"] * 0.75)}
        if strength >= 0.62:
            return {"action": "call"}
        return {"action": "fold"} if "fold" in legal else {"action": "check"}
    if strength >= 0.70 and "bet" in legal:
        return {"action": "bet", "amount": _clamp_raise(request, request["pot"] * 0.7)}
    return {"action": "check"}


def maniac(seed=0):
    """Bets and raises relentlessly, folds only to serious pressure."""
    rng = random.Random(seed)

    def play(request):
        legal = request["legal_actions"]
        strength = _strength(request)
        if "raise" in legal or "bet" in legal:
            if rng.random() < 0.65:
                verb = "raise" if "raise" in legal else "bet"
                size = request["pot"] * rng.choice((0.8, 1.2, 2.0))
                return {"action": verb, "amount": _clamp_raise(request, size)}
        if request["to_call"] > 0:
            over_half = request["to_call"] > request["your_stack"] * 0.5
            if over_half and strength < 0.5 and "fold" in legal:
                return {"action": "fold"}
            return {"action": "call"}
        return {"action": "check"}

    return play


def random_bot(seed=0):
    rng = random.Random(seed)

    def play(request):
        action = rng.choice(request["legal_actions"])
        if action in ("bet", "raise"):
            low, high = request["min_raise_to"], request["max_raise_to"]
            return {"action": action, "amount": rng.randint(low, min(high, low * 4))}
        return {"action": action}

    return play


def solid(request):
    """A competent equity-and-pot-odds opponent, with no bluffing."""
    legal = request["legal_actions"]
    strength = _strength(request)
    if request["to_call"] > 0:
        odds = request["to_call"] / (request["pot"] + request["to_call"])
        if strength >= 0.78 and "raise" in legal:
            return {"action": "raise", "amount": _clamp_raise(request, request["pot"] * 0.8)}
        if strength >= odds + 0.03:
            return {"action": "call"}
        return {"action": "fold"} if "fold" in legal else {"action": "check"}
    if strength >= 0.60 and "bet" in legal:
        return {"action": "bet", "amount": _clamp_raise(request, request["pot"] * 0.65)}
    return {"action": "check"}


def _strength(request) -> float:
    return eq.uniform_equity(request["your_number"], request.get("community_number"))


ARCHETYPES = {
    "calling_station": lambda seed: calling_station,
    "over_folder": lambda seed: over_folder,
    "nit": lambda seed: nit,
    "maniac": maniac,
    "random": random_bot,
    "solid": lambda seed: solid,
}


# --- running ---------------------------------------------------------------


def run_matches(opponent_name, matches=200, hands=HANDS_PER_MATCH, seed0=0):
    """Play `matches` matches, alternating which seat we occupy."""
    deltas, busts = [], 0
    for i in range(matches):
        seed = seed0 + i
        opponent = ARCHETYPES[opponent_name](seed)
        us = our_bot()
        our_seat = i % 2  # alternate seats so no positional bias survives
        bots = [us, opponent] if our_seat == 0 else [opponent, us]
        table = Table(bots, seed=seed, hands=hands, match_id=f"{opponent_name}-{seed}")
        result = table.play()
        deltas.append(result[our_seat])
        if table.stacks[our_seat] <= 0:
            busts += 1
    return deltas, busts


def summarize(name, deltas, busts):
    cleared = sum(1 for d in deltas if d >= 10)
    return {
        "opponent": name,
        "median": statistics.median(deltas),
        "mean": round(statistics.mean(deltas), 1),
        "clear_rate": cleared / len(deltas),
        "bust_rate": busts / len(deltas),
        "worst": min(deltas),
        "best": max(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--hands", type=int, default=HANDS_PER_MATCH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default=None)
    args = parser.parse_args()

    names = [args.opponent] if args.opponent else list(ARCHETYPES)
    header = f"{'opponent':<16}{'median':>8}{'mean':>8}{'clear':>8}{'bust':>7}{'worst':>8}{'best':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for name in names:
        deltas, busts = run_matches(name, args.matches, args.hands, args.seed)
        row = summarize(name, deltas, busts)
        rows.append(row)
        print(
            f"{row['opponent']:<16}{row['median']:>8.0f}{row['mean']:>8.1f}"
            f"{row['clear_rate']:>8.0%}{row['bust_rate']:>7.0%}"
            f"{row['worst']:>8.0f}{row['best']:>8.0f}"
        )
    print(f"\n{args.matches} matches x {args.hands} hands each, seat alternated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
