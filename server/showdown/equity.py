"""Equity math for SHOWDOWN, across all four table rules.

Each player holds one number 1-13 drawn independently, and one community
number is drawn the same way. Only the *showdown* changes between rules -
betting, blinds, position and sizing are identical - so the entire rule
difference collapses into one function, `rank`, and every table below is
derived from it rather than written out by hand. Adding a fifth rule would
mean editing `rank` and nothing else.

    standard      pair (number == community) beats non-pair, then higher wins
    low_ball      inverted: non-pair beats pair, then *lower* wins
    wild_seven    a 7 is always a pair of 7s; higher pair wins
    pair_bounty   ranks exactly as standard, but winning a showdown holding a
                  pair collects 5 chips from the house (handled in policy.py,
                  since it changes what a call is worth, not who wins)

The rule is not a detail. Under `standard` a 2 is nearly unplayable at 0.175
equity; under `low_ball` the same 2 is worth 0.825. Reading `table_rule` off
the request instead of assuming is the whole phase.

Two structural facts survive in some form under every rule:

  * Some holding cannot lose - it wins or ties, never loses. Under `standard`
    that is a pair; under `low_ball` it is the *lowest* non-pair. `is_locked`
    finds it generically, which is what the risk cap keys off.
  * Equity is monotone in the number (ascending or descending by rule), so
    strength is a dial rather than a cliff - except under `wild_seven`, where
    holding a 7 jumps out of the ordering entirely.
"""

RANKS = tuple(range(1, 14))
N_RANKS = len(RANKS)

STANDARD = "standard"
LOW_BALL = "low_ball"
WILD_SEVEN = "wild_seven"
PAIR_BOUNTY = "pair_bounty"
RULES = (STANDARD, LOW_BALL, WILD_SEVEN, PAIR_BOUNTY)
DEFAULT_RULE = STANDARD

WILD_NUMBER = 7
PAIR_BOUNTY_CHIPS = 5

# Never narrow an opponent to fewer than this fraction of their range. A
# model that says "they only ever hold the community number" makes our own
# pair look like a coin flip, which is exactly backwards.
MIN_RANGE_QUANTILE = 0.211

# How much to trust the range read over the raw uniform equity. The model is
# inferred from a house bot's behaviour over a handful of hands, so it gets a
# majority vote but never the whole vote.
RANGE_WEIGHT = 0.732


def normalize_rule(rule) -> str:
    return rule if rule in RULES else DEFAULT_RULE


def rank(number: int, community: int, rule: str = STANDARD) -> tuple:
    """Showdown strength, comparable with `<`. Higher tuple wins, equal ties.

    This is the only place a table rule is spelled out.
    """
    if rule == LOW_BALL:
        # Non-pairs beat pairs, and lower beats higher - so negate both keys.
        return (0 if number == community else 1, -number)
    if rule == WILD_SEVEN:
        # A 7 is a pair of 7s whatever the community number is. Matching the
        # community is still a pair of that number, and the higher pair wins,
        # so the pair's *value* is what goes in the tuple.
        if number == WILD_NUMBER:
            return (1, WILD_NUMBER)
        return (1, number) if number == community else (0, number)
    return (1 if number == community else 0, number)


def holds_pair(number: int, community: int | None, rule: str = STANDARD) -> bool:
    """Whether this holding counts as a pair under the rule."""
    if community is None:
        return False
    if rule == WILD_SEVEN:
        return number == WILD_NUMBER or number == community
    return number == community


def outcome(ours: int, theirs: int, community: int, rule: str = STANDARD) -> float:
    """Our share of the pot: 1.0 win, 0.5 split, 0.0 loss."""
    mine, yours = rank(ours, community, rule), rank(theirs, community, rule)
    if mine == yours:
        return 0.5
    return 1.0 if mine > yours else 0.0


def _post_table(rule):
    return ((),) + tuple(
        (0.0,) + tuple(
            sum(outcome(n, m, c, rule) for m in RANKS) / N_RANKS for c in RANKS
        )
        for n in RANKS
    )


def _pre_table(rule):
    return (0.0,) + tuple(
        sum(outcome(n, m, c, rule) for m in RANKS for c in RANKS) / (N_RANKS * N_RANKS)
        for n in RANKS
    )


# Built once at import: four rules by 13 by 13 is trivial, and it keeps every
# hot path a table lookup.
PRE_EQUITY = {rule: _pre_table(rule) for rule in RULES}
POST_EQUITY = {rule: _post_table(rule) for rule in RULES}


def pre_equity(ours: int, rule: str = STANDARD) -> float:
    """Equity before the reveal against a uniformly random opponent."""
    return PRE_EQUITY[rule][ours]


def post_equity(ours: int, community: int, rule: str = STANDARD) -> float:
    """Equity after the reveal against a uniformly random opponent."""
    return POST_EQUITY[rule][ours][community]


def uniform_equity(ours: int, community: int | None, rule: str = STANDARD) -> float:
    rule = normalize_rule(rule)
    return pre_equity(ours, rule) if community is None else post_equity(ours, community, rule)


def is_locked(ours: int, community: int | None, rule: str = STANDARD) -> bool:
    """Whether this holding cannot lose - it wins or ties against everything.

    Derived rather than special-cased, so it stays correct per rule: a pair
    under `standard`, the lowest non-pair under `low_ball`, and under
    `wild_seven` only a pair of 13s (a pair of 7s loses to a higher pair).
    Chips in with a locked hand are free, which is why the risk cap exempts it.
    """
    if community is None:
        return False
    return all(outcome(ours, m, community, rule) >= 0.5 for m in RANKS)


def strength_order(community: int | None, rule: str = STANDARD) -> tuple[int, ...]:
    """Opponent holdings from strongest to weakest under this rule.

    Ordered by each holding's own equity rather than by number, so it inverts
    for `low_ball` and lifts the 7 out of sequence for `wild_seven` without
    any of that being written down twice.
    """
    return tuple(
        sorted(RANKS, key=lambda m: uniform_equity(m, community, rule), reverse=True)
    )


def _top_slice(community: int | None, quantile: float, rule: str) -> tuple[int, ...]:
    order = strength_order(community, rule)
    keep = max(1, min(len(order), round(quantile * len(order))))
    return order[:keep]


def equity_vs_top_quantile(
    ours: int, community: int | None, quantile: float, rule: str = STANDARD
) -> float:
    """Equity when the opponent only holds their strongest `quantile` of hands.

    A bot that has just bet is not holding a uniform number, so raw equity
    overstates us. `quantile` comes from how often the opponent actually
    bets: one that bets everything collapses this back to uniform equity.
    """
    rule = normalize_rule(rule)
    hands = _top_slice(community, quantile, rule)
    if community is None:
        return sum(
            outcome(ours, theirs, c, rule) for theirs in hands for c in RANKS
        ) / (len(hands) * N_RANKS)
    return sum(outcome(ours, theirs, community, rule) for theirs in hands) / len(hands)


def equity(
    ours: int, community: int | None, quantile: float = 1.0, rule: str = STANDARD
) -> float:
    """Blended equity: mostly the range read, partly the uniform baseline.

    The blend is deliberate. The range read is inferred from a small sample
    of a bot we have never seen before, and a confidently wrong read is worse
    than no read - anchoring it to the uniform number bounds how far a bad
    model can push us.
    """
    rule = normalize_rule(rule)
    flat = uniform_equity(ours, community, rule)
    if quantile >= 1.0:
        return flat
    narrowed = equity_vs_top_quantile(
        ours, community, max(quantile, MIN_RANGE_QUANTILE), rule
    )
    return RANGE_WEIGHT * narrowed + (1.0 - RANGE_WEIGHT) * flat
