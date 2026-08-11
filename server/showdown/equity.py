"""Equity math for SHOWDOWN.

Each player holds one number 1-13 drawn independently, and one community
number is drawn the same way. A number equal to the community number is a
pair, and any pair beats any non-pair; otherwise the higher number wins and
equal numbers split.

Two facts drive the whole strategy:

  * A pair cannot lose. The only holding that matches it is the community
    number itself, which is a tie, so a made pair is worth 12.5/13 = 0.9615
    and every chip you can get in with one is free.
  * Pre-reveal equity is linear in your number - 0.1095 at 1, exactly 0.5000
    at 7, 0.8905 at 13, a flat 0.0651 per step. There is no "playable /
    unplayable" cliff to find; strength is a dial.

Less obviously, a non-pair is worth *more* when the community number is
above it (8 is worth 0.58 against a community 10 but only 0.50 against a
community 3). When the community number sits below yours it is a number you
would otherwise have beaten, and instead it makes a pair that beats you -
the swing counts twice.

Everything here is a pure function of small integers, so the tables are
built once at import and the range-restricted equities are 13-element loops.
"""

RANKS = tuple(range(1, 14))
N_RANKS = len(RANKS)

# Never narrow an opponent to fewer than this fraction of their range. A
# model that says "they only ever hold the community number" makes our own
# pair look like a coin flip, which is exactly backwards.
MIN_RANGE_QUANTILE = 0.25

# How much to trust the range read over the raw uniform equity. The model is
# inferred from a house bot's behaviour over a handful of hands, so it gets a
# majority vote but never the whole vote.
RANGE_WEIGHT = 0.65


def outcome(ours: int, theirs: int, community: int) -> float:
    """Our share of the pot: 1.0 win, 0.5 split, 0.0 loss."""
    ours_paired = ours == community
    theirs_paired = theirs == community
    if ours_paired != theirs_paired:
        return 1.0 if ours_paired else 0.0
    if ours == theirs:
        return 0.5
    return 1.0 if ours > theirs else 0.0


def _pre_equity(ours: int) -> float:
    return sum(
        outcome(ours, theirs, community) for theirs in RANKS for community in RANKS
    ) / (N_RANKS * N_RANKS)


def _post_equity(ours: int, community: int) -> float:
    return sum(outcome(ours, theirs, community) for theirs in RANKS) / N_RANKS


# Indexed by rank, so slot 0 is a placeholder to keep the indexing honest.
PRE_EQUITY = (0.0,) + tuple(_pre_equity(n) for n in RANKS)
POST_EQUITY = ((),) + tuple(
    (0.0,) + tuple(_post_equity(n, c) for c in RANKS) for n in RANKS
)

PAIR_EQUITY = POST_EQUITY[1][1]  # 12.5/13, identical for every rank


def pre_equity(ours: int) -> float:
    """Equity before the reveal against a uniformly random opponent."""
    return PRE_EQUITY[ours]


def post_equity(ours: int, community: int) -> float:
    """Equity after the reveal against a uniformly random opponent."""
    return POST_EQUITY[ours][community]


def uniform_equity(ours: int, community: int | None) -> float:
    return pre_equity(ours) if community is None else post_equity(ours, community)


def strength_order(community: int | None) -> tuple[int, ...]:
    """Opponent holdings from strongest to weakest.

    After the reveal the community number itself leads - it is the one
    holding that cannot lose - and the rest follow in descending order.
    Before the reveal strength is simply the number.
    """
    if community is None:
        return tuple(sorted(RANKS, reverse=True))
    return (community,) + tuple(sorted((r for r in RANKS if r != community), reverse=True))


def _top_slice(community: int | None, quantile: float) -> tuple[int, ...]:
    order = strength_order(community)
    keep = max(1, min(len(order), round(quantile * len(order))))
    return order[:keep]


def equity_vs_top_quantile(ours: int, community: int | None, quantile: float) -> float:
    """Equity when the opponent only holds their strongest `quantile` of hands.

    A bot that has just bet is not holding a uniform number, so raw equity
    overstates us. `quantile` comes from how often the opponent actually
    bets: one that bets everything collapses this back to uniform equity.
    """
    hands = _top_slice(community, quantile)
    if community is None:
        return sum(
            outcome(ours, theirs, c) for theirs in hands for c in RANKS
        ) / (len(hands) * N_RANKS)
    return sum(outcome(ours, theirs, community) for theirs in hands) / len(hands)


def equity(ours: int, community: int | None, quantile: float = 1.0) -> float:
    """Blended equity: mostly the range read, partly the uniform baseline.

    The blend is deliberate. The range read is inferred from a small sample
    of a bot we have never seen before, and a confidently wrong read is worse
    than no read - anchoring it to the uniform number bounds how far a bad
    model can push us.
    """
    flat = uniform_equity(ours, community)
    if quantile >= 1.0:
        return flat
    narrowed = equity_vs_top_quantile(ours, community, max(quantile, MIN_RANGE_QUANTILE))
    return RANGE_WEIGHT * narrowed + (1.0 - RANGE_WEIGHT) * flat
