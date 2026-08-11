"""Search the strategy's constants against the simulated table.

The constants in policy.py are not obvious from first principles - a bluff
threshold that beats a bot which folds too much is the same threshold that
loses to one that never folds. Rather than guess them one at a time, this
searches them, then re-scores the winner on seeds it never saw.

The objective deliberately weights the *worst* archetype heavily. The real
house bot is unknown, so a configuration that averages well by crushing three
archetypes and losing to a fourth is worth less than one that clears them all.

    python -m tests.showdown_tune                 # full search, then validate
    python -m tests.showdown_tune --evaluate      # just score what is checked in
"""

import argparse
import random
from multiprocessing import Pool

from tests import showdown_sim as sim

# Attribute path -> (low, high) for the search.
SEARCH_SPACE = {
    "policy.VALUE_BET_PRE": (0.42, 0.72),
    "policy.VALUE_BET_POST": (0.45, 0.75),
    "policy.VALUE_RAISE_PRE": (0.55, 0.88),
    "policy.VALUE_RAISE_POST": (0.55, 0.88),
    "policy.CALL_MARGIN": (-0.02, 0.14),
    "policy.POSITION_CREDIT": (0.0, 0.08),
    "policy.BLUFF_SIZE": (0.35, 0.90),
    "policy.BLUFF_RAISE_SIZE": (0.50, 1.20),
    "policy.VALUE_SIZE": (0.40, 1.20),
    "policy.BLUFF_EDGE": (0.0, 0.30),
    "policy.BLUFF_FREQUENCY": (0.0, 1.0),
    "policy.BLUFF_MIN_EQUITY": (0.0, 0.55),
    "policy.RISK_CAP": (0.10, 0.60),
    "policy.CAP_OVERRIDE_EQUITY": (0.60, 0.95),
    "policy.PAIR_SIZE_VS_STATION": (0.80, 2.50),
    "policy.PAIR_SIZE_VS_NORMAL": (0.50, 1.60),
    "policy.PAIR_SIZE_VS_FOLDER": (0.30, 1.20),
    "equity.RANGE_WEIGHT": (0.0, 1.0),
    "equity.MIN_RANGE_QUANTILE": (0.10, 0.60),
}

TRAIN_SEED = 0
VALIDATE_SEED = 100_000
TRAIN_MATCHES = 60
VALIDATE_MATCHES = 400


def apply(config: dict) -> None:
    from server.showdown import equity, policy

    modules = {"policy": policy, "equity": equity}
    for path, value in config.items():
        module, name = path.split(".")
        setattr(modules[module], name, value)


def score(config: dict, matches: int, seed0: int) -> dict:
    """Clear rate per archetype under `config`."""
    apply(config)
    rates = {}
    for name in sim.ARCHETYPES:
        deltas, _ = sim.run_matches(name, matches=matches, seed0=seed0)
        rates[name] = sum(1 for d in deltas if d >= 10) / len(deltas)
    return rates


def objective(rates: dict) -> float:
    """Worst archetype dominates; the average breaks ties.

    A bot that clears every archetype 70% of the time is worth more than one
    that clears five at 95% and the sixth at 20%, because we do not get to
    choose which one shows up.
    """
    values = list(rates.values())
    return min(values) + 0.35 * (sum(values) / len(values))


def _evaluate(job):
    config, matches, seed0 = job
    rates = score(config, matches, seed0)
    return objective(rates), rates, config


def random_config(rng: random.Random) -> dict:
    return {path: rng.uniform(low, high) for path, (low, high) in SEARCH_SPACE.items()}


def current_config() -> dict:
    from server.showdown import equity, policy

    modules = {"policy": policy, "equity": equity}
    return {path: getattr(modules[path.split(".")[0]], path.split(".")[1]) for path in SEARCH_SPACE}


def refine(pool, best_config, best_score, rng, rounds=2):
    """Coordinate descent: nudge one constant at a time, keep what helps."""
    for round_index in range(rounds):
        for path, (low, high) in SEARCH_SPACE.items():
            span = (high - low) * (0.30 if round_index == 0 else 0.12)
            candidates = []
            for _ in range(6):
                trial = dict(best_config)
                trial[path] = min(high, max(low, best_config[path] + rng.uniform(-span, span)))
                candidates.append((trial, TRAIN_MATCHES, TRAIN_SEED))
            for value, rates, config in pool.map(_evaluate, candidates):
                if value > best_score:
                    best_score, best_config = value, config
        print(f"  refine round {round_index + 1}: objective {best_score:.3f}")
    return best_config, best_score


def report(title, rates):
    print(f"\n{title}")
    for name, rate in rates.items():
        print(f"  {name:<16}{rate:>7.0%}")
    print(f"  {'objective':<16}{objective(rates):>7.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--evaluate", action="store_true", help="score the checked-in constants")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.evaluate:
        report("checked-in constants (validation seeds)",
               score(current_config(), VALIDATE_MATCHES, VALIDATE_SEED))
        return 0

    rng = random.Random(args.seed)
    baseline = current_config()

    with Pool() as pool:
        jobs = [(baseline, TRAIN_MATCHES, TRAIN_SEED)]
        jobs += [(random_config(rng), TRAIN_MATCHES, TRAIN_SEED) for _ in range(args.samples)]
        results = pool.map(_evaluate, jobs)
        best_score, _, best_config = max(results, key=lambda r: r[0])
        print(f"random search best objective: {best_score:.3f}")
        best_config, best_score = refine(pool, best_config, best_score, rng)

    report("best config (training seeds)", score(best_config, TRAIN_MATCHES, TRAIN_SEED))
    report("best config (held-out seeds)", score(best_config, VALIDATE_MATCHES, VALIDATE_SEED))
    report("baseline (held-out seeds)", score(baseline, VALIDATE_MATCHES, VALIDATE_SEED))

    print("\nconstants:")
    for path, value in sorted(best_config.items()):
        print(f"  {path:<36}= {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
