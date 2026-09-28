from __future__ import annotations

from math import comb, sqrt
from statistics import NormalDist


def _wilson_interval(successes: int, total: int, alpha: float) -> tuple[float, float]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("invalid binomial counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def paired_risk_difference(
    both_success: int,
    first_only: int,
    second_only: int,
    both_failure: int,
    *,
    alpha: float = 0.05,
) -> dict:
    """Return Newcombe method 10 interval and exact McNemar evidence.

    The risk difference is first-system success minus second-system success.
    The confidence interval accounts for pairing and remains within [-1, 1].
    """
    cells = (both_success, first_only, second_only, both_failure)
    if any(not isinstance(value, int) or value < 0 for value in cells):
        raise ValueError("paired table cells must be non-negative integers")
    total = sum(cells)
    if total < 1:
        raise ValueError("paired table must contain at least one pair")

    first_successes = both_success + first_only
    second_successes = both_success + second_only
    first_rate = first_successes / total
    second_rate = second_successes / total
    first_lower, first_upper = _wilson_interval(first_successes, total, alpha)
    second_lower, second_upper = _wilson_interval(second_successes, total, alpha)

    numerator = both_success * both_failure - first_only * second_only
    if numerator > 0:
        numerator = max(numerator - total / 2.0, 0.0)
    denominator = sqrt(
        (both_success + first_only)
        * (second_only + both_failure)
        * (both_success + second_only)
        * (first_only + both_failure)
    )
    correlation = 0.0 if denominator == 0.0 else numerator / denominator

    lower_distance = sqrt(
        (first_rate - first_lower) ** 2
        - 2.0
        * correlation
        * (first_rate - first_lower)
        * (second_upper - second_rate)
        + (second_upper - second_rate) ** 2
    )
    upper_distance = sqrt(
        (first_upper - first_rate) ** 2
        - 2.0
        * correlation
        * (first_upper - first_rate)
        * (second_rate - second_lower)
        + (second_rate - second_lower) ** 2
    )
    difference = first_rate - second_rate

    discordant = first_only + second_only
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        smaller = min(first_only, second_only)
        lower_tail = sum(comb(discordant, value) for value in range(smaller + 1))
        mcnemar_p = min(1.0, 2.0 * lower_tail / (2**discordant))

    return {
        "pairs": total,
        "first_rate": first_rate,
        "second_rate": second_rate,
        "risk_difference": difference,
        "confidence_level": 1.0 - alpha,
        "confidence_interval": [
            max(-1.0, difference - lower_distance),
            min(1.0, difference + upper_distance),
        ],
        "confidence_interval_method": "Newcombe paired method 10",
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": mcnemar_p,
    }
