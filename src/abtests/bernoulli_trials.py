__all__ = ["equals_hypothesis"]

from math import sqrt
from scipy.stats import norm


def f_derivative(x: float) -> float:
    value = 1 - 2 * x
    value /= 2 * sqrt(x * (1 - x))
    return value


def correction(avg: float, size: int, sigmas: float) -> float:
    std = sqrt(avg * (1-avg) / size)
    left_ = avg - sigmas * std
    right_ = avg + sigmas * std
    max_derivative = max(abs(f_derivative(left_)), abs(f_derivative(right_)))
    inv_corr = 1 + sigmas * max_derivative/sqrt(size)
    return 1 / inv_corr


def equals_hypothesis(a_len: int, a_successes: int,
                      b_len: int, b_successes: int,
                      verbose: int = 0,
                      alternative: str = "two-sided") -> float:
    """P-value when testing of means are equal."""

    if verbose == 1:
        print("A has", a_len, "elements")
        print("B has", b_len, "elements")

    xa = a_successes / a_len
    xb = b_successes / b_len
    x = (a_successes + b_successes) / (a_len + b_len)

    d = xa - xb
    std = sqrt(x * (1 - x) * (1 / a_len + 1 / b_len))
    value = d/std
    if verbose == 1:
        print("statistic", value)

    corr = correction(avg=x, size=(a_len + b_len), sigmas=5)
    if verbose == 1:
        print("correction", corr)
    if alternative in ["two-sided", "2s", "a==b"]:
        p_value = 2 * norm.sf(abs(value) * corr)
    elif alternative == "a>b":
        if value <= 0:
            if verbose == 1:
                print("P-value capped at 0.5")
            p_value = 0.5
        else:
            p_value = norm.sf(value * corr)
    elif alternative == "a<b":
        if value >= 0:
            if verbose == 1:
                print("P-value capped at 0.5")
            p_value = 0.5
        else:
            p_value = norm.sf(-value * corr)
    else:
        msg = ("alternative must be one of 'two-sided', '2s', 'a==b', "
               + "'a>b' or 'a<b'")
        raise ValueError(msg)

    if verbose == 1:
        print("pval", p_value)

    return p_value
