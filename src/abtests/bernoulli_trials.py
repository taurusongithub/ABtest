__all__ = ["equals_hypothesis"]

from math import sqrt
import numpy as np
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


def equals_hypothesis(a_results: np.array,
                      b_results: np.array,
                      verbose: int = 0) -> float:
    """P-value when testing of means are equal."""

    a_len = len(a_results)
    b_len = len(b_results)
    if verbose == 1:
        print("A has", a_len, "elements")
        print("B has", b_len, "elements")

    xa = np.mean(a_results)
    xb = np.mean(b_results)
    x = (np.sum(a_results) + np.sum(b_results)) / (a_len + b_len)

    d = xa - xb
    std = sqrt(x * (1 - x) * (1 / a_len + 1 / b_len))
    value = abs(d)/std
    if verbose == 1:
        print("statistic", value)

    corr = correction(avg=x, size=(a_len + b_len), sigmas=5)
    if verbose == 1:
        print("correction", corr)

    p_value = 2 * norm.sf(value * corr)
    if verbose == 1:
        print("pval", p_value)

    return p_value
