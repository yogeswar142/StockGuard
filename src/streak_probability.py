"""
src/streak_probability.py - Statistical Zero-Streak Probability Calculator.

Implements explainable probabilistic likelihoods of observing N consecutive zero-sales days:
1. Poisson Model: P(zero streak | healthy) = exp(-sum(lambda))
2. Negative Binomial Model: Handles over-dispersed retail demand (clustering, bulk buying).
   Var = lambda + alpha * lambda^2.
   P(X = 0) = (r / (r + lambda))^r where r = 1 / alpha.
"""

import numpy as np
from typing import Union, List


def streak_probability_poisson(lambdas: Union[float, np.ndarray, List[float]], streak_days: int = 1) -> float:
    """
    Computes Poisson probability of observing zero sales over consecutive days.
    If lambdas is a scalar, assumes constant expected demand lambda per day.
    """
    if isinstance(lambdas, (list, np.ndarray)):
        cum_lambda = float(np.sum(lambdas))
    else:
        cum_lambda = float(lambdas * streak_days)
        
    # Clip to prevent numerical underflow
    cum_lambda = max(0.0, cum_lambda)
    if cum_lambda > 700:
        return 0.0
    return float(np.exp(-cum_lambda))


def streak_probability_negbin(
    lambdas: Union[float, np.ndarray, List[float]],
    streak_days: int = 1,
    alpha: float = 0.30
) -> float:
    """
    Computes Negative Binomial probability of observing zero sales over consecutive days.
    Accounts for retail demand over-dispersion (clustering, lumpiness).
    alpha: dispersion parameter where Var = lambda + alpha * lambda^2.
    """
    r = 1.0 / max(0.01, alpha)
    
    if isinstance(lambdas, (list, np.ndarray)):
        log_prob = 0.0
        for lam in lambdas:
            lam = max(0.001, float(lam))
            p_single_zero = (r / (r + lam)) ** r
            log_prob += np.log(max(1e-15, p_single_zero))
        return float(np.exp(log_prob))
    else:
        lam = max(0.001, float(lambdas))
        p_single_zero = (r / (r + lam)) ** r
        return float(p_single_zero ** streak_days)


def compute_streak_surprise(
    expected_daily_demand: float,
    streak_days: int,
    use_negbin: bool = True,
    alpha: float = 0.30
) -> float:
    """
    Returns the statistical 'surprise' score in [0.0, 1.0]:
    Surprise = 1.0 - P(zero streak | healthy in-stock).
    """
    if use_negbin:
        p_luck = streak_probability_negbin(expected_daily_demand, streak_days, alpha=alpha)
    else:
        p_luck = streak_probability_poisson(expected_daily_demand, streak_days)
    return float(np.clip(1.0 - p_luck, 0.0, 1.0))
