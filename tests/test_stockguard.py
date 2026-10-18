"""
tests/test_stockguard.py - Unit tests for StockGuard Core statistical detection & ranking.
"""

import pytest
import numpy as np
import pandas as pd

from src.streak_probability import streak_probability_poisson, streak_probability_negbin, compute_streak_surprise
from src.bayesian_detector import BayesianPhantomDetector
from src.value_ranker import ValueRanker
from src.explanation import generate_plain_english_reason


def test_streak_probability_fast_vs_slow():
    """
    Core Plan Section 2.2 Proof:
    Fast item (lambda=2.0, 3 zero days) -> e^-6 ~ 0.25% (Very suspicious)
    Slow item (lambda=0.1, 30 zero days) -> e^-3 ~ 5.0% (Not suspicious)
    """
    p_fast = streak_probability_poisson(lambdas=2.0, streak_days=3)
    assert p_fast < 0.003, f"Expected <0.3%, got {p_fast*100:.2f}%"

    p_slow = streak_probability_poisson(lambdas=0.1, streak_days=30)
    assert p_slow > 0.04, f"Expected ~5%, got {p_slow*100:.2f}%"

    # Fast item surprise should be > 99%, slow item surprise should be much lower
    assert compute_streak_surprise(2.0, 3) > 0.99
    assert compute_streak_surprise(0.1, 5) < 0.50


def test_bayesian_posterior_monotonicity():
    """Verify that as zero streak increases, posterior probability strictly increases."""
    detector = BayesianPhantomDetector(prior_phantom_rate=0.03)
    p_day2 = detector.compute_posterior(expected_daily_demand=2.0, zero_streak_days=2)
    p_day4 = detector.compute_posterior(expected_daily_demand=2.0, zero_streak_days=4)
    p_day7 = detector.compute_posterior(expected_daily_demand=2.0, zero_streak_days=7)
    
    assert p_day2 < p_day4 < p_day7
    assert p_day7 > 0.90, "7 zero days on a 2 unit/day item must have high posterior"


def test_value_ranking_budget_cap():
    """Verify that ValueRanker enforces exactly at most 20 checks per store-day."""
    ranker = ValueRanker(max_checks_per_store=20, min_posterior_threshold=0.30)
    
    # 60 eligible candidates
    df = pd.DataFrame({
        "sku_id": [f"SKU_{i:03d}" for i in range(60)],
        "posterior_p": np.random.uniform(0.5, 0.95, size=60),
        "expected_value_inr": np.random.uniform(10, 500, size=60),
        "daily_margin_at_risk": np.random.uniform(20, 200, size=60),
        "zero_streak": np.random.randint(3, 15, size=60)
    })
    
    ranked = ranker.rank_store_candidates(df)
    assert len(ranked) == 20
    assert ranked["rank"].tolist() == list(range(1, 21))
    # Must be sorted descending by expected value
    assert ranked["expected_value_inr"].is_monotonic_decreasing


def test_plain_english_reason():
    """Verify reason text is informative, plain English, and contains key numbers."""
    reason = generate_plain_english_reason(
        expected_daily_demand=3.5,
        daily_revenue_inr=840.0,
        zero_streak_days=6,
        system_stock=12,
        peer_sales_7d=42,
        substitute_name="Tata Tea Gold",
        substitute_lift_pct=34.0,
        confidence_pct=92.5
    )
    assert "Usually sells ~3.5 units/day" in reason
    assert "840" in reason
    assert "6 consecutive days" in reason
    assert "12 units recorded in stock" in reason
    assert "Nearby stores sold 42 units this week" in reason
    assert "Tata Tea Gold" in reason
    assert "+34%" in reason
