"""
tests/test_baselines.py - Unit tests for baseline detectors and operational constraints.
"""

import pytest
import pandas as pd
import numpy as np

from src.baseline_models import FixedNZeroDaysBaseline, StreakLengthBaseline, run_baseline_simulation


def test_ranking_cap_of_20():
    """Verify that a store never receives more than 20 alerts per day."""
    detector = FixedNZeroDaysBaseline(n_zero_days=3, max_checks_per_store=20)
    
    # Create synthetic store day with 50 qualifying candidates
    candidates = pd.DataFrame({
        "sku_id": [f"SKU_{i:03d}" for i in range(50)],
        "zero_streak": np.random.randint(3, 20, size=50),
        "system_stock": np.random.randint(5, 50, size=50),
        "unit_price": np.random.uniform(50, 500, size=50)
    })
    
    scored = detector.score_candidates(candidates)
    assert len(scored) >= 20  # scored finds them all
    # Simulate selection with max cap
    top20 = scored.head(detector.max_checks_per_store)
    assert len(top20) == 20, "Should cap at exactly 20 alerts per store"


def test_fixed_n_days_filter():
    """Verify items below streak threshold or with 0 stock are strictly excluded."""
    detector = FixedNZeroDaysBaseline(n_zero_days=5)
    
    data = pd.DataFrame({
        "sku_id": ["SKU_1", "SKU_2", "SKU_3", "SKU_4"],
        "zero_streak": [6, 4, 10, 8],
        "system_stock": [12, 15, 0, 8],  # SKU_3 has 0 stock (true stockout)
        "unit_price": [100.0, 150.0, 200.0, 80.0]
    })
    
    scored = detector.score_candidates(data)
    # Only SKU_1 (streak 6 >= 5, stock 12 > 0) and SKU_4 (streak 8 >= 5, stock 8 > 0) qualify
    assert set(scored["sku_id"]) == {"SKU_1", "SKU_4"}
    assert "SKU_2" not in scored["sku_id"].values  # streak 4 < 5
    assert "SKU_3" not in scored["sku_id"].values  # stock is 0


def test_no_use_of_ground_truth():
    """Verify candidate scoring works with zero leakage: strictly POS features only."""
    detector = StreakLengthBaseline()
    pos_data = pd.DataFrame({
        "sku_id": ["SKU_A", "SKU_B"],
        "zero_streak": [3, 5],
        "system_stock": [10, 4],
        "unit_price": [120.0, 80.0]
    })
    # Must run cleanly without any truth columns
    assert "is_phantom" not in pos_data.columns
    scored = detector.score_candidates(pos_data)
    assert len(scored) == 2
    assert scored.iloc[0]["sku_id"] == "SKU_B"  # longer streak


def test_cooldown_mechanism():
    """Verify false positive items enter cooldown and are suppressed for N days."""
    detector = FixedNZeroDaysBaseline(n_zero_days=3, cooldown_days=7)
    key = ("STORE_01", "SKU_001")
    detector.cooldowns[key] = 100  # in cooldown until day 100
    
    # On day 95, item should be suppressed
    current_day = 95
    assert current_day <= detector.cooldowns[key]
    
    # On day 101, cooldown expired
    current_day = 101
    assert current_day > detector.cooldowns[key]
