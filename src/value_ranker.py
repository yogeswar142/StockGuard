"""
src/value_ranker.py - Operational Value-Based Prioritization Engine for StockGuard.

Ranks alerts under the manager's 20-check daily budget:
    Expected Value = P(Phantom) * Daily Margin at Risk * Expected Days Saved - Audit Cost
    
    where:
    - Daily Margin at Risk = lambda * unit_price * gross_margin_pct (INR)
    - Expected Days Saved = max(2, mean_cycle_days - streak_days)
    - Audit Cost = ~5 INR (2 minutes of staff time at 150 INR/hr)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union


class ValueRanker:
    """
    Ranks candidate items by expected economic value of shelf audit,
    enforcing a hard cap of at most 20 checks per store per day.
    """
    def __init__(
        self,
        max_checks_per_store: int = 20,
        mean_cycle_count_interval: float = 21.0,
        audit_cost_inr: float = 5.0,
        min_posterior_threshold: Union[float, Dict[str, float]] = 0.35
    ):
        self.max_checks_per_store = max_checks_per_store
        self.mean_cycle_count_interval = mean_cycle_count_interval
        self.audit_cost_inr = audit_cost_inr
        self.min_posterior_threshold = min_posterior_threshold

    def compute_expected_value(
        self,
        posterior_p: float,
        expected_daily_demand: float,
        unit_price: float,
        gross_margin_pct: float,
        zero_streak_days: int
    ) -> Dict[str, float]:
        """
        Calculates daily margin at risk, expected days saved, and net expected value.
        """
        daily_margin_at_risk = expected_daily_demand * unit_price * gross_margin_pct
        days_saved = max(2.0, self.mean_cycle_count_interval - float(zero_streak_days))
        
        gross_expected_recovery = posterior_p * daily_margin_at_risk * days_saved
        net_expected_value = gross_expected_recovery - self.audit_cost_inr
        
        return {
            "daily_margin_at_risk": round(daily_margin_at_risk, 2),
            "days_saved": round(days_saved, 1),
            "expected_value_inr": round(net_expected_value, 2)
        }

    def rank_store_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks candidate items for a single store-day by expected value and takes top 20.
        Expects columns: ['posterior_p', 'expected_value_inr', 'daily_margin_at_risk', 'zero_streak']
        """
        if candidates_df.empty:
            return pd.DataFrame()
            
        # Filter: must meet minimum statistical posterior confidence and positive expected value
        if isinstance(self.min_posterior_threshold, dict) and "velocity_band" in candidates_df.columns:
            thresh_series = candidates_df["velocity_band"].map(self.min_posterior_threshold).fillna(0.35)
            eligible = candidates_df[
                (candidates_df["posterior_p"] >= thresh_series) &
                (candidates_df["expected_value_inr"] > 0)
            ].copy()
        else:
            thresh = float(self.min_posterior_threshold) if not isinstance(self.min_posterior_threshold, dict) else 0.35
            eligible = candidates_df[
                (candidates_df["posterior_p"] >= thresh) &
                (candidates_df["expected_value_inr"] > 0)
            ].copy()

        
        if eligible.empty:
            return pd.DataFrame()
            
        # Sort descending by expected value, then by posterior confidence, then by margin at risk
        eligible = eligible.sort_values(
            by=["expected_value_inr", "posterior_p", "daily_margin_at_risk"],
            ascending=[False, False, False]
        )
        
        # Enforce hard top-20 budget cap
        top_alerts = eligible.head(self.max_checks_per_store).copy()
        top_alerts["rank"] = np.arange(1, len(top_alerts) + 1)
        return top_alerts
