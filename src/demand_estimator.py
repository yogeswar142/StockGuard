"""
src/demand_estimator.py - Expected Demand Estimation for StockGuard.

Calculates expected daily demand lambda_{s, i, t} for each SKU-store-day:
- Rolling baseline rate over prior window (e.g. 56 days) of active/in-stock days
- Day-of-week seasonal multiplier (e.g. weekend lift)
- Promotional lift multiplier (from promo status or observed promo index)
- Calibrated solely on historical POS data without ground truth access.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class DemandEstimator:
    """
    Estimates expected demand per SKU-store-day using rolling history and seasonality factors.
    """
    def __init__(self, rolling_window_days: int = 56, min_history_days: int = 14):
        self.rolling_window_days = rolling_window_days
        self.min_history_days = min_history_days
        
        # Precomputed global or store-level day-of-week multipliers
        self.dow_multipliers = np.array([0.88, 0.90, 0.93, 0.96, 1.10, 1.30, 1.25])
        
        # SKU-store baseline rate: (store_id, sku_id) -> float
        self.sku_store_baselines: Dict[Tuple[str, str], float] = {}
        # Sibling substitute mapping: sku_id -> substitute_sku_id
        self.substitutes: Dict[str, str] = {}

    def fit_calibration_period(self, df_calibration: pd.DataFrame, df_sku: Optional[pd.DataFrame] = None):
        """
        Calibrates baselines and Day-of-Week multipliers using the calibration period.
        """
        if df_sku is not None and "substitute_sku_id" in df_sku.columns:
            for r in df_sku.itertuples():
                if pd.notna(r.substitute_sku_id):
                    self.substitutes[r.sku_id] = str(r.substitute_sku_id)

        # Estimate day-of-week multipliers from calibration data
        df_cal = df_calibration.copy()
        df_cal["dow"] = pd.to_datetime(df_cal["date"]).dt.weekday
        
        # Average sales per day of week normalized by overall mean
        overall_mean = df_cal["sales"].mean()
        if overall_mean > 0:
            dow_means = df_cal.groupby("dow")["sales"].mean()
            self.dow_multipliers = (dow_means / overall_mean).values
            
        # Store-SKU baseline mean demand during non-zero stock periods
        in_stock_cal = df_cal[df_cal["system_stock"] > 0]
        means = in_stock_cal.groupby(["store_id", "sku_id"])["sales"].mean()
        for (s_id, sku_id), val in means.items():
            self.sku_store_baselines[(s_id, sku_id)] = max(0.02, float(val))

    def estimate_demand(
        self,
        store_id: str,
        sku_id: str,
        dow: int,
        on_promo: int = 0,
        rolling_sales_window: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimates lambda for a specific SKU-store on day with given dow and promo status.
        """
        # Base rate from rolling window if provided and sufficient, else fallback to calibrated baseline
        if rolling_sales_window is not None and len(rolling_sales_window) >= self.min_history_days:
            # Exclude extreme outliers or recent zero streaks when computing healthy baseline
            non_zero_recent = rolling_sales_window[rolling_sales_window > 0]
            if len(non_zero_recent) >= 5:
                base_rate = float(np.mean(non_zero_recent) * (len(non_zero_recent) / len(rolling_sales_window)))
            else:
                base_rate = float(np.mean(rolling_sales_window))
            base_rate = max(0.02, base_rate)
        else:
            base_rate = self.sku_store_baselines.get((store_id, sku_id), 0.5)

        dow_factor = self.dow_multipliers[dow] if dow < len(self.dow_multipliers) else 1.0
        promo_factor = 2.0 if on_promo == 1 else 1.0
        
        expected_lambda = base_rate * dow_factor * promo_factor
        return max(0.01, expected_lambda)
