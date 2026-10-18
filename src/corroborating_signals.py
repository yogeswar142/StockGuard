"""
src/corroborating_signals.py - Corroborating Signals Engine for StockGuard.

Computes multi-source evidence:
1. Peer-Store Sales: Normal sales in peer stores confirm regional demand is active.
2. Substitute SKU Lift: Demand transferring to category sibling item.
3. Flat System Stock: POS record frozen for N consecutive days with zero delta.
4. Inter-Sale Recency Ratio: Streak length normalized by expected inter-purchase time (1 / lambda).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class CorroboratingEvidenceEngine:
    """
    Extracts multi-signal corroborating evidence across peer stores and substitute SKUs.
    """
    def __init__(self, substitute_map: Optional[Dict[str, str]] = None):
        self.substitute_map = substitute_map or {}

    def compute_peer_signal(
        self,
        target_store: str,
        target_sku: str,
        recent_peer_sales_df: pd.DataFrame
    ) -> Tuple[float, int]:
        """
        Calculates total peer store sales for target_sku in past 7 days (excluding target_store).
        Returns:
            peer_multiplier: evidence weight in [0.8, 1.8]
            peer_units_7d: total units sold across peers
        """
        if recent_peer_sales_df.empty:
            return 1.0, 0
            
        peer_mask = (recent_peer_sales_df["sku_id"] == target_sku) & (recent_peer_sales_df["store_id"] != target_store)
        peer_sales = int(recent_peer_sales_df.loc[peer_mask, "sales"].sum())
        
        # If peer stores are selling well, high confidence that product demand is active
        if peer_sales >= 25:
            peer_mult = 1.75
        elif peer_sales >= 10:
            peer_mult = 1.45
        elif peer_sales >= 3:
            peer_mult = 1.20
        elif peer_sales == 0:
            # Chain-wide freeze: could be supplier delisting or general lull
            peer_mult = 0.85
        else:
            peer_mult = 1.0
            
        return peer_mult, peer_sales

    def compute_substitute_lift(
        self,
        target_store: str,
        target_sku: str,
        recent_store_sales_df: pd.DataFrame,
        substitute_baseline: float
    ) -> Tuple[float, float, Optional[str]]:
        """
        Checks if the sibling substitute SKU in this store has elevated sales (+20% to +60%).
        Returns:
            sub_multiplier: evidence weight in [1.0, 1.5]
            sub_lift_pct: percentage lift observed
            sub_sku_id: id of the substitute item
        """
        sub_sku = self.substitute_map.get(target_sku)
        if not sub_sku or recent_store_sales_df.empty:
            return 1.0, 0.0, None
            
        sub_sales = recent_store_sales_df[
            (recent_store_sales_df["store_id"] == target_store) & 
            (recent_store_sales_df["sku_id"] == sub_sku)
        ]["sales"].sum()
        
        expected_sub_sales = max(0.5, substitute_baseline * len(recent_store_sales_df["date"].unique()))
        lift_ratio = sub_sales / expected_sub_sales
        
        if lift_ratio >= 1.30:
            sub_mult = 1.40
            lift_pct = round((lift_ratio - 1.0) * 100, 1)
        elif lift_ratio >= 1.15:
            sub_mult = 1.20
            lift_pct = round((lift_ratio - 1.0) * 100, 1)
        else:
            sub_mult = 1.0
            lift_pct = 0.0
            
        return sub_mult, lift_pct, sub_sku

    def compute_flat_stock_signal(self, stock_history: np.ndarray) -> float:
        """
        Checks if system stock has remained completely frozen (variance == 0) over the streak.
        """
        if len(stock_history) < 3:
            return 1.0
        # If stock is unchanged across recent days, confirmation that system is frozen
        if np.all(stock_history == stock_history[0]):
            return 1.25
        return 1.0
