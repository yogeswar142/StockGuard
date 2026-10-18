"""
src/explanation.py - Plain-English Reason Generator for StockGuard.

Converts mathematical parameters into intuitive, store-manager-ready explanations:
Example:
    "Usually sells ~3.2 units/day (about ₹760/day). Zero sales for 8 days (system shows 9 units in stock).
     Peer stores sold 34 units this week. Category substitute demand up +28%."
"""

from typing import Optional


def generate_plain_english_reason(
    expected_daily_demand: float,
    daily_revenue_inr: float,
    zero_streak_days: int,
    system_stock: int,
    peer_sales_7d: int,
    substitute_name: Optional[str] = None,
    substitute_lift_pct: float = 0.0,
    confidence_pct: float = 0.0
) -> str:
    """
    Constructs a plain-English, actionable explanation sentence for a flagged alert.
    """
    reason_parts = []
    
    # Core velocity and revenue context
    vel_str = f"Usually sells ~{expected_daily_demand:.1f} units/day (~₹{daily_revenue_inr:,.0f}/day margin at risk)"
    streak_str = f"Zero sales for {zero_streak_days} consecutive days with {system_stock} units recorded in stock"
    reason_parts.append(f"{vel_str}. {streak_str}.")
    
    # Corroborating peer store signal
    if peer_sales_7d > 0:
        reason_parts.append(f"Nearby stores sold {peer_sales_7d} units this week (chain demand active).")
    else:
        reason_parts.append("Zero sales across chain peers this week.")
        
    # Substitute category lift
    if substitute_name and substitute_lift_pct > 10.0:
        reason_parts.append(f"Demand for substitute '{substitute_name}' surged +{substitute_lift_pct:.0f}% in this store.")
        
    return " ".join(reason_parts)
