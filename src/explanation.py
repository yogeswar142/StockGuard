"""
src/explanation.py - Plain-English Reason Generator for StockGuard.

Converts mathematical parameters into intuitive, store-manager-ready explanations:
Example:
    "Usually sells ~3.2 units/day (~₹760/day margin at risk). Zero sales for 8 days
     (system shows 9 units in stock). Peer stores sold 34 units this week."

Signal suppression rules (avoids overconfident mentions of weak evidence):
  - Peer-store sentence: only shown if peer_sales_7d >= 10 (≥10 chain units last 7 days).
  - Substitute sentence: only shown if substitute_lift_pct >= 15 AND sub_units_7d >= 5.
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
    confidence_pct: float = 0.0,
    sub_units_7d: int = 0,
) -> str:
    """
    Constructs a plain-English, actionable explanation sentence for a flagged alert.

    Only mentions corroborating signals when they are large enough to be meaningful:
      • Peer-store demand: mentioned only if >= 10 units across chain peers (last 7 days).
      • Substitute lift: mentioned only if lift >= 15% AND substitute sold >= 5 units.
    """
    reason_parts = []

    # Core velocity + revenue context (always included)
    vel_str = (
        f"Usually sells ~{expected_daily_demand:.1f} units/day "
        f"(~₹{daily_revenue_inr:,.0f}/day margin at risk)"
    )
    streak_str = (
        f"Zero sales for {zero_streak_days} consecutive days "
        f"with {system_stock} units recorded in stock"
    )
    reason_parts.append(f"{vel_str}. {streak_str}.")

    # Peer-store signal — only mention when >= 10 chain units last 7 days
    if peer_sales_7d >= 10:
        reason_parts.append(
            f"Peer stores sold {peer_sales_7d} units this week — chain demand is active."
        )

    # Substitute category lift — only mention when lift >= 15% AND volume >= 5 units
    if substitute_name and substitute_lift_pct >= 15.0 and sub_units_7d >= 5:
        reason_parts.append(
            f"Category substitute '{substitute_name}' up +{substitute_lift_pct:.0f}% "
            f"({sub_units_7d} units) — shoppers may be buying alternatives."
        )

    return " ".join(reason_parts)

