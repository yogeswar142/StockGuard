"""
src/baseline_models.py - Heuristic Baseline Detection Models for StockGuard.

Implements standard industry heuristics:
1. FixedNZeroDaysBaseline: Flags items with system_stock > 0 and zero-sales streak >= N.
2. StreakLengthBaseline: Ranks all items with system_stock > 0 by raw zero-streak length.

Includes daily operational constraints:
- Hard cap of at most 20 checks per store per day.
- Audit feedback & cooldown loop:
  - True phantom audited -> resolved / inventory corrected.
  - False positive audited -> enters N-day cooldown to prevent list starvation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set


class BaseDetector:
    """Base class for all phantom stock detectors (heuristics and StockGuard)."""
    def __init__(self, name: str, cooldown_days: int = 7, max_checks_per_store: int = 20):
        self.name = name
        self.cooldown_days = cooldown_days
        self.max_checks_per_store = max_checks_per_store
        
        # State tracking: (store_id, sku_id) -> cooldown_until_day
        self.cooldowns: Dict[Tuple[str, str], int] = {}
        # Tracking resolved phantom episodes: set of episode_ids or (store, sku, day)
        self.resolved_episodes: Set[str] = set()

    def reset_state(self):
        self.cooldowns.clear()
        self.resolved_episodes.clear()


class FixedNZeroDaysBaseline(BaseDetector):
    """
    Industry Baseline A: Fixed N-zero-days rule.
    Flags items where sales have been 0 for >= N consecutive days with system stock > 0.
    Ranks ties by unit price * zero_days or longest streak, capped at 20 per store-day.
    """
    def __init__(self, n_zero_days: int = 5, cooldown_days: int = 7, max_checks_per_store: int = 20):
        super().__init__(f"Fixed_{n_zero_days}_Days_Rule", cooldown_days, max_checks_per_store)
        self.n_zero_days = n_zero_days

    def score_candidates(self, df_store_day: pd.DataFrame) -> pd.DataFrame:
        """
        Scores candidate items in a store on a given day.
        Expects columns: ['sku_id', 'zero_streak', 'system_stock', 'unit_price']
        """
        # Filter: stock > 0 and streak >= N
        mask = (df_store_day["system_stock"] > 0) & (df_store_day["zero_streak"] >= self.n_zero_days)
        candidates = df_store_day[mask].copy()
        if len(candidates) == 0:
            return pd.DataFrame()
            
        # Score by streak length (longer first) then price
        candidates["baseline_score"] = candidates["zero_streak"] * 1000.0 + candidates["unit_price"]
        candidates = candidates.sort_values("baseline_score", ascending=False)
        return candidates


class StreakLengthBaseline(BaseDetector):
    """
    Industry Baseline B: Streak-Length Ranking.
    Ranks all items with system stock > 0 and streak >= 1 strictly by zero-streak length.
    Top 20 longest zero streaks per store per day are checked.
    """
    def __init__(self, min_streak: int = 1, cooldown_days: int = 7, max_checks_per_store: int = 20):
        super().__init__("Streak_Length_Ranking", cooldown_days, max_checks_per_store)
        self.min_streak = min_streak

    def score_candidates(self, df_store_day: pd.DataFrame) -> pd.DataFrame:
        """
        Scores candidate items in a store on a given day.
        Expects columns: ['sku_id', 'zero_streak', 'system_stock', 'unit_price']
        """
        mask = (df_store_day["system_stock"] > 0) & (df_store_day["zero_streak"] >= self.min_streak)
        candidates = df_store_day[mask].copy()
        if len(candidates) == 0:
            return pd.DataFrame()
            
        # Score directly by streak length, break ties with unit price
        candidates["baseline_score"] = candidates["zero_streak"] * 1000.0 + candidates["unit_price"]
        candidates = candidates.sort_values("baseline_score", ascending=False)
        return candidates


def run_baseline_simulation(
    detector: BaseDetector,
    df_pos: pd.DataFrame,
    df_truth: Optional[pd.DataFrame] = None,
    eval_start_date: Optional[str] = None,
    resolve_phantoms: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Runs day-by-day simulation of baseline detector with cooldown and audit feedback.
    
    If df_truth is provided:
      - Simulates physical store manager check on flagged items.
      - If item is a true phantom: resolves it (cannot trigger future alerts).
      - If item is not phantom: places SKU into cooldown for N days.
    """
    detector.reset_state()
    
    # Sort chronologically
    df_pos = df_pos.sort_values(["date", "store_id", "sku_id"]).copy()
    
    # Track historical zero streaks across time
    # Store-SKU state: (store_id, sku_id) -> current zero streak
    streaks: Dict[Tuple[str, str], int] = {}
    
    # Pre-index ground truth if available for fast lookup
    truth_map = {}
    if df_truth is not None:
        for r in df_truth.itertuples():
            truth_map[(r.date, r.store_id, r.sku_id)] = {
                "is_phantom": r.is_phantom,
                "episode_id": r.episode_id,
                "days_since_onset": r.days_since_onset
            }
            
    dates = sorted(df_pos["date"].unique())
    if eval_start_date is None:
        eval_idx = int(len(dates) * 0.6)
        eval_start_date = dates[eval_idx]
        
    all_alerts = []
    daily_stats = []
    
    # Set of currently resolved active phantoms
    active_resolved_episodes = set()

    grouped_by_date = list(df_pos.groupby("date", sort=True))

    for day_idx, (date_str, df_day) in enumerate(grouped_by_date):
        is_eval_day = (date_str >= eval_start_date)
        
        # 1. Update streak counts for all store-SKU rows today
        if not is_eval_day:
            # Fast streak update during calibration period
            for row in df_day.itertuples():
                key = (row.store_id, row.sku_id)
                streaks[key] = 0 if row.sales > 0 else (streaks.get(key, 0) + 1)
            continue

        # During evaluation period: record streak and prepare candidates
        streak_vals = []
        for row in df_day.itertuples():
            key = (row.store_id, row.sku_id)
            if row.sales > 0:
                streaks[key] = 0
            else:
                streaks[key] = streaks.get(key, 0) + 1
            streak_vals.append(streaks[key])
            
        df_day_features = df_day.copy()
        df_day_features["zero_streak"] = streak_vals
            
        # 2. For each store, score candidates and select top 20 respecting cooldown
        day_alerts_count = 0
        day_true_positives = 0
        
        for s_id, s_group in df_day_features.groupby("store_id"):
            candidates = detector.score_candidates(s_group)
            if candidates.empty:
                continue
                
            # Filter out SKUs currently in cooldown
            eligible = []
            for c_row in candidates.itertuples():
                key = (s_id, c_row.sku_id)
                cooldown_until = detector.cooldowns.get(key, -1)
                
                # Check if this SKU was already resolved during current episode
                if df_truth is not None:
                    t_info = truth_map.get((date_str, s_id, c_row.sku_id), {})
                    ep_id = t_info.get("episode_id")
                    if ep_id and ep_id in active_resolved_episodes:
                        continue  # Already audited and resolved!
                        
                if day_idx > cooldown_until:
                    eligible.append(c_row)
                    if len(eligible) >= detector.max_checks_per_store:
                        break
                        
            # Record alerts and simulate feedback
            for rank, c_row in enumerate(eligible, start=1):
                key = (s_id, c_row.sku_id)
                day_alerts_count += 1
                
                # Check ground truth
                is_true_phantom = 0
                ep_id = None
                onset_delay = -1
                if df_truth is not None:
                    t_info = truth_map.get((date_str, s_id, c_row.sku_id), {})
                    is_true_phantom = t_info.get("is_phantom", 0)
                    ep_id = t_info.get("episode_id")
                    onset_delay = t_info.get("days_since_onset", -1)
                    
                if is_true_phantom:
                    day_true_positives += 1
                    if resolve_phantoms and ep_id:
                        # Episode caught! Resolved in simulation
                        active_resolved_episodes.add(ep_id)
                        # Streak resets in real world after physical stock corrected
                        streaks[key] = 0
                else:
                    # False positive check -> apply cooldown
                    detector.cooldowns[key] = day_idx + detector.cooldown_days
                    
                all_alerts.append({
                    "date": date_str,
                    "store_id": s_id,
                    "sku_id": c_row.sku_id,
                    "rank": rank,
                    "model": detector.name,
                    "zero_streak": c_row.zero_streak,
                    "system_stock": c_row.system_stock,
                    "unit_price": c_row.unit_price,
                    "is_true_phantom": is_true_phantom,
                    "episode_id": ep_id,
                    "days_since_onset": onset_delay
                })
                
        daily_stats.append({
            "date": date_str,
            "alerts": day_alerts_count,
            "true_positives": day_true_positives
        })

    df_alerts = pd.DataFrame(all_alerts)
    return df_alerts, {"daily_stats": daily_stats, "eval_start_date": eval_start_date}
