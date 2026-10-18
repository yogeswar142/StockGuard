"""
src/stockguard_detector.py - Integrated StockGuard Detection Engine.

Combines:
1. DemandEstimator (rolling baseline + weekday + promo factors)
2. Negative Binomial Streak Probability (per-SKU statistical surprise)
3. BayesianPhantomDetector (odds-based posterior with prior)
4. CorroboratingEvidenceEngine (peer stores, substitute lift, flat inventory)
5. ValueRanker (top-20 budget optimization by expected rupees recovered)
6. Plain-English Explainability Generator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set

from src.baseline_models import BaseDetector
from src.demand_estimator import DemandEstimator
from src.bayesian_detector import BayesianPhantomDetector
from src.corroborating_signals import CorroboratingEvidenceEngine
from src.value_ranker import ValueRanker
from src.explanation import generate_plain_english_reason


class StockGuardDetector(BaseDetector):
    """
    StockGuard Core Phantom Stock-Out Detector.
    """
    def __init__(
        self,
        cooldown_days: int = 7,
        max_checks_per_store: int = 20,
        prior_phantom_rate: float = 0.028,
        min_posterior_threshold: float = 0.35,
        overdispersion_alpha: float = 0.30,
        audit_cost_inr: float = 5.0
    ):
        super().__init__("StockGuard_Core", cooldown_days, max_checks_per_store)
        self.demand_estimator = DemandEstimator()
        self.bayesian_detector = BayesianPhantomDetector(
            prior_phantom_rate=prior_phantom_rate,
            alpha_dispersion=overdispersion_alpha
        )
        self.corroborating_engine = CorroboratingEvidenceEngine()
        self.value_ranker = ValueRanker(
            max_checks_per_store=max_checks_per_store,
            audit_cost_inr=audit_cost_inr,
            min_posterior_threshold=min_posterior_threshold
        )
        self.sku_names: Dict[str, str] = {}
        self.sku_substitutes: Dict[str, str] = {}

    def fit_calibration(self, df_calibration: pd.DataFrame, df_sku: Optional[pd.DataFrame] = None):
        """
        Calibrates model components strictly on the calibration period without ground truth labels.
        """
        self.demand_estimator.fit_calibration_period(df_calibration, df_sku)
        if df_sku is not None:
            for r in df_sku.itertuples():
                self.sku_names[r.sku_id] = r.sku_name
                if pd.notna(r.substitute_sku_id):
                    self.sku_substitutes[r.sku_id] = str(r.substitute_sku_id)
            self.corroborating_engine.substitute_map = self.sku_substitutes


def run_stockguard_simulation(
    detector: StockGuardDetector,
    df_pos: pd.DataFrame,
    df_truth: Optional[pd.DataFrame] = None,
    df_sku: Optional[pd.DataFrame] = None,
    eval_start_date: Optional[str] = None,
    resolve_phantoms: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Runs day-by-day StockGuard detection simulation across time series.
    """
    detector.reset_state()
    df_pos = df_pos.sort_values(["date", "store_id", "sku_id"]).copy()
    
    unique_dates = sorted(df_pos["date"].unique())
    if eval_start_date is None:
        split_idx = int(len(unique_dates) * 0.6)
        eval_start_date = unique_dates[split_idx]
    else:
        split_idx = unique_dates.index(eval_start_date) if eval_start_date in unique_dates else int(len(unique_dates) * 0.6)
        
    # Fit detector strictly on calibration data
    df_cal = df_pos[df_pos["date"] < eval_start_date].copy()
    detector.fit_calibration(df_cal, df_sku)
    
    # Pre-index ground truth if available for fast evaluation lookup
    truth_map = {}
    if df_truth is not None:
        for r in df_truth.itertuples():
            truth_map[(r.date, r.store_id, r.sku_id)] = {
                "is_phantom": r.is_phantom,
                "episode_id": r.episode_id,
                "days_since_onset": r.days_since_onset
            }
            
    # Track historical states
    streaks: Dict[Tuple[str, str], int] = {}
    recent_stock_history: Dict[Tuple[str, str], List[int]] = {}
    
    # Rolling 7-day sales buffer for peer store analysis
    recent_days_sales: List[pd.DataFrame] = []
    
    all_alerts = []
    daily_stats = []
    active_resolved_episodes = set()
    
    grouped_by_date = list(df_pos.groupby("date", sort=True))
    
    for day_idx, (date_str, df_day) in enumerate(grouped_by_date):
        is_eval_day = (date_str >= eval_start_date)
        dt_dow = pd.to_datetime(date_str).weekday()
        
        # Maintain rolling 7-day sales window for peer analysis
        recent_days_sales.append(df_day[["date", "store_id", "sku_id", "sales"]])
        if len(recent_days_sales) > 7:
            recent_days_sales.pop(0)
            
        # Update zero streaks and stock history
        if not is_eval_day:
            for r in df_day.itertuples():
                key = (r.store_id, r.sku_id)
                streaks[key] = 0 if r.sales > 0 else (streaks.get(key, 0) + 1)
                hist = recent_stock_history.setdefault(key, [])
                hist.append(r.system_stock)
                if len(hist) > 10:
                    hist.pop(0)
            continue
            
        # Evaluation day: update states and score candidates
        for r in df_day.itertuples():
            key = (r.store_id, r.sku_id)
            if r.sales > 0:
                streaks[key] = 0
            else:
                streaks[key] = streaks.get(key, 0) + 1
            hist = recent_stock_history.setdefault(key, [])
            hist.append(r.system_stock)
            if len(hist) > 10:
                hist.pop(0)
                
        df_peer_window = pd.concat(recent_days_sales, ignore_index=True)
        total_sku_sales_7d = df_peer_window.groupby("sku_id")["sales"].sum().to_dict()
        store_sku_sales_7d = df_peer_window.groupby(["store_id", "sku_id"])["sales"].sum().to_dict()
        
        day_alerts_count = 0
        day_true_positives = 0
        
        # Process each store
        for s_id, s_group in df_day.groupby("store_id"):
            candidate_rows = []
            
            for r in s_group.itertuples():
                key = (s_id, r.sku_id)
                streak = streaks[key]
                stock = r.system_stock
                
                # Minimum eligibility: positive system stock and at least 2 zero sales days
                if stock <= 0 or streak < 2:
                    continue
                    
                # Check cooldown
                if day_idx <= detector.cooldowns.get(key, -1):
                    continue
                    
                # Check if episode was already resolved in simulation
                if df_truth is not None:
                    t_info = truth_map.get((date_str, s_id, r.sku_id), {})
                    ep_id = t_info.get("episode_id")
                    if ep_id and ep_id in active_resolved_episodes:
                        continue
                        
                # 1. Expected demand lambda
                exp_lambda = detector.demand_estimator.estimate_demand(
                    store_id=s_id,
                    sku_id=r.sku_id,
                    dow=dt_dow,
                    on_promo=r.on_promo
                )
                
                # 2. Corroborating signals via fast dictionary lookups
                total_sku_sold = total_sku_sales_7d.get(r.sku_id, 0)
                this_store_sold = store_sku_sales_7d.get((s_id, r.sku_id), 0)
                peer_units_7d = max(0, total_sku_sold - this_store_sold)
                
                if peer_units_7d >= 25:
                    peer_mult = 1.75
                elif peer_units_7d >= 10:
                    peer_mult = 1.45
                elif peer_units_7d >= 3:
                    peer_mult = 1.20
                elif peer_units_7d == 0:
                    peer_mult = 0.85
                else:
                    peer_mult = 1.0
                    
                sub_sku_id = detector.sku_substitutes.get(r.sku_id)
                if sub_sku_id:
                    sub_sales = store_sku_sales_7d.get((s_id, sub_sku_id), 0)
                    sub_base = detector.demand_estimator.sku_store_baselines.get((s_id, sub_sku_id), 1.0)
                    expected_sub_sales = max(0.5, sub_base * len(recent_days_sales))
                    lift_ratio = sub_sales / expected_sub_sales
                    if lift_ratio >= 1.30:
                        sub_mult = 1.40
                        sub_lift_pct = round((lift_ratio - 1.0) * 100, 1)
                        sub_name = detector.sku_names.get(sub_sku_id, sub_sku_id)
                    elif lift_ratio >= 1.15:
                        sub_mult = 1.20
                        sub_lift_pct = round((lift_ratio - 1.0) * 100, 1)
                        sub_name = detector.sku_names.get(sub_sku_id, sub_sku_id)
                    else:
                        sub_mult = 1.0
                        sub_lift_pct = 0.0
                        sub_name = None
                else:
                    sub_mult = 1.0
                    sub_lift_pct = 0.0
                    sub_name = None
                    
                stock_hist = np.array(recent_stock_history.get(key, [stock]))
                flat_mult = 1.25 if (len(stock_hist) >= 3 and np.all(stock_hist == stock_hist[0])) else 1.0
                
                # 3. Bayesian Posterior
                posterior_p = detector.bayesian_detector.compute_posterior(
                    expected_daily_demand=exp_lambda,
                    zero_streak_days=streak,
                    peer_multiplier=peer_mult,
                    substitute_multiplier=sub_mult,
                    flat_stock_multiplier=flat_mult,
                    use_negbin=True
                )
                
                # 4. Expected Value
                val_dict = detector.value_ranker.compute_expected_value(
                    posterior_p=posterior_p,
                    expected_daily_demand=exp_lambda,
                    unit_price=r.unit_price,
                    gross_margin_pct=r.gross_margin_pct,
                    zero_streak_days=streak
                )
                
                candidate_rows.append({
                    "date": date_str,
                    "store_id": s_id,
                    "sku_id": r.sku_id,
                    "zero_streak": streak,
                    "system_stock": stock,
                    "unit_price": r.unit_price,
                    "gross_margin_pct": r.gross_margin_pct,
                    "expected_lambda": exp_lambda,
                    "posterior_p": posterior_p,
                    "daily_margin_at_risk": val_dict["daily_margin_at_risk"],
                    "days_saved": val_dict["days_saved"],
                    "expected_value_inr": val_dict["expected_value_inr"],
                    "peer_units_7d": peer_units_7d,
                    "sub_name": sub_name,
                    "sub_lift_pct": sub_lift_pct
                })
                
            if not candidate_rows:
                continue
                
            df_candidates = pd.DataFrame(candidate_rows)
            top_ranked = detector.value_ranker.rank_store_candidates(df_candidates)
            
            # Record alerts and simulate audit feedback
            for _, c_row in top_ranked.iterrows():
                key = (s_id, c_row["sku_id"])
                day_alerts_count += 1
                
                # Check ground truth
                is_true_phantom = 0
                ep_id = None
                onset_delay = -1
                if df_truth is not None:
                    t_info = truth_map.get((date_str, s_id, c_row["sku_id"]), {})
                    is_true_phantom = t_info.get("is_phantom", 0)
                    ep_id = t_info.get("episode_id")
                    onset_delay = t_info.get("days_since_onset", -1)
                    
                if is_true_phantom:
                    day_true_positives += 1
                    if resolve_phantoms and ep_id:
                        active_resolved_episodes.add(ep_id)
                        streaks[key] = 0
                else:
                    detector.cooldowns[key] = day_idx + detector.cooldown_days
                    
                # Generate explanation
                sub_label = detector.sku_names.get(c_row["sub_name"], c_row["sub_name"]) if c_row["sub_name"] else None
                reason = generate_plain_english_reason(
                    expected_daily_demand=c_row["expected_lambda"],
                    daily_revenue_inr=c_row["daily_margin_at_risk"],
                    zero_streak_days=int(c_row["zero_streak"]),
                    system_stock=int(c_row["system_stock"]),
                    peer_sales_7d=int(c_row["peer_units_7d"]),
                    substitute_name=sub_label,
                    substitute_lift_pct=float(c_row["sub_lift_pct"]),
                    confidence_pct=float(c_row["posterior_p"] * 100)
                )
                
                all_alerts.append({
                    "date": date_str,
                    "store_id": s_id,
                    "sku_id": c_row["sku_id"],
                    "sku_name": detector.sku_names.get(c_row["sku_id"], c_row["sku_id"]),
                    "rank": int(c_row["rank"]),
                    "model": detector.name,
                    "confidence": round(float(c_row["posterior_p"]), 4),
                    "zero_streak": int(c_row["zero_streak"]),
                    "system_stock": int(c_row["system_stock"]),
                    "expected_daily_sales": round(float(c_row["expected_lambda"]), 2),
                    "daily_margin_at_risk_inr": float(c_row["daily_margin_at_risk"]),
                    "expected_value_inr": float(c_row["expected_value_inr"]),
                    "days_saved": float(c_row["days_saved"]),
                    "peer_sales_7d": int(c_row["peer_units_7d"]),
                    "reason": reason,
                    "is_true_phantom": int(is_true_phantom),
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
