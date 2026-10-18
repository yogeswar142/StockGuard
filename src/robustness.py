"""
src/robustness.py - Robustness Grid Runner for StockGuard.

Evaluates StockGuard performance across stressed operational scenarios:
1. Baseline: Default parameters (Seed 42, alpha=0.30, 60% slow movers, promo_freq=0.08)
2. Alternative Seed: Seed 101 (tests random episode injection stability)
3. Noisier Demand: alpha=0.60 (heavy demand clustering / overdispersion)
4. Heavier Promos: promo_freq=0.16 (2x promo density, more forward-buying dips)
5. More Slow Movers: 75% slow movers (longer retail tail)

Outputs structured comparison table for report.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.generator import generate_synthetic_data
from src.stockguard_detector import StockGuardDetector, run_stockguard_simulation
from src.baseline_models import FixedNZeroDaysBaseline, run_baseline_simulation
from src.evaluation_metrics import compute_metrics


def run_robustness_scenario(
    name: str,
    seed: int = 42,
    overdispersion_alpha: float = 0.30,
    promo_frequency: float = 0.08,
    velocity_mix: tuple = (0.10, 0.30, 0.60),
    n_stores: int = 10,
    n_skus: int = 150,
    n_days: int = 360,
    burn_in_days: int = 30,
) -> Dict[str, Any]:
    """
    Generates scenario data and evaluates StockGuard vs Baseline A out-of-sample.
    """
    print(f"\n--- Running Robustness Scenario: {name} ---")
    
    temp_dir = f"data_robustness_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Generate scenario data
    df_store, df_sku, df_promo, df_pos, df_truth = generate_synthetic_data(
        n_stores=n_stores,
        n_skus=n_skus,
        n_days=n_days,
        burn_in_days=burn_in_days,
        seed=seed,
        phantom_onset_prob=0.001429,
        mean_phantom_duration=21,
        overdispersion_alpha=overdispersion_alpha,
        promo_frequency=promo_frequency,
        slow_mover_share=velocity_mix[2],
        medium_mover_share=velocity_mix[1],
        fast_mover_share=velocity_mix[0],
        output_dir=temp_dir,
    )
    df_episodes = pd.read_csv(os.path.join(temp_dir, "ground_truth_episodes.csv"))
    
    unique_dates = sorted(df_pos["date"].unique())
    split_idx = int(len(unique_dates) * 0.6)
    eval_start_date = unique_dates[split_idx]
    days_in_eval = len(unique_dates) - split_idx
    
    # 2. Run StockGuard
    detector = StockGuardDetector(
        cooldown_days=7,
        max_checks_per_store=20,
        prior_phantom_rate=0.028,
        min_posterior_threshold=0.35,
        overdispersion_alpha=overdispersion_alpha,
        audit_cost_inr=5.0,
        min_streak_for_peer_boost=4,
    )
    sg_alerts, _ = run_stockguard_simulation(
        detector=detector,
        df_pos=df_pos,
        df_truth=df_truth,
        df_sku=df_sku,
        eval_start_date=eval_start_date,
        resolve_phantoms=True,
    )
    
    # 3. Compute Metrics
    sg_metrics = compute_metrics(sg_alerts, df_truth, df_episodes, df_sku, eval_start_date)
    
    # 4. Episode value & days saved
    eval_eps = df_episodes[
        (df_episodes["failure_type"] == "phantom") &
        (df_episodes["end_date"] >= eval_start_date)
    ].copy()
    eval_eps = eval_eps.merge(
        df_sku[["sku_id", "unit_price_inr", "gross_margin_pct", "base_demand_rate"]],
        on="sku_id",
        how="left"
    )
    
    tp_alerts = sg_alerts[sg_alerts["is_true_phantom"] == 1].dropna(subset=["episode_id"])
    first_alerts = tp_alerts.groupby("episode_id")["days_since_onset"].min().to_dict()
    
    days_saved_list = []
    margin_saved_list = []
    for r in eval_eps.itertuples():
        actual_dur = r.duration_days
        daily_margin = r.unit_price_inr * r.gross_margin_pct * r.base_demand_rate
        if r.episode_id in first_alerts:
            d_alert = first_alerts[r.episode_id]
            d_saved = max(0, actual_dur - (d_alert + 1))
        else:
            d_saved = 0
        days_saved_list.append(d_saved)
        margin_saved_list.append(d_saved * daily_margin)
        
    eval_eps["days_saved"] = days_saved_list
    eval_eps["margin_saved"] = margin_saved_list
    
    total_phantom_value = (eval_eps["duration_days"] * eval_eps["unit_price_inr"] * eval_eps["gross_margin_pct"] * eval_eps["base_demand_rate"]).sum()
    caught_value = eval_eps[eval_eps["days_saved"] > 0]["duration_days"] * eval_eps[eval_eps["days_saved"] > 0]["unit_price_inr"] * eval_eps[eval_eps["days_saved"] > 0]["gross_margin_pct"] * eval_eps[eval_eps["days_saved"] > 0]["base_demand_rate"]
    vw_recall = caught_value.sum() / max(1.0, total_phantom_value)
    
    scale = 365.0 / max(1, days_in_eval)
    gross_annual = eval_eps["margin_saved"].sum() * scale
    net_recovery = gross_annual * 0.60 - (len(sg_alerts) * 5.0 * scale)
    
    result = {
        "scenario": name,
        "total_alerts": len(sg_alerts),
        "true_positives": sg_metrics["total_true_positives"],
        "raw_precision_pct": round(sg_metrics["raw_precision_at_20"] * 100, 2),
        "norm_precision_pct": round(sg_metrics["norm_precision_at_20"] * 100, 2),
        "recall_overall_pct": round(sg_metrics["episode_recall_overall"] * 100, 2),
        "recall_7d_pct": round(sg_metrics["episode_recall_7d"] * 100, 2),
        "mean_ttd_days": round(sg_metrics["mean_time_to_detect_days"], 1),
        "value_weighted_recall_pct": round(vw_recall * 100, 2),
        "net_annual_recovery_inr": round(net_recovery, 0),
    }
    print(f"  Result: Raw Prec={result['raw_precision_pct']}% | Recall@7d={result['recall_7d_pct']}% | VWR={result['value_weighted_recall_pct']}% | Net={result['net_annual_recovery_inr']:,.0f}")
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
        
    return result


def run_full_robustness_grid() -> List[Dict[str, Any]]:
    """Runs all 5 predefined stress scenarios."""
    scenarios = [
        ("Base Scenario (Seed 42)", {}),
        ("Alternative Seed (Seed 101)", {"seed": 101}),
        ("Noisier Demand (alpha=0.60)", {"overdispersion_alpha": 0.60}),
        ("Heavier Promos (freq=0.16)", {"promo_frequency": 0.16}),
        ("Longer Slow Tail (75% Slow)", {"velocity_mix": (0.08, 0.17, 0.75)}),
    ]
    results = []
    for name, kwargs in scenarios:
        res = run_robustness_scenario(name, **kwargs)
        results.append(res)
    return results
