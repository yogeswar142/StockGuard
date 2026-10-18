"""
src/evaluation_metrics.py - Rigorous Evaluation Engine for StockGuard & Baselines.

Computes:
1. Raw Precision@20 per store-day.
2. Normalized Precision@20: raw_p@20 / min(20, active_phantoms_today).
3. Episode-level Recall within K days of onset (K = 3, 7, 14 days) and overall recall.
4. Mean Time-to-Detect (days from onset to first alert).
5. Precision broken down by SKU velocity band (Fast, Medium, Slow).
6. False positive breakdown by confuser type (demand shocks, post-promo lulls, slow mover zeros).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def compute_metrics(
    df_alerts: pd.DataFrame,
    df_truth: pd.DataFrame,
    df_episodes: pd.DataFrame,
    df_sku: pd.DataFrame,
    eval_start_date: str,
) -> Dict[str, Any]:
    """
    Computes all headline evaluation metrics for a given model's alerts dataframe.
    """
    if df_alerts.empty:
        return {
            "model": "None",
            "total_alerts": 0,
            "raw_precision_at_20": 0.0,
            "norm_precision_at_20": 0.0,
            "episode_recall_overall": 0.0,
            "episode_recall_3d": 0.0,
            "episode_recall_7d": 0.0,
            "episode_recall_14d": 0.0,
            "mean_time_to_detect": 0.0,
            "velocity_precision": {},
            "fp_breakdown": {}
        }
        
    model_name = df_alerts["model"].iloc[0]
    
    # Filter ground truth to evaluation period
    eval_truth = df_truth[df_truth["date"] >= eval_start_date].copy()
    
    # -------------------------------------------------------------
    # 1. RAW & NORMALIZED PRECISION@20 PER STORE-DAY
    # -------------------------------------------------------------
    # Count active phantoms per store-day in ground truth
    active_phantoms_per_day = (
        eval_truth.groupby(["date", "store_id"])["is_phantom"]
        .sum()
        .reset_index(name="active_phantoms")
    )
    
    daily_alerts_summary = (
        df_alerts.groupby(["date", "store_id"])
        .agg(
            alerts_count=("sku_id", "count"),
            true_positives=("is_true_phantom", "sum")
        )
        .reset_index()
    )
    
    merged_daily = pd.merge(daily_alerts_summary, active_phantoms_per_day, on=["date", "store_id"], how="left")
    merged_daily["active_phantoms"] = merged_daily["active_phantoms"].fillna(0)
    
    # Raw precision = true_positives / alerts_count
    total_alerts = df_alerts.shape[0]
    total_tp = df_alerts["is_true_phantom"].sum()
    raw_precision = total_tp / total_alerts if total_alerts > 0 else 0.0
    
    # Normalized precision = true_positives / min(20, active_phantoms)
    # Average across store-days where active_phantoms > 0
    norm_precisions = []
    for _, row in merged_daily.iterrows():
        cap = min(20, row["active_phantoms"])
        if cap > 0:
            norm_precisions.append(min(1.0, row["true_positives"] / cap))
        elif row["alerts_count"] > 0:
            norm_precisions.append(0.0)
            
    norm_precision = float(np.mean(norm_precisions)) if norm_precisions else 0.0
    
    # -------------------------------------------------------------
    # 2. EPISODE-LEVEL RECALL & TIME-TO-DETECT
    # -------------------------------------------------------------
    # Active phantom episodes that overlap with the evaluation period
    eval_episodes = df_episodes[
        (df_episodes["failure_type"] == "phantom") &
        (df_episodes["end_date"] >= eval_start_date)
    ].copy()
    total_eval_episodes = len(eval_episodes)
    
    # Find first detection date for each episode
    tp_alerts = df_alerts[df_alerts["is_true_phantom"] == 1].dropna(subset=["episode_id"])
    first_detections = (
        tp_alerts.groupby("episode_id")["days_since_onset"]
        .min()
        .to_dict()
    )
    
    caught_count = 0
    caught_3d = 0
    caught_7d = 0
    caught_14d = 0
    detection_delays = []
    
    for ep_id in eval_episodes["episode_id"]:
        if ep_id in first_detections:
            caught_count += 1
            delay = first_detections[ep_id]
            detection_delays.append(delay)
            if delay <= 3:
                caught_3d += 1
            if delay <= 7:
                caught_7d += 1
            if delay <= 14:
                caught_14d += 1
                
    recall_overall = caught_count / total_eval_episodes if total_eval_episodes > 0 else 0.0
    recall_3d = caught_3d / total_eval_episodes if total_eval_episodes > 0 else 0.0
    recall_7d = caught_7d / total_eval_episodes if total_eval_episodes > 0 else 0.0
    recall_14d = caught_14d / total_eval_episodes if total_eval_episodes > 0 else 0.0
    mean_ttd = float(np.mean(detection_delays)) if detection_delays else 0.0

    # -------------------------------------------------------------
    # 3. PRECISION BY SKU VELOCITY BAND
    # -------------------------------------------------------------
    df_alerts_sku = pd.merge(df_alerts, df_sku[["sku_id", "velocity_band"]], on="sku_id", how="left")
    vel_metrics = {}
    for band in ["Fast", "Medium", "Slow"]:
        band_alerts = df_alerts_sku[df_alerts_sku["velocity_band"] == band]
        b_count = len(band_alerts)
        b_tp = band_alerts["is_true_phantom"].sum()
        b_prec = b_tp / b_count if b_count > 0 else 0.0
        vel_metrics[band] = {
            "alerts": int(b_count),
            "true_positives": int(b_tp),
            "precision": round(float(b_prec), 4)
        }

    # -------------------------------------------------------------
    # 4. FALSE POSITIVE BREAKDOWN BY CONFUSER TYPE
    # -------------------------------------------------------------
    fp_alerts = df_alerts[df_alerts["is_true_phantom"] == 0].copy()
    total_fp = len(fp_alerts)
    fp_breakdown = {}
    
    if total_fp > 0:
        fp_merged = pd.merge(
            fp_alerts,
            eval_truth[["date", "store_id", "sku_id", "is_demand_shock", "is_post_promo_lull", "is_slow_mover_zero", "is_misplaced"]],
            on=["date", "store_id", "sku_id"],
            how="left"
        )
        fp_slow = int(fp_merged["is_slow_mover_zero"].sum())
        fp_shock = int(fp_merged["is_demand_shock"].sum())
        fp_lull = int(fp_merged["is_post_promo_lull"].sum())
        fp_misplaced = int(fp_merged["is_misplaced"].sum())
        fp_other = max(0, total_fp - fp_slow - fp_shock - fp_lull - fp_misplaced)
        
        fp_breakdown = {
            "total_false_positives": total_fp,
            "slow_mover_zeros": fp_slow,
            "slow_mover_share_pct": round(fp_slow / total_fp * 100, 1),
            "demand_shocks": fp_shock,
            "demand_shock_share_pct": round(fp_shock / total_fp * 100, 1),
            "post_promo_lulls": fp_lull,
            "post_promo_share_pct": round(fp_lull / total_fp * 100, 1),
            "misplaced_items": fp_misplaced,
            "misplaced_share_pct": round(fp_misplaced / total_fp * 100, 1),
            "other_noise": fp_other,
        }

    return {
        "model": model_name,
        "eval_start_date": eval_start_date,
        "total_alerts": total_alerts,
        "total_true_positives": int(total_tp),
        "raw_precision_at_20": round(float(raw_precision), 4),
        "norm_precision_at_20": round(float(norm_precision), 4),
        "total_eval_episodes": total_eval_episodes,
        "episodes_caught": caught_count,
        "episode_recall_overall": round(float(recall_overall), 4),
        "episode_recall_3d": round(float(recall_3d), 4),
        "episode_recall_7d": round(float(recall_7d), 4),
        "episode_recall_14d": round(float(recall_14d), 4),
        "mean_time_to_detect_days": round(float(mean_ttd), 2),
        "velocity_precision": vel_metrics,
        "fp_breakdown": fp_breakdown
    }
