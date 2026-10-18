"""
src/run_baselines.py - Runs and evaluates Baseline A and Baseline B on the synthetic dataset.
"""

import os
import sys
import json
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.baseline_models import FixedNZeroDaysBaseline, StreakLengthBaseline, run_baseline_simulation
from src.evaluation_metrics import compute_metrics


def main():
    print("=" * 70)
    print(" STOCKGUARD STEP 2: RUNNING & EVALUATING BASELINE MODELS")
    print("=" * 70)
    
    data_dir = "data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load synthetic data
    print("Loading datasets...")
    df_pos = pd.read_csv(os.path.join(data_dir, "daily_sales_and_stock.csv.gz"))
    df_truth = pd.read_csv(os.path.join(data_dir, "ground_truth_labels.csv.gz"))
    df_episodes = pd.read_csv(os.path.join(data_dir, "ground_truth_episodes.csv"))
    df_sku = pd.read_csv(os.path.join(data_dir, "sku_master.csv"))
    
    # Time split: first 60% calibration / warm-up, last 40% out-of-sample evaluation
    unique_dates = sorted(df_pos["date"].unique())
    split_idx = int(len(unique_dates) * 0.6)
    eval_start_date = unique_dates[split_idx]
    
    print(f"Total dates: {len(unique_dates)} ({unique_dates[0]} to {unique_dates[-1]})")
    print(f"Calibration period (first 60%): {unique_dates[0]} to {unique_dates[split_idx-1]} ({split_idx} days)")
    print(f"Evaluation period  (last 40%): {eval_start_date} to {unique_dates[-1]} ({len(unique_dates) - split_idx} days)")
    
    # -------------------------------------------------------------
    # 1. RUN BASELINE A: Fixed 5-Zero-Days Rule
    # -------------------------------------------------------------
    print("\n[Running Baseline A: Fixed 5-Zero-Days Rule (Top-20 Budget + 7-Day Cooldown)]...")
    baseline_a = FixedNZeroDaysBaseline(n_zero_days=5, cooldown_days=7, max_checks_per_store=20)
    alerts_a, meta_a = run_baseline_simulation(
        baseline_a, df_pos, df_truth, eval_start_date=eval_start_date, resolve_phantoms=True
    )
    metrics_a = compute_metrics(alerts_a, df_truth, df_episodes, df_sku, eval_start_date)
    
    # -------------------------------------------------------------
    # 2. RUN BASELINE B: Streak-Length Ranking
    # -------------------------------------------------------------
    print("\n[Running Baseline B: Streak-Length Ranking (Top-20 Budget + 7-Day Cooldown)]...")
    baseline_b = StreakLengthBaseline(cooldown_days=7, max_checks_per_store=20)
    alerts_b, meta_b = run_baseline_simulation(
        baseline_b, df_pos, df_truth, eval_start_date=eval_start_date, resolve_phantoms=True
    )
    metrics_b = compute_metrics(alerts_b, df_truth, df_episodes, df_sku, eval_start_date)
    
    # -------------------------------------------------------------
    # DISPLAY METRICS COMPARISON
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" BASELINE MODELS EVALUATION SUMMARY (OUT-OF-SAMPLE TEST PERIOD)")
    print("=" * 70)
    
    for m in [metrics_a, metrics_b]:
        print(f"\nModel: {m['model']}")
        print(f"  - Total Alerts Issued (Capped <=20/store-day): {m['total_alerts']:,}")
        print(f"  - True Phantoms Caught (Alert Level)       : {m['total_true_positives']:,}")
        print(f"  - Raw Precision@20                         : {m['raw_precision_at_20']*100:.2f}%")
        print(f"  - Normalized Precision@20                  : {m['norm_precision_at_20']*100:.2f}%")
        print(f"  - Episode Recall (Overall)                 : {m['episode_recall_overall']*100:.2f}% ({m['episodes_caught']}/{m['total_eval_episodes']})")
        print(f"  - Episode Recall within 3 Days of Onset    : {m['episode_recall_3d']*100:.2f}%")
        print(f"  - Episode Recall within 7 Days of Onset    : {m['episode_recall_7d']*100:.2f}%")
        print(f"  - Episode Recall within 14 Days of Onset   : {m['episode_recall_14d']*100:.2f}%")
        print(f"  - Mean Time to Detect (Caught Episodes)    : {m['mean_time_to_detect_days']:.1f} days")
        print("  - Precision by Velocity Band:")
        for band, v in m['velocity_precision'].items():
            print(f"      * {band:7s}: Prec = {v['precision']*100:5.2f}% ({v['true_positives']}/{v['alerts']})")
        print("  - False Positive Breakdown:")
        fp = m['fp_breakdown']
        if fp:
            print(f"      * Slow Mover Natural Zeros: {fp['slow_mover_share_pct']}% ({fp['slow_mover_zeros']:,} FPs)")
            print(f"      * Sustained Demand Shocks : {fp['demand_shock_share_pct']}% ({fp['demand_shocks']:,} FPs)")
            print(f"      * Post-Promo Lulls        : {fp['post_promo_share_pct']}% ({fp['post_promo_lulls']:,} FPs)")
            print(f"      * Misplaced Items         : {fp['misplaced_share_pct']}% ({fp['misplaced_items']:,} FPs)")
    
    # Save metrics to results
    baseline_results = {
        "baseline_a_fixed_5d": metrics_a,
        "baseline_b_streak_len": metrics_b
    }
    with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\nBaseline results written to: {os.path.join(results_dir, 'baseline_metrics.json')}")


if __name__ == "__main__":
    main()
