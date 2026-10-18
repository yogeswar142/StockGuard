"""
run_stockguard.py - CLI tool to run the StockGuard Detection Pipeline.

Usage:
    python run_stockguard.py
    python run_stockguard.py --data_dir data --output_dir results
"""

import os
import sys
import argparse
import json
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.stockguard_detector import StockGuardDetector, run_stockguard_simulation
from src.evaluation_metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Run StockGuard Phantom Stock-Out Detector")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--eval_split", type=float, default=0.6, help="Fraction of timeline used for calibration")
    args = parser.parse_args()

    print("=" * 70)
    print(" STOCKGUARD PHANTOM STOCK-OUT DETECTOR - RUNNING PIPELINE")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Loading datasets...")
    df_pos = pd.read_csv(os.path.join(args.data_dir, "daily_sales_and_stock.csv.gz"))
    df_truth = pd.read_csv(os.path.join(args.data_dir, "ground_truth_labels.csv.gz"))
    df_episodes = pd.read_csv(os.path.join(args.data_dir, "ground_truth_episodes.csv"))
    df_sku = pd.read_csv(os.path.join(args.data_dir, "sku_master.csv"))

    unique_dates = sorted(df_pos["date"].unique())
    split_idx = int(len(unique_dates) * args.eval_split)
    eval_start_date = unique_dates[split_idx]

    print(f"   Total Timeline: {len(unique_dates)} days ({unique_dates[0]} to {unique_dates[-1]})")
    print(f"   Calibration Period (60%): {unique_dates[0]} to {unique_dates[split_idx-1]} ({split_idx} days)")
    print(f"   Out-of-Sample Test (40%): {eval_start_date} to {unique_dates[-1]} ({len(unique_dates) - split_idx} days)")

    print("\n2. Initializing StockGuard Core Engine...")
    detector = StockGuardDetector(
        cooldown_days=7,
        max_checks_per_store=20,
        prior_phantom_rate=0.028,
        min_posterior_threshold=0.35,
        overdispersion_alpha=0.30,
        audit_cost_inr=5.0,
        min_streak_for_peer_boost=4,
    )

    print("3. Executing day-by-day detection & audit feedback simulation...")
    alerts_df, meta = run_stockguard_simulation(
        detector=detector,
        df_pos=df_pos,
        df_truth=df_truth,
        df_sku=df_sku,
        eval_start_date=eval_start_date,
        resolve_phantoms=True
    )

    # Save alerts table
    alerts_path = os.path.join(args.output_dir, "stockguard_alerts.csv")
    alerts_df.to_csv(alerts_path, index=False)
    print(f"   Saved {len(alerts_df):,} alerts to: {alerts_path}")

    print("\n4. Computing Evaluation Metrics against Hidden Ground Truth...")
    metrics = compute_metrics(
        df_alerts=alerts_df,
        df_truth=df_truth,
        df_episodes=df_episodes,
        df_sku=df_sku,
        eval_start_date=eval_start_date
    )

    metrics_path = os.path.join(args.output_dir, "stockguard_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Display results
    print("\n" + "=" * 70)
    print(" STOCKGUARD PERFORMANCE SUMMARY (OUT-OF-SAMPLE TEST PERIOD)")
    print("=" * 70)
    print(f"Model: {metrics['model']}")
    print(f"  - Total Alerts Issued (Capped <=20/store-day): {metrics['total_alerts']:,}")
    print(f"  - True Phantoms Caught (Alert Level)       : {metrics['total_true_positives']:,}")
    print(f"  - Raw Precision@20                         : {metrics['raw_precision_at_20']*100:.2f}%")
    print(f"  - Normalized Precision@20                  : {metrics['norm_precision_at_20']*100:.2f}%")
    print(f"  - Episode Recall (Overall)                 : {metrics['episode_recall_overall']*100:.2f}% ({metrics['episodes_caught']}/{metrics['total_eval_episodes']})")
    print(f"  - Episode Recall within 3 Days of Onset    : {metrics['episode_recall_3d']*100:.2f}%")
    print(f"  - Episode Recall within 7 Days of Onset    : {metrics['episode_recall_7d']*100:.2f}%")
    print(f"  - Episode Recall within 14 Days of Onset   : {metrics['episode_recall_14d']*100:.2f}%")
    print(f"  - Mean Time to Detect (Caught Episodes)    : {metrics['mean_time_to_detect_days']:.1f} days")

    print("\n  - Precision by Velocity Band:")
    for band, v in metrics["velocity_precision"].items():
        print(f"      * {band:7s}: Prec = {v['precision']*100:5.2f}% ({v['true_positives']}/{v['alerts']})")

    print("\n  - False Positive Breakdown by Confuser Type:")
    fp = metrics["fp_breakdown"]
    if fp:
        print(f"      * Slow Mover Natural Zeros: {fp['slow_mover_share_pct']}% ({fp['slow_mover_zeros']:,} FPs)")
        print(f"      * Sustained Demand Shocks : {fp['demand_shock_share_pct']}% ({fp['demand_shocks']:,} FPs)")
        print(f"      * Post-Promo Lulls        : {fp['post_promo_share_pct']}% ({fp['post_promo_lulls']:,} FPs)")
        print(f"      * Misplaced Items         : {fp['misplaced_share_pct']}% ({fp['misplaced_items']:,} FPs)")

    print("\n--- SAMPLE TOP-RANKED ALERTS & PLAIN-ENGLISH REASONS ---")
    sample_cols = ["date", "store_id", "sku_name", "confidence", "expected_value_inr", "reason"]
    print(alerts_df[sample_cols].head(5).to_string(index=False))

    print("\nStep 3 (StockGuard Core) complete!")


if __name__ == "__main__":
    main()
