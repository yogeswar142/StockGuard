"""
generate_data.py - CLI tool to generate synthetic datasets for StockGuard.

Usage:
    python generate_data.py --mode mvp          # 10 stores x 150 SKUs x 180 days (Round 1 MVP)
    python generate_data.py --mode full         # 25 stores x 800 SKUs x 540 days (Full 18-month scale)
    python generate_data.py --mode robustness   # Generates robustness test suites
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.generator import generate_synthetic_data


def generate_sanity_plot(df_pos: pd.DataFrame, df_truth: pd.DataFrame, df_sku: pd.DataFrame, output_dir: str = "results"):
    """
    Plots a sample SKU-store trajectory showing sales vs system stock vs ground truth phantom episode.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge a single store-SKU trajectory with a phantom episode
    phantoms = df_truth[df_truth["is_phantom"] == 1]
    if len(phantoms) > 0:
        sample_row = phantoms.iloc[len(phantoms) // 2]
        s_id = sample_row["store_id"]
        sku_id = sample_row["sku_id"]
    else:
        s_id = df_pos["store_id"].iloc[0]
        sku_id = df_pos["sku_id"].iloc[0]
        
    pos_traj = df_pos[(df_pos["store_id"] == s_id) & (df_pos["sku_id"] == sku_id)].copy().sort_values("date")
    truth_traj = df_truth[(df_truth["store_id"] == s_id) & (df_truth["sku_id"] == sku_id)].copy().sort_values("date")
    
    merged = pd.merge(pos_traj, truth_traj[["date", "is_phantom", "physical_stock", "is_true_stockout"]], on="date")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    
    sku_info = df_sku[df_sku["sku_id"] == sku_id].iloc[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # Top plot: Sales vs System Stock vs Physical Stock
    ax1.plot(merged["date_dt"], merged["system_stock"], label="System Stock (POS Record)", color="#1f77b4", linewidth=2.5)
    ax1.plot(merged["date_dt"], merged["physical_stock"], label="True Physical Shelf Stock", color="#2ca02c", linestyle="--", linewidth=2)
    ax1.bar(merged["date_dt"], merged["sales"], label="Recorded Daily Sales", color="#ff7f0e", alpha=0.7, width=0.8)
    
    # Highlight phantom episode periods
    phantom_mask = merged["is_phantom"] == 1
    if phantom_mask.any():
        ax1.fill_between(
            merged["date_dt"],
            0,
            merged["system_stock"].max() * 1.15,
            where=phantom_mask,
            color="red",
            alpha=0.18,
            label="Hidden Phantom Stock-Out Episode"
        )
        
    ax1.set_title(
        f"StockGuard Generator Sanity Check: Store {s_id} | {sku_info['sku_name']} ({sku_info['velocity_band']} mover)\n"
        f"Notice: During phantom episode (red band), physical stock drops to 0, sales freeze to 0, while system stock remains frozen!",
        fontsize=12, pad=12, weight="bold"
    )
    ax1.set_ylabel("Units", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Bottom plot: Promo and stock-out flags
    ax2.step(merged["date_dt"], merged["on_promo"], label="On Promotion (Discount Lift)", color="#9467bd", where="mid")
    ax2.step(merged["date_dt"], merged["is_true_stockout"], label="True Stock-Out (Stock=0)", color="#8c564b", linestyle=":", where="mid")
    ax2.set_ylabel("Status Flag", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["No", "Yes"])
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "generator_sanity_plot.png")
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Sanity plot saved to: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="StockGuard Synthetic Data Generator")
    parser.add_argument("--mode", type=str, choices=["mvp", "full", "robustness"], default="mvp",
                        help="Mode: mvp (10 stores x 150 SKUs x 180 days), full (25x800x540), or robustness")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print(f" STOCKGUARD SYNTHETIC DATA GENERATOR - Mode: {args.mode.upper()}")
    print("=" * 70)

def compute_detailed_diagnostics(df_pos: pd.DataFrame, df_truth: pd.DataFrame, df_sku: pd.DataFrame, df_episodes: pd.DataFrame):
    """
    Computes and prints required statistical diagnostics:
    - lambda per velocity band
    - phantom share per band
    - duration distribution
    - non-phantom zero streak frequencies (3+, 5+, 7+ days) with stock > 0
    - verification that all phantom rows have system_stock > 0
    """
    merged = pd.merge(df_pos, df_sku[["sku_id", "velocity_band", "base_demand_rate"]], on="sku_id")
    merged = pd.merge(merged, df_truth[["date", "store_id", "sku_id", "is_phantom", "is_true_stockout", "is_demand_shock", "is_post_promo_lull", "is_slow_mover_zero"]], on=["date", "store_id", "sku_id"])
    
    print("\n" + "=" * 70)
    print(" STATISTICAL DIAGNOSTICS & DATASET SANITY CHECKS")
    print("=" * 70)
    
    # 1. Lambda & Sales per velocity band
    print("\n[1] DEMAND VELOCITY BY BAND (Empirical Sales & Inherent Lambda):")
    for band in ["Fast", "Medium", "Slow"]:
        sub = merged[merged["velocity_band"] == band]
        # In normal in-stock conditions
        normal_sub = sub[(sub["is_phantom"] == 0) & (sub["system_stock"] > 0)]
        mean_sales = normal_sub["sales"].mean()
        base_lambda = sub["base_demand_rate"].mean()
        print(f"  - {band:7s} Movers: Base Lambda = {base_lambda:.2f}/day | In-Stock Realized Mean Sales = {mean_sales:.2f}/day (Total rows: {len(sub):,})")

    # 2. Phantom share per band
    print("\n[2] PHANTOM PREVALENCE BY VELOCITY BAND:")
    for band in ["Fast", "Medium", "Slow"]:
        sub = merged[merged["velocity_band"] == band]
        p_share = sub["is_phantom"].mean() * 100
        p_count = sub["is_phantom"].sum()
        print(f"  - {band:7s} Movers: Phantom Row Share = {p_share:.2f}% ({p_count:,} active phantom SKU-days)")

    # 3. Episode duration distribution
    phantom_eps = df_episodes[df_episodes["failure_type"] == "phantom"]
    durations = phantom_eps["duration_days"]
    print("\n[3] PHANTOM EPISODE DURATION DISTRIBUTION (Days):")
    print(f"  - Count : {len(durations)}")
    print(f"  - Mean  : {durations.mean():.1f} days")
    print(f"  - Min   : {durations.min()} days")
    print(f"  - 25%   : {durations.quantile(0.25):.0f} days")
    print(f"  - 50%   : {durations.quantile(0.50):.0f} days (Median)")
    print(f"  - 75%   : {durations.quantile(0.75):.0f} days")
    print(f"  - 95%   : {durations.quantile(0.95):.0f} days")
    print(f"  - Max   : {durations.max()} days")

    # 4. Zero streak frequencies for non-phantom rows with stock > 0
    print("\n[4] NATURAL ZERO STREAK FREQUENCIES (Non-Phantom Rows with System Stock > 0):")
    # Compute streak lengths for each store-SKU trajectory
    merged = merged.sort_values(["store_id", "sku_id", "date"]).copy()
    
    # Calculate consecutive zero sales days
    is_zero = (merged["sales"] == 0).astype(int)
    # Consecutive zero streak count
    streak_counts = []
    current_streak = 0
    prev_key = None
    
    # Vectorized / fast series calculation of zero streaks
    merged["is_zero_sale"] = (merged["sales"] == 0)
    # Group cumulative count of zero sales
    merged["streak_id"] = (~merged["is_zero_sale"]).groupby([merged["store_id"], merged["sku_id"]]).cumsum()
    merged["zero_streak"] = merged.groupby(["store_id", "sku_id", "streak_id"])["is_zero_sale"].cumsum()
    
    non_phantom_in_stock = merged[(merged["is_phantom"] == 0) & (merged["system_stock"] > 0)]
    for band in ["Fast", "Medium", "Slow"]:
        band_sub = non_phantom_in_stock[non_phantom_in_stock["velocity_band"] == band]
        total_band_rows = len(band_sub)
        n_gte_3 = (band_sub["zero_streak"] >= 3).sum()
        n_gte_5 = (band_sub["zero_streak"] >= 5).sum()
        n_gte_7 = (band_sub["zero_streak"] >= 7).sum()
        print(f"  - {band:7s} Movers: 3+ Zeros: {n_gte_3/total_band_rows*100:5.2f}% ({n_gte_3:,}) | "
              f"5+ Zeros: {n_gte_5/total_band_rows*100:5.2f}% ({n_gte_5:,}) | "
              f"7+ Zeros: {n_gte_7/total_band_rows*100:5.2f}% ({n_gte_7:,})")

    # 5. Check all phantom rows have system_stock > 0
    print("\n[5] INTEGRITY CHECK: All Phantom Rows Have System Stock > 0?")
    phantom_rows = merged[merged["is_phantom"] == 1]
    invalid_phantom_count = (phantom_rows["system_stock"] <= 0).sum()
    min_stock_phantom = phantom_rows["system_stock"].min() if len(phantom_rows) > 0 else 0
    print(f"  - Phantom rows total: {len(phantom_rows):,}")
    print(f"  - Phantom rows with system_stock <= 0: {invalid_phantom_count}")
    print(f"  - Minimum system stock during phantom episode: {min_stock_phantom}")
    assert invalid_phantom_count == 0, "ERROR: Found phantom rows with system_stock <= 0!"
    print("  -> PASSED: 100% of phantom episodes satisfy system_stock > 0.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="StockGuard Synthetic Data Generator")
    parser.add_argument("--mode", type=str, choices=["mvp", "full", "robustness"], default="mvp",
                        help="Mode: mvp (10 stores x 150 SKUs x 540 saved days), full (25x800x540), or robustness")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print(f" STOCKGUARD SYNTHETIC DATA GENERATOR - Mode: {args.mode.upper()}")
    print("=" * 70)

    if args.mode == "mvp":
        n_stores = 10
        n_skus = 150
        n_days = 540
        burn_in = 60
        df_store, df_sku, df_promo, df_pos, df_truth = generate_synthetic_data(
            n_stores=n_stores,
            n_skus=n_skus,
            n_days=n_days,
            burn_in_days=burn_in,
            seed=args.seed,
            output_dir=args.output_dir
        )
    elif args.mode == "full":
        n_stores = 25
        n_skus = 800
        n_days = 540
        burn_in = 60
        df_store, df_sku, df_promo, df_pos, df_truth = generate_synthetic_data(
            n_stores=n_stores,
            n_skus=n_skus,
            n_days=n_days,
            burn_in_days=burn_in,
            seed=args.seed,
            output_dir=args.output_dir
        )
    elif args.mode == "robustness":
        variants = [
            ("noisy_demand", {"overdispersion_alpha": 0.65}),
            ("heavy_promos", {"promo_frequency": 0.15}),
            ("more_slow_movers", {"slow_mover_share": 0.80, "medium_mover_share": 0.15, "fast_mover_share": 0.05}),
        ]
        for v_name, v_params in variants:
            v_dir = os.path.join(args.output_dir, "robustness", v_name)
            print(f"\nGenerating Robustness Variant: {v_name} into {v_dir}...")
            generate_synthetic_data(
                n_stores=10,
                n_skus=150,
                n_days=540,
                burn_in_days=60,
                seed=args.seed + 10,
                output_dir=v_dir,
                **v_params
            )
        print("\nAll robustness variants generated successfully.")
        return

    # Print summary & sample rows
    with open(os.path.join(args.output_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    print("\n--- DATASET GENERATION SUMMARY ---")
    print(f"Stores: {meta['n_stores']} | SKUs: {meta['n_skus']} | Saved Days: {meta['n_days']} (Burn-in: 60 days discarded)")
    print(f"Total SKU-Store-Days: {meta['total_sku_days']:,}")
    print(f"Total Injected Phantom Episodes: {meta['total_phantom_episodes']}")
    print(f"Total Misplaced Episodes: {meta['total_misplaced_episodes']}")
    print(f"Total Delisted Episodes: {meta['total_delisted_episodes']}")
    print(f"Active Phantom Row Share: {meta['phantom_row_share']*100:.2f}% (Target: ~3%)")
    print(f"True Stock-Out Row Share: {meta['true_stockout_row_share']*100:.2f}%")
    print(f"Misplaced Row Share: {meta['misplaced_row_share']*100:.2f}%")

    print("\n--- SAMPLE POS RECORDS (Visible to Model) ---")
    print(df_pos.head(6).to_string(index=False))

    print("\n--- SAMPLE GROUND TRUTH LABELS (Hidden for Evaluation) ---")
    print(df_truth[df_truth["is_phantom"] == 1].head(5).to_string(index=False))

    # Read ground truth episodes table for duration analytics
    df_episodes = pd.read_csv(os.path.join(args.output_dir, "ground_truth_episodes.csv"))

    # Compute and display required diagnostics
    compute_detailed_diagnostics(df_pos, df_truth, df_sku, df_episodes)

    # Generate sanity plot
    generate_sanity_plot(df_pos, df_truth, df_sku, output_dir="results")
    print("Step 1 (Data Generation) complete!")


if __name__ == "__main__":
    main()

