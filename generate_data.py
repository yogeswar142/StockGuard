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

    if args.mode == "mvp":
        n_stores = 10
        n_skus = 150
        n_days = 180
        df_store, df_sku, df_promo, df_pos, df_truth = generate_synthetic_data(
            n_stores=n_stores,
            n_skus=n_skus,
            n_days=n_days,
            seed=args.seed,
            output_dir=args.output_dir
        )
    elif args.mode == "full":
        n_stores = 25
        n_skus = 800
        n_days = 540
        df_store, df_sku, df_promo, df_pos, df_truth = generate_synthetic_data(
            n_stores=n_stores,
            n_skus=n_skus,
            n_days=n_days,
            seed=args.seed,
            output_dir=args.output_dir
        )
    elif args.mode == "robustness":
        # Generates 3 robustness variants
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
                n_days=180,
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
    print(f"Stores: {meta['n_stores']} | SKUs: {meta['n_skus']} | Days: {meta['n_days']}")
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

    # Generate sanity plot
    generate_sanity_plot(df_pos, df_truth, df_sku, output_dir="results")
    print("\nStep 1 (Data Generation) complete!")


if __name__ == "__main__":
    main()
