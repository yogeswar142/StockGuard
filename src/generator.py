"""
src/generator.py - Synthetic Data Generator for StockGuard.

Simulates realistic grocery retail POS and inventory data:
- Store master (format, size, tier, traffic)
- SKU master (fast/medium/slow velocity mix, category, substitutes, margins)
- Promo calendar & Indian festive calendar
- Day-by-day inventory dynamics: physical vs system stock, replenishment lag
- Injected phantom stock-outs (~3% prevalence via hazard rate ~0.00143/day)
- Realistic confusers: true stockouts (stock=0), slow-mover zero runs, post-promo dips
- Stretch cases: misplaced inventory, delisted lines
- Hidden ground truth saved completely separately from POS data.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional


def generate_synthetic_data(
    n_stores: int = 10,
    n_skus: int = 150,
    n_days: int = 540,
    burn_in_days: int = 60,
    start_date: str = "2025-01-01",
    seed: int = 42,
    phantom_onset_prob: float = 0.001429,  # ~0.03 prevalence / 21 days mean duration
    mean_phantom_duration: int = 21,
    overdispersion_alpha: float = 0.3,
    slow_mover_share: float = 0.60,
    medium_mover_share: float = 0.30,
    fast_mover_share: float = 0.10,
    promo_frequency: float = 0.06,
    output_dir: str = "data",
    compress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic store inventory, POS sales, and hidden ground-truth labels.
    Includes a 60-day burn-in discarded before saving, 540 saved days (18 months),
    and hard confusers with stock > 0 (demand shock, post-promo lull, slow mover zeros).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    total_sim_days = burn_in_days + n_days
    
    # ---------------------------------------------------------
    # 1. STORE MASTER
    # ---------------------------------------------------------
    store_names_tier1 = [
        ("Indiranagar", "Bengaluru", "Tier-1"),
        ("Koramangala", "Bengaluru", "Tier-1"),
        ("Whitefield", "Bengaluru", "Tier-1"),
        ("Bandra West", "Mumbai", "Tier-1"),
        ("Powai", "Mumbai", "Tier-1"),
        ("Andheri East", "Mumbai", "Tier-1"),
        ("South Extension", "Delhi-NCR", "Tier-1"),
        ("Cyber City", "Gurugram", "Tier-1"),
        ("Gachibowli", "Hyderabad", "Tier-1"),
        ("Banjara Hills", "Hyderabad", "Tier-1"),
        ("Koregaon Park", "Pune", "Tier-1"),
        ("Anna Nagar", "Chennai", "Tier-1"),
        ("Salt Lake", "Kolkata", "Tier-1"),
    ]
    store_names_tier2 = [
        ("Gomti Nagar", "Lucknow", "Tier-2"),
        ("Vaishali Nagar", "Jaipur", "Tier-2"),
        ("Satellite", "Ahmedabad", "Tier-2"),
        ("Kalyani Nagar", "Pune", "Tier-2"),
        ("RS Puram", "Coimbatore", "Tier-2"),
        ("Palarivattom", "Kochi", "Tier-2"),
        ("Khadki", "Chandigarh", "Tier-2"),
        ("Kakkanad", "Kochi", "Tier-2"),
        ("Bistupur", "Jamshedpur", "Tier-2"),
        ("Vijay Nagar", "Indore", "Tier-2"),
        ("Navrangpura", "Ahmedabad", "Tier-2"),
        ("Alkapuri", "Vadodara", "Tier-2"),
    ]
    
    all_store_locations = store_names_tier1 + store_names_tier2
    if n_stores > len(all_store_locations):
        for i in range(len(all_store_locations), n_stores):
            all_store_locations.append((f"Branch {i+1}", f"City {i%5+1}", "Tier-2"))
    
    selected_stores = all_store_locations[:n_stores]
    store_records = []
    formats = ["Supermarket", "Hypermarket", "Express"]
    
    for i, (name, city, tier) in enumerate(selected_stores):
        store_id = f"STORE_{i+1:02d}"
        s_format = np.random.choice(formats, p=[0.55, 0.25, 0.20])
        if s_format == "Hypermarket":
            sqft = int(np.random.uniform(15000, 30000))
            traffic_mult = np.random.uniform(1.2, 1.5)
        elif s_format == "Supermarket":
            sqft = int(np.random.uniform(6000, 14000))
            traffic_mult = np.random.uniform(0.9, 1.2)
        else:
            sqft = int(np.random.uniform(2500, 5000))
            traffic_mult = np.random.uniform(0.7, 0.9)
            
        if tier == "Tier-2":
            traffic_mult *= 0.88
            
        store_records.append({
            "store_id": store_id,
            "store_name": f"{name} ({city})",
            "city": city,
            "city_tier": tier,
            "format": s_format,
            "size_sqft": sqft,
            "traffic_multiplier": round(traffic_mult, 3)
        })
    df_store = pd.DataFrame(store_records)
    
    # ---------------------------------------------------------
    # 2. SKU MASTER
    # ---------------------------------------------------------
    categories = [
        ("Staples", ["Atta", "Rice", "Edible Oil", "Pulses", "Sugar & Salt"]),
        ("Dairy & Breakfast", ["Milk", "Butter", "Cheese", "Cereal", "Bread"]),
        ("Beverages", ["Tea", "Coffee", "Fruit Juices", "Soft Drinks"]),
        ("Snacks & Packaged", ["Biscuits", "Instant Noodles", "Namkeen", "Chocolates"]),
        ("Personal Care", ["Soaps", "Shampoo", "Toothpaste", "Deodorant"]),
        ("Home Care", ["Detergent Powder", "Dishwash Gel", "Floor Cleaner", "Mosquito Repellent"])
    ]
    
    sku_records = []
    n_fast = int(n_skus * fast_mover_share)
    n_med = int(n_skus * medium_mover_share)
    n_slow = n_skus - n_fast - n_med
    velocity_bands = ["Fast"] * n_fast + ["Medium"] * n_med + ["Slow"] * n_slow
    np.random.shuffle(velocity_bands)
    
    sku_counter = 1
    for cat_name, subcats in categories:
        skus_in_cat = n_skus // len(categories)
        if cat_name == categories[-1][0]:
            skus_in_cat = n_skus - len(sku_records)
            
        for _ in range(skus_in_cat):
            subcat = np.random.choice(subcats)
            vel = velocity_bands[len(sku_records)]
            sku_id = f"SKU_{sku_counter:03d}"
            sku_counter += 1
            
            # Base daily demand rate
            if vel == "Fast":
                base_demand = np.random.uniform(3.0, 7.5)
                unit_price = np.random.choice([40, 65, 95, 120, 180, 240, 320])
                margin_pct = np.random.uniform(0.18, 0.28)
                reorder_pt = int(base_demand * np.random.uniform(4, 6))
                reorder_qty = int(base_demand * np.random.uniform(7, 12))
            elif vel == "Medium":
                base_demand = np.random.uniform(0.8, 2.5)
                unit_price = np.random.choice([75, 130, 210, 350, 480])
                margin_pct = np.random.uniform(0.22, 0.35)
                reorder_pt = int(base_demand * np.random.uniform(5, 8)) + 1
                reorder_qty = int(base_demand * np.random.uniform(10, 16)) + 2
            else:  # Slow
                base_demand = np.random.uniform(0.04, 0.45)
                unit_price = np.random.choice([150, 290, 450, 650, 950, 1400])
                margin_pct = np.random.uniform(0.28, 0.42)
                reorder_pt = max(2, int(base_demand * 10))
                reorder_qty = max(4, int(base_demand * 25))
                
            lead_time = int(np.random.choice([2, 3, 4]))
            
            sku_records.append({
                "sku_id": sku_id,
                "sku_name": f"{subcat} Item {sku_counter-1}",
                "category": cat_name,
                "sub_category": subcat,
                "velocity_band": vel,
                "base_demand_rate": round(base_demand, 3),
                "unit_price_inr": float(unit_price),
                "gross_margin_pct": round(margin_pct, 3),
                "reorder_point": reorder_pt,
                "reorder_quantity": reorder_qty,
                "lead_time_days": lead_time,
                "substitute_sku_id": None
            })
            
    df_sku = pd.DataFrame(sku_records)
    
    # Assign sibling substitute within same sub-category
    for subcat, grp in df_sku.groupby("sub_category"):
        sku_list = grp["sku_id"].tolist()
        for s_idx in sku_list:
            peers = [x for x in sku_list if x != s_idx]
            sub_target = np.random.choice(peers) if peers else None
            df_sku.loc[df_sku["sku_id"] == s_idx, "substitute_sku_id"] = sub_target

    # ---------------------------------------------------------
    # 3. CALENDAR & PROMO SCHEDULE (Full timeline with burn-in)
    # ---------------------------------------------------------
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    sim_date_list = [base_dt + timedelta(days=d) for d in range(total_sim_days)]
    
    # Indian Festive Calendar multi-year dates
    festive_calendar = {
        "01-26": 1.55,  # Republic Day Big Sale
        "01-25": 1.40,
        "01-24": 1.35,
        "03-14": 1.45,  # Holi week
        "03-15": 1.50,
        "08-15": 1.50,  # Independence Day Sale
        "08-14": 1.35,
        "10-20": 1.65,  # Dussehra / Diwali build-up
        "10-21": 1.70,
        "10-22": 1.85,
        "11-01": 1.60,  # Diwali peak
        "11-02": 1.75,
        "12-25": 1.30,  # Christmas / Year end
        "12-31": 1.40,
    }
    
    # Pre-select Demand Shock SKUs (5% of SKUs undergo genuine prolonged drop)
    demand_shock_skus = set(df_sku.sample(n=max(2, int(n_skus * 0.05)), random_state=seed)["sku_id"])
    
    # Generate promo table: (date, store_id, sku_id) -> promo_discount, promo_lift
    promo_rows = []
    promo_end_dates = set()  # (store_id, sku_id, date) for post-promo lull
    
    for s_id in df_store["store_id"]:
        for sku_id in df_sku["sku_id"]:
            n_promo_blocks = np.random.poisson(total_sim_days * promo_frequency / 5.0)
            for _ in range(n_promo_blocks):
                p_start = np.random.randint(0, max(1, total_sim_days - 10))
                p_len = np.random.randint(3, 8)
                disc = np.random.choice([0.10, 0.15, 0.20, 0.25])
                lift = round(1.0 + disc * np.random.uniform(3.0, 5.0), 2)
                for day_offset in range(p_len):
                    if p_start + day_offset < total_sim_days:
                        p_date = sim_date_list[p_start + day_offset].strftime("%Y-%m-%d")
                        promo_rows.append((p_date, s_id, sku_id, disc, lift))
                        
                # Record post-promo lull window: 3-5 days after promo ends
                lull_start = p_start + p_len
                for l_day in range(lull_start, min(total_sim_days, lull_start + np.random.randint(3, 6))):
                    promo_end_dates.add((s_id, sku_id, l_day))
                    
    df_promo_all = pd.DataFrame(promo_rows, columns=["date", "store_id", "sku_id", "discount_pct", "promo_lift"]).drop_duplicates(subset=["date", "store_id", "sku_id"])

    # ---------------------------------------------------------
    # 4. SIMULATION DYNAMICS (Day-by-Day Loop)
    # ---------------------------------------------------------
    store_map = {row["store_id"]: row for _, row in df_store.iterrows()}
    sku_map = {row["sku_id"]: row for _, row in df_sku.iterrows()}
    promo_dict = {(r.date, r.store_id, r.sku_id): r.promo_lift for r in df_promo_all.itertuples()}
    
    # Delisted SKUs (2% of SKUs delisted starting after burn-in + 200 days)
    delisted_candidates = df_sku.sample(n=max(1, int(n_skus * 0.02)), random_state=seed+1)["sku_id"].tolist()
    delisted_start_day = burn_in_days + int(n_days * 0.55)
    
    # Demand shock timing: mid-year seasonal drop (e.g. days 180-260)
    shock_start_day = burn_in_days + 120
    shock_end_day = burn_in_days + 240
    
    pos_records = []
    ground_truth_records = []
    episode_records = []
    episode_id_counter = 1
    
    # State tracking: store_id -> sku_id -> dict
    state = {}
    for s_id in df_store["store_id"]:
        state[s_id] = {}
        for sku_id in df_sku["sku_id"]:
            s_info = sku_map[sku_id]
            init_stock = int(s_info["reorder_quantity"] * np.random.uniform(1.0, 1.8) + s_info["reorder_point"])
            state[s_id][sku_id] = {
                "physical_stock": init_stock,
                "system_stock": init_stock,
                "pending_orders": [],
                "active_episode": None,
            }
            
    dow_multipliers = [0.85, 0.88, 0.92, 0.95, 1.12, 1.35, 1.30]
    
    for day_idx, dt in enumerate(sim_date_list):
        date_str = dt.strftime("%Y-%m-%d")
        mm_dd = dt.strftime("%m-%d")
        dow = dt.weekday()
        dow_factor = dow_multipliers[dow]
        festive_factor = festive_calendar.get(mm_dd, 1.0)
        
        # Track which SKUs have 0 physical stock today for substitute lift
        empty_physical_by_store = {}
        for s_id in df_store["store_id"]:
            empty_physical_by_store[s_id] = {
                sku_id for sku_id, st in state[s_id].items() if st["physical_stock"] <= 0
            }
            
        for s_id in df_store["store_id"]:
            t_mult = store_map[s_id]["traffic_multiplier"]
            
            for sku_id in df_sku["sku_id"]:
                st = state[s_id][sku_id]
                s_info = sku_map[sku_id]
                
                # A. Arriving replenishment orders
                remaining_orders = []
                for arr_day, qty in st["pending_orders"]:
                    if arr_day <= day_idx:
                        st["system_stock"] += qty
                        if st["active_episode"] and st["active_episode"]["type"] == "phantom":
                            pass  # Stolen / lost / not placed on shelf
                        else:
                            st["physical_stock"] += qty
                    else:
                        remaining_orders.append((arr_day, qty))
                st["pending_orders"] = remaining_orders
                
                # B. Episode Lifecycle (Onset & Expiry)
                if st["active_episode"] and day_idx >= st["active_episode"]["end_day"]:
                    ep = st["active_episode"]
                    # If this episode was active during the saved window, log it
                    if ep["end_day"] >= burn_in_days:
                        episode_records.append({
                            "episode_id": ep["episode_id"],
                            "store_id": s_id,
                            "sku_id": sku_id,
                            "onset_date": sim_date_list[ep["onset_day"]].strftime("%Y-%m-%d"),
                            "end_date": date_str,
                            "duration_days": day_idx - ep["onset_day"],
                            "failure_type": ep["type"]
                        })
                    st["physical_stock"] = max(0, st["system_stock"])
                    st["active_episode"] = None
                    
                # New Episode Injection (burn-in allowed to run smoothly; onset strictly requires system_stock >= 3)
                if st["active_episode"] is None and day_idx < total_sim_days - 10:
                    if sku_id in delisted_candidates and day_idx >= delisted_start_day:
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "delisted",
                            "onset_day": day_idx,
                            "end_day": total_sim_days
                        }
                        episode_id_counter += 1
                        st["physical_stock"] = 0
                    elif st["system_stock"] >= 3 and np.random.rand() < phantom_onset_prob:
                        # Skewed duration: Geometric distribution with mean 21 days (clipped [7, 56])
                        dur = int(np.clip(np.random.geometric(p=1.0 / mean_phantom_duration), 7, 56))
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "phantom",
                            "onset_day": day_idx,
                            "end_day": min(total_sim_days, day_idx + dur)
                        }
                        episode_id_counter += 1
                        st["physical_stock"] = 0  # Physical shelf emptied; system stock remains > 0!
                    elif st["system_stock"] >= 3 and np.random.rand() < 0.00025:
                        dur = int(np.random.uniform(5, 18))
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "misplaced",
                            "onset_day": day_idx,
                            "end_day": min(total_sim_days, day_idx + dur)
                        }
                        episode_id_counter += 1
                
                # C. Confuser Status & Demand Modifiers
                promo_lift = promo_dict.get((date_str, s_id, sku_id), 1.0)
                is_on_promo = 1 if promo_lift > 1.0 else 0
                
                is_post_promo_lull = 1 if ((s_id, sku_id, day_idx) in promo_end_dates and not is_on_promo) else 0
                is_demand_shock = 1 if (sku_id in demand_shock_skus and shock_start_day <= day_idx <= shock_end_day) else 0
                
                # Demand multipliers for confusers
                confuser_mult = 1.0
                if is_demand_shock:
                    confuser_mult *= 0.15  # 85% drop in demand
                if is_post_promo_lull:
                    confuser_mult *= 0.25  # 75% lull post-promo
                    
                sub_sku = s_info["substitute_sku_id"]
                sub_lift = 1.35 if (sub_sku and sub_sku in empty_physical_by_store[s_id]) else 1.0
                
                expected_lambda = (
                    s_info["base_demand_rate"]
                    * t_mult
                    * dow_factor
                    * festive_factor
                    * promo_lift
                    * sub_lift
                    * confuser_mult
                )
                
                # Realized demand from Negative Binomial
                if expected_lambda <= 0.001:
                    realized_demand = 0
                else:
                    var = expected_lambda + overdispersion_alpha * (expected_lambda ** 2)
                    p_nb = expected_lambda / var
                    n_nb = (expected_lambda ** 2) / (var - expected_lambda)
                    realized_demand = int(np.random.negative_binomial(max(0.1, n_nb), min(0.999, p_nb)))
                
                # D. Realize Sales
                current_ep = st["active_episode"]
                if current_ep and current_ep["type"] == "phantom":
                    sales = 0
                elif current_ep and current_ep["type"] == "delisted":
                    sales = 0
                elif current_ep and current_ep["type"] == "misplaced":
                    trickle_stock = max(0, int(st["physical_stock"] * 0.2))
                    sales = min(trickle_stock, realized_demand if np.random.rand() < 0.25 else 0)
                    st["physical_stock"] = max(0, st["physical_stock"] - sales)
                    st["system_stock"] = max(0, st["system_stock"] - sales)
                else:
                    sales = min(st["physical_stock"], realized_demand)
                    st["physical_stock"] = max(0, st["physical_stock"] - sales)
                    st["system_stock"] = max(0, st["system_stock"] - sales)
                    
                # E. Replenishment Trigger
                total_pipeline = st["system_stock"] + sum(q for _, q in st["pending_orders"])
                if total_pipeline <= s_info["reorder_point"]:
                    arrival_day = day_idx + s_info["lead_time_days"]
                    st["pending_orders"].append((arrival_day, s_info["reorder_quantity"]))
                    
                # F. Filter Out Burn-In Period (Days 0 to burn_in_days - 1)
                if day_idx >= burn_in_days:
                    pos_records.append({
                        "date": date_str,
                        "store_id": s_id,
                        "sku_id": sku_id,
                        "sales": sales,
                        "system_stock": st["system_stock"],
                        "on_promo": is_on_promo,
                        "unit_price": s_info["unit_price_inr"],
                        "gross_margin_pct": s_info["gross_margin_pct"],
                    })
                    
                    ep_type = current_ep["type"] if current_ep else "none"
                    is_phantom = 1 if ep_type == "phantom" else 0
                    is_misplaced = 1 if ep_type == "misplaced" else 0
                    is_delisted = 1 if ep_type == "delisted" else 0
                    is_true_so = 1 if (st["physical_stock"] == 0 and st["system_stock"] == 0) else 0
                    is_slow_zero = 1 if (sales == 0 and st["system_stock"] > 0 and not is_phantom and not is_misplaced and not is_delisted and not is_demand_shock and not is_post_promo_lull) else 0
                    ep_id = current_ep["episode_id"] if current_ep else None
                    days_since_onset = (day_idx - current_ep["onset_day"]) if current_ep else -1
                    
                    ground_truth_records.append({
                        "date": date_str,
                        "store_id": s_id,
                        "sku_id": sku_id,
                        "is_phantom": is_phantom,
                        "is_misplaced": is_misplaced,
                        "is_delisted": is_delisted,
                        "is_true_stockout": is_true_so,
                        "is_demand_shock": is_demand_shock,
                        "is_post_promo_lull": is_post_promo_lull,
                        "is_slow_mover_zero": is_slow_zero,
                        "physical_stock": st["physical_stock"],
                        "episode_id": ep_id,
                        "days_since_onset": days_since_onset,
                        "unfulfilled_demand": max(0, realized_demand - sales)
                    })


    # Close any ongoing episodes at simulation end
    min_saved_date = sim_date_list[burn_in_days].strftime("%Y-%m-%d")
    for s_id in df_store["store_id"]:
        for sku_id in df_sku["sku_id"]:
            ep = state[s_id][sku_id]["active_episode"]
            if ep and ep["end_day"] >= burn_in_days:
                episode_records.append({
                    "episode_id": ep["episode_id"],
                    "store_id": s_id,
                    "sku_id": sku_id,
                    "onset_date": sim_date_list[ep["onset_day"]].strftime("%Y-%m-%d"),
                    "end_date": sim_date_list[-1].strftime("%Y-%m-%d"),
                    "duration_days": total_sim_days - ep["onset_day"],
                    "failure_type": ep["type"]
                })

    df_pos = pd.DataFrame(pos_records)
    df_truth = pd.DataFrame(ground_truth_records)
    df_episodes = pd.DataFrame(episode_records).drop_duplicates(subset=["episode_id"])
    df_promo = df_promo_all[df_promo_all["date"] >= min_saved_date].copy()
    
    # ---------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # ---------------------------------------------------------
    df_store.to_csv(os.path.join(output_dir, "store_master.csv"), index=False)
    df_sku.to_csv(os.path.join(output_dir, "sku_master.csv"), index=False)
    
    if compress:
        df_promo.to_csv(os.path.join(output_dir, "promo_calendar.csv.gz"), index=False, compression="gzip")
        df_pos.to_csv(os.path.join(output_dir, "daily_sales_and_stock.csv.gz"), index=False, compression="gzip")
        df_truth.to_csv(os.path.join(output_dir, "ground_truth_labels.csv.gz"), index=False, compression="gzip")
    else:
        df_promo.to_csv(os.path.join(output_dir, "promo_calendar.csv"), index=False)
        df_pos.to_csv(os.path.join(output_dir, "daily_sales_and_stock.csv"), index=False)
        df_truth.to_csv(os.path.join(output_dir, "ground_truth_labels.csv"), index=False)
        
    df_episodes.to_csv(os.path.join(output_dir, "ground_truth_episodes.csv"), index=False)
    
    # Save metadata summary
    phantom_share = df_truth["is_phantom"].mean()
    true_so_share = df_truth["is_true_stockout"].mean()
    misplaced_share = df_truth["is_misplaced"].mean()
    
    metadata = {
        "n_stores": n_stores,
        "n_skus": n_skus,
        "n_days": n_days,
        "start_date": start_date,
        "seed": seed,
        "total_sku_days": len(df_pos),
        "total_phantom_episodes": len(df_episodes[df_episodes["failure_type"] == "phantom"]),
        "total_misplaced_episodes": len(df_episodes[df_episodes["failure_type"] == "misplaced"]),
        "total_delisted_episodes": len(df_episodes[df_episodes["failure_type"] == "delisted"]),
        "phantom_row_share": round(float(phantom_share), 5),
        "true_stockout_row_share": round(float(true_so_share), 5),
        "misplaced_row_share": round(float(misplaced_share), 5),
        "generation_timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return df_store, df_sku, df_promo, df_pos, df_truth
