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
    n_days: int = 180,
    start_date: str = "2026-01-01",
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
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
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
        # generate extra synthetic locations
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
                "substitute_sku_id": None  # Assigned next
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
    # 3. CALENDAR & PROMO SCHEDULE
    # ---------------------------------------------------------
    base_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [base_dt + timedelta(days=d) for d in range(n_days)]
    
    # Festive calendar impacts (Republic day, Holi, Festive weekend, etc.)
    festive_dates = {
        "2026-01-26": 1.55,  # Republic Day Big Sale
        "2026-01-25": 1.40,
        "2026-01-24": 1.35,
        "2026-03-03": 1.45,  # Holi week
        "2026-03-04": 1.50,
        "2026-04-14": 1.30,  # Baisakhi / Tamil New Year
        "2026-05-01": 1.25,  # May Day holiday
    }
    
    # Generate promo table: (date, store_id, sku_id) -> promo_discount, promo_lift
    promo_rows = []
    # Vectorized promo assignment: select active promo blocks
    # SKUs run promos for 3-7 day blocks
    for s_id in df_store["store_id"]:
        for sku_id in df_sku["sku_id"]:
            # On average ~ promo_frequency share of days
            n_promo_blocks = np.random.poisson(n_days * promo_frequency / 5.0)
            for _ in range(n_promo_blocks):
                p_start = np.random.randint(0, max(1, n_days - 7))
                p_len = np.random.randint(3, 8)
                disc = np.random.choice([0.10, 0.15, 0.20, 0.25])
                lift = round(1.0 + disc * np.random.uniform(3.0, 5.0), 2)
                for day_offset in range(min(p_len, n_days - p_start)):
                    p_date = (base_dt + timedelta(days=p_start + day_offset)).strftime("%Y-%m-%d")
                    promo_rows.append((p_date, s_id, sku_id, disc, lift))
                    
    df_promo = pd.DataFrame(promo_rows, columns=["date", "store_id", "sku_id", "discount_pct", "promo_lift"]).drop_duplicates(subset=["date", "store_id", "sku_id"])

    # ---------------------------------------------------------
    # 4. SIMULATION DYNAMICS (Day-by-Day Loop)
    # ---------------------------------------------------------
    # Fast indexing structures
    store_map = {row["store_id"]: row for _, row in df_store.iterrows()}
    sku_map = {row["sku_id"]: row for _, row in df_sku.iterrows()}
    promo_dict = {(r.date, r.store_id, r.sku_id): r.promo_lift for r in df_promo.itertuples()}
    
    # Pre-select Delisted SKUs (e.g. 2% of SKUs delisted halfway through)
    delisted_candidates = df_sku.sample(n=max(1, int(n_skus * 0.02)), random_state=seed)["sku_id"].tolist()
    delisted_start_day = int(n_days * 0.65)
    
    # Daily logs
    pos_records = []
    ground_truth_records = []
    episode_records = []
    
    episode_id_counter = 1
    
    # State tracking: store_id -> sku_id -> dict(physical_stock, system_stock, orders, episode)
    state = {}
    for s_id in df_store["store_id"]:
        state[s_id] = {}
        t_mult = store_map[s_id]["traffic_multiplier"]
        for sku_id in df_sku["sku_id"]:
            s_info = sku_map[sku_id]
            init_stock = int(s_info["reorder_quantity"] * np.random.uniform(0.9, 1.8) + s_info["reorder_point"])
            state[s_id][sku_id] = {
                "physical_stock": init_stock,
                "system_stock": init_stock,
                "pending_orders": [],  # list of (arrival_day, qty)
                "active_episode": None,  # dict(episode_id, type, onset_day, end_day)
                "cooldown_until": 0,
            }
            
    # Day-of-week multipliers: Mon=0.85, Tue=0.88, Wed=0.92, Thu=0.95, Fri=1.12, Sat=1.35, Sun=1.30
    dow_multipliers = [0.85, 0.88, 0.92, 0.95, 1.12, 1.35, 1.30]
    
    for day_idx, dt in enumerate(date_list):
        date_str = dt.strftime("%Y-%m-%d")
        dow = dt.weekday()
        dow_factor = dow_multipliers[dow]
        festive_factor = festive_dates.get(date_str, 1.0)
        
        # Track which SKUs in each store have 0 physical stock today for substitute lift
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
                
                # A. Process arriving replenishment orders
                remaining_orders = []
                for arr_day, qty in st["pending_orders"]:
                    if arr_day <= day_idx:
                        # Order arrived!
                        st["system_stock"] += qty
                        # Physical stock only replenishes if goods are actually placed on shelf
                        # (If phantom is active, physical shelf was emptied and replenishment delivery
                        # does not happen if reorder never triggered; but if an order was already in-flight before phantom onset,
                        # it lands on shelf unless stolen).
                        if st["active_episode"] and st["active_episode"]["type"] == "phantom":
                            # Phantom: stock disappears / lost in backroom / theft
                            pass
                        else:
                            st["physical_stock"] += qty
                    else:
                        remaining_orders.append((arr_day, qty))
                st["pending_orders"] = remaining_orders
                
                # B. Check Episode Onsets / Expirations
                # Check if current episode has expired
                if st["active_episode"] and day_idx >= st["active_episode"]["end_day"]:
                    # Episode ended (e.g. periodic cycle count discovery / reset)
                    ep = st["active_episode"]
                    episode_records.append({
                        "episode_id": ep["episode_id"],
                        "store_id": s_id,
                        "sku_id": sku_id,
                        "onset_date": date_list[ep["onset_day"]].strftime("%Y-%m-%d"),
                        "end_date": date_str,
                        "duration_days": day_idx - ep["onset_day"],
                        "failure_type": ep["type"]
                    })
                    # Reset physical stock to system stock upon audit/cycle count
                    st["physical_stock"] = max(0, st["system_stock"])
                    st["active_episode"] = None
                    
                # Inject new episode if eligible
                if st["active_episode"] is None and day_idx < n_days - 10:
                    # Check Delisted condition
                    if sku_id in delisted_candidates and day_idx >= delisted_start_day:
                        ep_dur = n_days - day_idx
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "delisted",
                            "onset_day": day_idx,
                            "end_day": day_idx + ep_dur
                        }
                        episode_id_counter += 1
                        st["physical_stock"] = 0
                        # System stock freezes or slowly decays
                    # Check Phantom Onset (Plan: ~0.143% hazard per day, requires system_stock > 2)
                    elif st["system_stock"] >= 3 and np.random.rand() < phantom_onset_prob:
                        dur = int(np.clip(np.random.exponential(mean_phantom_duration), 7, 45))
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "phantom",
                            "onset_day": day_idx,
                            "end_day": min(n_days, day_idx + dur)
                        }
                        episode_id_counter += 1
                        st["physical_stock"] = 0  # Physical shelf emptied!
                        # System stock remains frozen and positive!
                    # Stretch: Check Misplaced Onset (low probability ~0.00025)
                    elif st["system_stock"] >= 3 and np.random.rand() < 0.00025:
                        dur = int(np.random.uniform(5, 18))
                        st["active_episode"] = {
                            "episode_id": f"EP_{episode_id_counter:05d}",
                            "type": "misplaced",
                            "onset_day": day_idx,
                            "end_day": min(n_days, day_idx + dur)
                        }
                        episode_id_counter += 1
                        # Misplaced on adjacent bay: small trickle of sales (50-80% reduced availability)
                
                # C. Compute Expected & Realized Demand
                promo_lift = promo_dict.get((date_str, s_id, sku_id), 1.0)
                is_on_promo = 1 if promo_lift > 1.0 else 0
                
                # Substitute boost: if sibling SKU has 0 physical stock in this store, boost this SKU
                sub_sku = s_info["substitute_sku_id"]
                sub_lift = 1.0
                if sub_sku and sub_sku in empty_physical_by_store[s_id]:
                    sub_lift = 1.35  # 35% category substitution lift
                    
                expected_lambda = s_info["base_demand_rate"] * t_mult * dow_factor * festive_factor * promo_lift * sub_lift
                
                # Realized demand from Negative Binomial distribution for over-dispersion
                # NegBin parameterization: mean = mu, var = mu + alpha * mu^2
                if expected_lambda <= 0.001:
                    realized_demand = 0
                else:
                    var = expected_lambda + overdispersion_alpha * (expected_lambda ** 2)
                    p_nb = expected_lambda / var
                    n_nb = (expected_lambda ** 2) / (var - expected_lambda)
                    realized_demand = int(np.random.negative_binomial(max(0.1, n_nb), min(0.999, p_nb)))
                
                # D. Realize Sales and Decrement Stock
                current_ep = st["active_episode"]
                if current_ep and current_ep["type"] == "phantom":
                    # Shelf empty: 0 sales possible
                    sales = 0
                elif current_ep and current_ep["type"] == "delisted":
                    sales = 0
                elif current_ep and current_ep["type"] == "misplaced":
                    # Trickle sales only
                    trickle_stock = max(0, int(st["physical_stock"] * 0.2))
                    sales = min(trickle_stock, realized_demand if np.random.rand() < 0.25 else 0)
                    st["physical_stock"] = max(0, st["physical_stock"] - sales)
                    st["system_stock"] = max(0, st["system_stock"] - sales)
                else:
                    # Normal sales
                    sales = min(st["physical_stock"], realized_demand)
                    st["physical_stock"] = max(0, st["physical_stock"] - sales)
                    st["system_stock"] = max(0, st["system_stock"] - sales)
                    
                # E. Replenishment Trigger Logic
                # If system stock + in-flight orders <= reorder_point, place order!
                total_pipeline = st["system_stock"] + sum(q for _, q in st["pending_orders"])
                if total_pipeline <= s_info["reorder_point"]:
                    arrival_day = day_idx + s_info["lead_time_days"]
                    st["pending_orders"].append((arrival_day, s_info["reorder_quantity"]))
                    
                # F. Record POS & Inventory data (visible to detector)
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
                
                # G. Record Ground Truth (hidden, for evaluation only)
                ep_type = current_ep["type"] if current_ep else "none"
                is_phantom = 1 if ep_type == "phantom" else 0
                is_misplaced = 1 if ep_type == "misplaced" else 0
                is_delisted = 1 if ep_type == "delisted" else 0
                is_true_stockout = 1 if (st["physical_stock"] == 0 and st["system_stock"] == 0) else 0
                ep_id = current_ep["episode_id"] if current_ep else None
                days_since_onset = (day_idx - current_ep["onset_day"]) if current_ep else -1
                
                ground_truth_records.append({
                    "date": date_str,
                    "store_id": s_id,
                    "sku_id": sku_id,
                    "is_phantom": is_phantom,
                    "is_misplaced": is_misplaced,
                    "is_delisted": is_delisted,
                    "is_true_stockout": is_true_stockout,
                    "physical_stock": st["physical_stock"],
                    "episode_id": ep_id,
                    "days_since_onset": days_since_onset,
                    "unfulfilled_demand": max(0, realized_demand - sales)
                })

    # Close any ongoing episodes at simulation end
    for s_id in df_store["store_id"]:
        for sku_id in df_sku["sku_id"]:
            ep = state[s_id][sku_id]["active_episode"]
            if ep:
                episode_records.append({
                    "episode_id": ep["episode_id"],
                    "store_id": s_id,
                    "sku_id": sku_id,
                    "onset_date": date_list[ep["onset_day"]].strftime("%Y-%m-%d"),
                    "end_date": date_list[-1].strftime("%Y-%m-%d"),
                    "duration_days": n_days - ep["onset_day"],
                    "failure_type": ep["type"]
                })

    df_pos = pd.DataFrame(pos_records)
    df_truth = pd.DataFrame(ground_truth_records)
    df_episodes = pd.DataFrame(episode_records).drop_duplicates(subset=["episode_id"])
    
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
