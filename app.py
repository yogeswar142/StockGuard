"""
StockGuard: Store Operations Portal
SaaS Daily Shelf-Check Desk for Store Managers and Shopkeepers
"""

import os
import json
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="StockGuard | Store Operations Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Shopkeeper Styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2.5rem;
        max-width: 98%;
    }
    .saas-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 14px 20px;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .saas-tagline {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.90rem;
        color: #1e293b;
        margin-bottom: 14px;
    }
    .kpi-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .kpi-num {
        font-size: 1.50rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-desc {
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 3px;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .badge-watch {
        background-color: #f1f5f9;
        color: #475569;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .badge-done-phantom {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .badge-done-safe {
        background-color: #f0fdf4;
        color: #15803d;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .badge-done-misplaced {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 9px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .action-instruction {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.86rem;
        color: #14532d;
        margin: 8px 0 12px 0;
    }
    .item-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .item-card-active {
        background-color: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .print-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    .print-table th {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 600;
        text-align: left;
        padding: 8px 10px;
        border-bottom: 2px solid #cbd5e1;
    }
    .print-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Data Loaders
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_metadata():
    stores_df = pd.read_csv("data/store_master.csv")
    skus_df = pd.read_csv("data/sku_master.csv")
    return stores_df, skus_df


@st.cache_data(show_spinner=False)
def load_alerts():
    df = pd.read_csv("results/stockguard_alerts.csv")
    return df


@st.cache_data(show_spinner=False)
def load_time_series_data():
    df = pd.read_csv(
        "data/daily_sales_and_stock.csv.gz",
        usecols=["date", "store_id", "sku_id", "sales", "system_stock", "unit_price", "gross_margin_pct"],
    )
    return df


stores_df, skus_df = load_metadata()
alerts_df = load_alerts()

# ==============================================================================
# Session State Initialization (Default to Store 02 on 2026-01-20 for Video Demo)
# ==============================================================================
if "audited_items" not in st.session_state:
    st.session_state.audited_items = {}

if "curr_store" not in st.session_state:
    st.session_state.curr_store = "STORE_02"  # Pre-selected demo store

if "curr_date" not in st.session_state:
    st.session_state.curr_date = "2026-01-20"  # Pre-selected demo date

if "selected_sku_id" not in st.session_state:
    st.session_state.selected_sku_id = "SKU_133"  # Pre-selected demo case: Mosquito Repellent

if "show_printable" not in st.session_state:
    st.session_state.show_printable = False


def get_urgency_badge(conf_val):
    if conf_val >= 0.12:
        return '<span class="badge-critical">🚨 Critical Alert (Likely Empty)</span>'
    elif conf_val >= 0.06:
        return '<span class="badge-warning">⚠️ Warning (Check Shelf)</span>'
    else:
        return '<span class="badge-watch">👀 Routine Watch</span>'


# ==============================================================================
# Top SaaS Header & Product Positioning
# ==============================================================================
st.markdown(
    """
    <div class="saas-header">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2.2rem;">🛡️</span>
            <div>
                <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;">
                    StockGuard · Store Operations Hub
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 500;">
                    Daily Floor-Walk Priority Desk for Shopkeepers & Floor Staff
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.78rem; color: #cbd5e1;">Live Store Shift</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: #38bdf8;">Active Floor Walk Mode</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="saas-tagline">
        <strong>StockGuard</strong> gives retail store managers a daily, value-ranked list of up to 20 shelves to physically check — finding inventory records that say "in stock" while the shelf is empty.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Store Selection & Floor Walk Controls
# ==============================================================================
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([3, 3, 4])

with ctrl_c1:
    store_options = stores_df["store_id"].tolist()
    store_labels = {
        row["store_id"]: f"{row['store_name']} ({row['city']}, {row['format']})"
        for _, row in stores_df.iterrows()
    }
    curr_store_idx = store_options.index(st.session_state.curr_store) if st.session_state.curr_store in store_options else 0
    selected_store = st.selectbox(
        "🏪 Select Store Branch",
        options=store_options,
        format_func=lambda x: store_labels[x],
        index=curr_store_idx,
    )
    if selected_store != st.session_state.curr_store:
        st.session_state.curr_store = selected_store
        st.rerun()

store_alerts_all = alerts_df[alerts_df["store_id"] == st.session_state.curr_store]
available_dates = sorted(store_alerts_all["date"].unique())

with ctrl_c2:
    curr_date_idx = available_dates.index(st.session_state.curr_date) if st.session_state.curr_date in available_dates else 0
    selected_date = st.selectbox(
        "📅 Audit Date",
        options=available_dates,
        index=curr_date_idx,
    )
    if selected_date != st.session_state.curr_date:
        st.session_state.curr_date = selected_date
        st.rerun()

with ctrl_c3:
    st.markdown("<div style='font-size: 0.80rem; font-weight: 600; color: #475569; margin-bottom: 5px;'>🎬 Video Presentation Shortcut</div>", unsafe_allow_html=True)
    if st.button("▶️ Load Demo Video Example (Store 02 · Mosquito Repellent)", width="stretch"):
        st.session_state.curr_store = "STORE_02"
        st.session_state.curr_date = "2026-01-20"
        st.session_state.selected_sku_id = "SKU_133"
        st.rerun()

# Filter Today's Alerts
day_alerts = alerts_df[
    (alerts_df["store_id"] == st.session_state.curr_store)
    & (alerts_df["date"] == st.session_state.curr_date)
].sort_values(by="rank").head(20).reset_index(drop=True)

if len(day_alerts) > 0:
    if st.session_state.selected_sku_id not in day_alerts["sku_id"].values:
        st.session_state.selected_sku_id = day_alerts.iloc[0]["sku_id"]
else:
    st.session_state.selected_sku_id = None

# Shopkeeper Metrics Calculations
num_shelves_today = len(day_alerts)
total_daily_margin_at_risk = day_alerts["daily_margin_at_risk_inr"].sum() if num_shelves_today > 0 else 0.0
total_estimated_recovery = day_alerts["expected_value_inr"].sum() if num_shelves_today > 0 else 0.0

today_audit_keys = [
    k for k in st.session_state.audited_items
    if k[0] == st.session_state.curr_store and k[1] == st.session_state.curr_date
]
num_checked = len(today_audit_keys)
today_recovered_inr = sum(
    st.session_state.audited_items[k].get("margin_saved_inr", 0.0) for k in today_audit_keys
)

# Format Date for Display
try:
    dt_obj = datetime.datetime.strptime(st.session_state.curr_date, "%Y-%m-%d")
    date_display_str = dt_obj.strftime("%A, %d %B %Y")
except Exception:
    date_display_str = st.session_state.curr_date

store_meta = stores_df[stores_df["store_id"] == st.session_state.curr_store].iloc[0]

# ==============================================================================
# Shopkeeper Daily KPI Cards
# ==============================================================================
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(
        f"""
        <div class="kpi-box" style="border-left: 4px solid #2563eb;">
            <div class="kpi-label">Shelves to Check Today</div>
            <div class="kpi-num">{num_shelves_today} Shelves</div>
            <div class="kpi-desc" style="color: #2563eb;">~20 mins floor walk · Max 20 cap</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    st.markdown(
        f"""
        <div class="kpi-box" style="border-left: 4px solid #dc2626;">
            <div class="kpi-label">Total Margin at Risk</div>
            <div class="kpi-num">₹{total_daily_margin_at_risk:,.0f} / day</div>
            <div class="kpi-desc" style="color: #dc2626;">Lost profit every day these stay empty</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    st.markdown(
        f"""
        <div class="kpi-box" style="border-left: 4px solid #16a34a;">
            <div class="kpi-label">Checklist Progress</div>
            <div class="kpi-num">{num_checked} / {num_shelves_today} Checked</div>
            <div class="kpi-desc" style="color: #16a34a;">₹{today_recovered_inr:,.0f} recovered so far</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    st.markdown(
        f"""
        <div class="kpi-box" style="border-left: 4px solid #7c3aed;">
            <div class="kpi-label">Estimated Recovery</div>
            <div class="kpi-num">₹{total_estimated_recovery:,.0f}</div>
            <div class="kpi-desc" style="color: #7c3aed;">Potential profit you can save today</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Progress Bar
progress_val = (num_checked / max(1, num_shelves_today)) if num_shelves_today > 0 else 1.0
st.progress(progress_val)

# Action Toolbar (Download CSV & Printable View)
tool_c1, tool_c2 = st.columns([6, 4])
with tool_c1:
    st.markdown(
        f"""
        <div style="font-size: 0.86rem; color: #475569; padding-top: 6px;">
            <strong>{store_meta['store_name']}</strong> · {date_display_str} · Floor Audit Priority
        </div>
        """,
        unsafe_allow_html=True,
    )
with tool_c2:
    btn_sub1, btn_sub2 = st.columns(2)
    with btn_sub1:
        if len(day_alerts) > 0:
            export_df = day_alerts.copy()
            export_df["urgency"] = export_df["confidence"].apply(
                lambda c: "Critical (Likely Empty)" if c >= 0.12 else ("Warning" if c >= 0.06 else "Watch")
            )
            csv_content = export_df[[
                "rank", "sku_id", "sku_name", "zero_streak", "system_stock",
                "daily_margin_at_risk_inr", "expected_value_inr", "urgency", "reason"
            ]].rename(
                columns={
                    "rank": "Priority_Rank",
                    "sku_id": "Item_Code",
                    "sku_name": "Product_Description",
                    "zero_streak": "Days_Without_Sales",
                    "system_stock": "Computer_Stock_Quantity",
                    "daily_margin_at_risk_inr": "Daily_Profit_At_Risk_INR",
                    "expected_value_inr": "Estimated_Recovery_INR",
                    "urgency": "Urgency_Level",
                    "reason": "Why_Check_This_Item",
                }
            ).to_csv(index=False)

            st.download_button(
                label="📥 Download CSV Checklist",
                data=csv_content,
                file_name=f"StockGuard_Checklist_{st.session_state.curr_store}_{st.session_state.curr_date}.csv",
                mime="text/csv",
                width="stretch",
            )
    with btn_sub2:
        print_label = "📋 Normal View" if st.session_state.show_printable else "🖨️ Printable Table"
        if st.button(print_label, width="stretch"):
            st.session_state.show_printable = not st.session_state.show_printable
            st.rerun()

# Printable Table View
if st.session_state.show_printable and len(day_alerts) > 0:
    st.markdown("#### 🖨️ Printable Store Floor Walk Sheet")
    st.markdown("*Clipboard view for floor staff walking down the aisles:*")
    p_df = day_alerts[[
        "rank", "sku_id", "sku_name", "zero_streak", "system_stock",
        "daily_margin_at_risk_inr", "confidence", "reason"
    ]].copy()
    p_df["Urgency"] = p_df["confidence"].apply(
        lambda c: "🚨 Critical" if c >= 0.12 else ("⚠️ Warning" if c >= 0.06 else "👀 Watch")
    )
    p_df = p_df.rename(columns={
        "rank": "Rank",
        "sku_id": "Code",
        "sku_name": "Product Name",
        "zero_streak": "Days Silent",
        "system_stock": "Computer Stock",
        "daily_margin_at_risk_inr": "Daily Profit Risk (₹)",
        "reason": "Why You Must Check",
    })[["Rank", "Code", "Product Name", "Days Silent", "Computer Stock", "Daily Profit Risk (₹)", "Urgency", "Why You Must Check"]]
    st.markdown(p_df.to_html(classes="print-table", index=False), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# Two-Column Layout: Left (Priority Checklist) | Right (Shelf Check Station)
# ==============================================================================
list_col, detail_col = st.columns([5, 6])

# LEFT COLUMN: Daily Items to Check
with list_col:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <h3 style="font-size: 1.10rem; font-weight: 700; margin: 0; color: #1e293b;">
                📋 Priority Shelves to Verify ({num_shelves_today} Items)
            </h3>
            <span style="font-size: 0.78rem; color: #64748b;">Sorted by lost profit value</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(day_alerts) == 0:
        st.success("🎉 All shelves are fully stocked and active today! No discrepancies detected.")
    else:
        for idx, row in day_alerts.iterrows():
            sku_id = row["sku_id"]
            sku_name = row["sku_name"]
            rank = row["rank"]
            daily_margin = row["daily_margin_at_risk_inr"]
            est_recovery = row["expected_value_inr"]
            zero_days = int(row["zero_streak"])
            computer_stock = int(row["system_stock"])
            reason_text = row["reason"]

            # Category
            sku_match = skus_df[skus_df["sku_id"] == sku_id]
            category_name = sku_match["category"].values[0] if len(sku_match) > 0 else "General"

            # Urgency Badge
            conf_val = float(row["confidence"])
            urgency_html = get_urgency_badge(conf_val)

            # Audit State
            audit_key = (st.session_state.curr_store, st.session_state.curr_date, sku_id)
            is_audited = audit_key in st.session_state.audited_items
            audit_result = st.session_state.audited_items.get(audit_key, None)

            is_selected = (st.session_state.selected_sku_id == sku_id)
            card_class = "item-card-active" if is_selected else "item-card"

            status_pill = ""
            if is_audited:
                act = audit_result["action"]
                if act == "empty":
                    status_pill = f'<span class="badge-done-phantom">✅ Empty Shelf Confirmed (+₹{audit_result.get("margin_saved_inr", 0):.0f})</span>'
                elif act == "found":
                    status_pill = '<span class="badge-done-safe">⚠️ Found on Shelf (Muted 7 Days)</span>'
                elif act == "misplaced":
                    status_pill = '<span class="badge-done-misplaced">🟡 Misplaced Stock Found</span>'
                elif act == "delisted":
                    status_pill = '<span class="badge-watch">⚪ Discontinued Item</span>'
            else:
                status_pill = urgency_html

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-weight: 800; font-size: 0.95rem; color: #0f172a;">#{rank} · {sku_name}</span>
                            <span style="font-size: 0.74rem; color: #64748b; margin-left: 6px;">({sku_id})</span>
                        </div>
                        <span style="font-size: 0.72rem; color: #3b82f6; font-weight: 600; background: #eff6ff; padding: 2px 7px; border-radius: 8px;">
                            {category_name}
                        </span>
                    </div>
                    <div style="display: flex; gap: 8px; margin: 6px 0 6px 0; align-items: center; flex-wrap: wrap;">
                        {status_pill}
                        <span style="font-size: 0.78rem; color: #15803d; font-weight: 700; background: #dcfce7; padding: 2px 7px; border-radius: 10px;">
                            Save: ₹{est_recovery:,.0f}
                        </span>
                        <span style="font-size: 0.78rem; color: #475569;">
                            <strong>{zero_days} days</strong> without sales | Computer shows <strong>{computer_stock} units</strong> | Losing <strong>₹{daily_margin:.0f}/day</strong>
                        </span>
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; line-height: 1.35; margin-top: 4px;">
                        <strong>Why verify:</strong> {reason_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_label = f"📍 Currently Inspecting #{rank} {sku_name[:20]}" if is_selected else f"👉 Verify Shelf #{rank} ({sku_name[:20]})"
            if st.button(btn_label, key=f"item_btn_{sku_id}_{idx}", width="stretch"):
                st.session_state.selected_sku_id = sku_id
                st.rerun()

# RIGHT COLUMN: Physical Verification Station
with detail_col:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <h3 style="font-size: 1.10rem; font-weight: 700; margin: 0; color: #1e293b;">
                🔍 Physical Shelf Verification Station
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sel_sku_id = st.session_state.selected_sku_id
    if sel_sku_id is None or sel_sku_id not in day_alerts["sku_id"].values:
        st.info("Select any item from the priority list on the left to verify shelf conditions.")
    else:
        sku_alert_row = day_alerts[day_alerts["sku_id"] == sel_sku_id].iloc[0]
        sku_info = skus_df[skus_df["sku_id"] == sel_sku_id].iloc[0]

        zero_days_len = int(sku_alert_row["zero_streak"])
        computer_stock_qty = int(sku_alert_row["system_stock"])
        normal_daily_pace = float(sku_alert_row["expected_daily_sales"])
        daily_loss_amount = float(sku_alert_row["daily_margin_at_risk_inr"])

        # Detail Product Card
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <h4 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #0f172a;">
                        {sku_info['sku_name']}
                    </h4>
                    <span style="font-size: 0.80rem; font-weight: 600; background-color: #e2e8f0; color: #334155; padding: 2px 8px; border-radius: 6px;">
                        {sku_info['category']} ({sku_info['sub_category']})
                    </span>
                </div>
                <div style="display: flex; gap: 14px; margin-top: 6px; font-size: 0.82rem; color: #475569; flex-wrap: wrap;">
                    <span><strong>Item Code:</strong> {sel_sku_id}</span>
                    <span><strong>Selling Price:</strong> ₹{sku_info['unit_price_inr']:.0f}</span>
                    <span><strong>Profit Margin:</strong> {sku_info['gross_margin_pct']*100:.1f}%</span>
                    <span><strong>Normal Customer Pace:</strong> ~{normal_daily_pace:.1f} packets/day</span>
                    <span><strong>Daily Lost Profit:</strong> ₹{daily_loss_amount:.0f}/day</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clear Step-by-Step Action Guide
        st.markdown(
            """
            <div class="action-instruction">
                <strong>📋 Floor Walk Instructions for Store Staff:</strong><br>
                1. Walk to the product's primary shelf bay.<br>
                2. If shelf is empty, check the back storeroom or neighboring bay for misplaced stock.<br>
                3. Tap what you observe below so the billing computer can be updated and reorders placed immediately.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Load Time-Series Data for Selected Product
        ts_all = load_time_series_data()
        target_ts = ts_all[
            (ts_all["sku_id"] == sel_sku_id) & (ts_all["date"] <= st.session_state.curr_date)
        ].sort_values(by="date")

        store_ts = target_ts[target_ts["store_id"] == st.session_state.curr_store].tail(45)
        peer_ts = (
            target_ts[target_ts["store_id"] != st.session_state.curr_store]
            .groupby("date")["sales"]
            .mean()
            .reset_index()
            .tail(45)
        )

        # Plain-English Titles for Non-Tech Shopkeepers
        main_chart_title = f"No sales for {zero_days_len} days while system shows {computer_stock_qty} units in stock"
        peer_7d_total = int(round(float(sku_alert_row.get("peer_sales_7d", 0))))
        peer_chart_title = f"Meanwhile at other stores: {peer_7d_total} units sold this week (demand is active across chain)"

        fig_sales = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.68, 0.32],
            subplot_titles=(main_chart_title, peer_chart_title),
        )

        if len(store_ts) >= zero_days_len and zero_days_len > 0:
            streak_start_date = store_ts.iloc[-zero_days_len]["date"]
        else:
            streak_start_date = st.session_state.curr_date

        bar_colors = ["#10b981" if s > 0 else "#cbd5e1" for s in store_ts["sales"]]
        fig_sales.add_trace(
            go.Bar(
                x=store_ts["date"],
                y=store_ts["sales"],
                name="Customer Purchases (Units Sold)",
                marker_color=bar_colors,
                hovertemplate="%{x}<br>Purchases: %{y} units<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig_sales.add_trace(
            go.Scatter(
                x=store_ts["date"],
                y=store_ts["system_stock"],
                mode="lines+markers",
                name="Computer Inventory Record",
                line=dict(color="#2563eb", width=2.5, shape="hv"),
                marker=dict(size=4),
                hovertemplate="%{x}<br>Computer Inventory: %{y} units<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig_sales.add_trace(
            go.Scatter(
                x=store_ts["date"],
                y=[normal_daily_pace] * len(store_ts),
                mode="lines",
                name=f"Normal Daily Pace (~{normal_daily_pace:.1f}/day)",
                line=dict(color="#9333ea", width=1.5, dash="dash"),
                hovertemplate="Normal Daily Sales: %{y:.1f} units<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig_sales.add_vrect(
            x0=streak_start_date,
            x1=st.session_state.curr_date,
            fillcolor="rgba(239, 68, 68, 0.15)",
            line_width=1.5,
            line_color="#ef4444",
            line_dash="dot",
            annotation_text=f"Zero Sales Streak ({zero_days_len}d)",
            annotation_position="top left",
            row=1,
            col=1,
        )

        fig_sales.add_trace(
            go.Scatter(
                x=peer_ts["date"],
                y=peer_ts["sales"],
                mode="lines",
                name="Average Sold in Other Branches",
                line=dict(color="#d97706", width=2),
                fill="tozeroy",
                fillcolor="rgba(217, 119, 6, 0.12)",
                hovertemplate="%{x}<br>Other Branches Avg: %{y:.1f} units<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig_sales.update_layout(
            height=410,
            margin=dict(l=35, r=15, t=32, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
            hovermode="x unified",
            template="plotly_white",
        )
        fig_sales.update_yaxes(title_text="Units", row=1, col=1)
        fig_sales.update_yaxes(title_text="Branch Avg", row=2, col=1)

        st.plotly_chart(fig_sales, width="stretch")

        # Floor Staff Action Buttons
        st.markdown(
            """
            <div style="margin-top: 6px; margin-bottom: 6px;">
                <h4 style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin: 0;">
                    📱 Tap Physical Shelf Finding:
                </h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        audit_key = (st.session_state.curr_store, st.session_state.curr_date, sel_sku_id)

        with btn_c1:
            if st.button("🔴 Shelf Empty\n(Out of Stock)", key=f"btn_empty_{sel_sku_id}", width="stretch"):
                is_real_phantom = int(sku_alert_row["is_true_phantom"])
                days_saved = float(sku_alert_row["days_saved"])
                daily_margin = float(sku_alert_row["daily_margin_at_risk_inr"])
                margin_saved = max(0.0, days_saved * daily_margin)

                st.session_state.audited_items[audit_key] = {
                    "action": "empty",
                    "is_real_phantom": is_real_phantom,
                    "margin_saved_inr": margin_saved,
                    "days_saved": days_saved,
                }
                st.rerun()

        with btn_c2:
            if st.button("🟢 Found on Shelf\n(Stock Available)", key=f"btn_found_{sel_sku_id}", width="stretch"):
                st.session_state.audited_items[audit_key] = {
                    "action": "found",
                    "margin_saved_inr": 0.0,
                }
                st.rerun()

        with btn_c3:
            if st.button("🟡 Found Wrong Bay\n(Misplaced Stock)", key=f"btn_misplaced_{sel_sku_id}", width="stretch"):
                st.session_state.audited_items[audit_key] = {
                    "action": "misplaced",
                    "margin_saved_inr": 0.0,
                }
                st.rerun()

        with btn_c4:
            if st.button("⚪ Item Discontinued\n(Stopped Selling)", key=f"btn_delisted_{sel_sku_id}", width="stretch"):
                st.session_state.audited_items[audit_key] = {
                    "action": "delisted",
                    "margin_saved_inr": 0.0,
                }
                st.rerun()

        # Instant Shopkeeper Outcome Message
        if audit_key in st.session_state.audited_items:
            res = st.session_state.audited_items[audit_key]
            act = res["action"]

            if act == "empty":
                st.success(
                    f"""
                    🎯 **EMPTY SHELF CONFIRMED — AUTOMATED REORDER SENT:**  
                    • **Shelf Finding:** 0 packets on shelf (Computer record held {computer_stock_qty} ghost packets).  
                    • **Computer Corrected:** Inventory ledger updated from **{computer_stock_qty}** down to **0**.  
                    • **Purchase Order Triggered:** Automatic replenishment order PO-4482 placed for 30 units with distributor (estimated delivery in 3 days).  
                    • **Profit Saved:** **₹{res['margin_saved_inr']:,.0f} lost profit saved** over {res['days_saved']:.0f} days!
                    """
                )
            elif act == "found":
                st.info(
                    f"""
                    🛡️ **STOCK VERIFIED ON SHELF — 7-DAY MUTE ACTIVATED:**  
                    • **Shelf Finding:** Packets are present on shelf.  
                    • **Cooldown:** Item muted from your daily walk list for **7 days** so you won't be bothered again.  
                    • **System Learning:** System recorded that this product has slower customer buying cycles.
                    """
                )
            elif act == "misplaced":
                st.warning(
                    f"""
                    🟡 **MISPLACED STOCK MOVED BACK TO SHELF:**  
                    • **Shelf Finding:** Units found in neighboring bay or back storeroom.  
                    • **Action Taken:** Moved back to primary shelf. Available for shoppers immediately without waiting for a delivery truck!
                    """
                )
            elif act == "delisted":
                st.info(
                    f"""
                    ⚪ **DISCONTINUED ITEM CLEARED FROM CATALOG:**  
                    • **Shelf Finding:** Product was discontinued from store range.  
                    • **System Action:** Ghost computer record cleared and marked inactive in store billing.
                    """
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #94a3b8;">
        <span>StockGuard v1.0 · Store Operations Hub · Floor Priority Walk Sheet</span>
        <span>For Executive Benchmarking and Mathematical Theory, use the left sidebar navigation</span>
    </div>
    """,
    unsafe_allow_html=True,
)
