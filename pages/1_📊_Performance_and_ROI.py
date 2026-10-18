"""
StockGuard: Executive Benchmarking & ROI Suite
Detailed out-of-sample performance benchmarks, budget sweeps, calibration, and ROI simulator.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="StockGuard | Executive Benchmark & ROI",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 98%;
    }
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 10px;
    }
    .benchmark-table th {
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 600;
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid #cbd5e1;
    }
    .benchmark-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a;
    }
    .benchmark-table tr.highlight-row {
        background-color: #f0fdf4;
        font-weight: 600;
    }
    .callout-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #14532d;
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open("results/metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics


metrics = load_metrics()

# Header
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 2.0rem;">📊</span>
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #0f172a;">
                Executive Performance Benchmark & Financial Impact
            </h1>
            <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.88rem;">
                Rigorous empirical comparison against retail industry heuristics, random audits, and oracle on the out-of-sample evaluation period (216 days).
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

# Headline Benchmark Table
m_sg = metrics["stockguard"]
m_ba = metrics["baseline_a_fixed_5d"]
m_bam = metrics["baseline_a_matched_volume"]
m_bb = metrics["baseline_b_streak_len"]
m_rnd = metrics["random"]
m_orc = metrics["oracle"]

benchmark_html = f"""
<table class="benchmark-table">
    <thead>
        <tr>
            <th>Model / Heuristic</th>
            <th>Daily Ranking Logic</th>
            <th>Alerts Issued</th>
            <th>True Phantoms</th>
            <th>Raw Precision@20</th>
            <th>Norm Precision@20</th>
            <th>Recall @ 7d</th>
            <th>Overall Recall</th>
            <th>Mean TTD</th>
            <th>Value-Weighted Recall</th>
        </tr>
    </thead>
    <tbody>
        <tr class="highlight-row">
            <td><strong>StockGuard (Ours)</strong></td>
            <td>Bayesian + Expected Value (₹)</td>
            <td>{m_sg['total_alerts']:,}</td>
            <td>{m_sg['total_true_positives']:,}</td>
            <td><strong>{m_sg['raw_precision_at_20']*100:.2f}%</strong></td>
            <td><strong>{m_sg['norm_precision_at_20']*100:.2f}%</strong></td>
            <td>{m_sg['episode_recall_7d']*100:.2f}%</td>
            <td>{m_sg['episode_recall_overall']*100:.2f}%</td>
            <td>{m_sg['mean_time_to_detect_days']:.1f} days</td>
            <td><strong>{m_sg['value_weighted_recall']*100:.2f}%</strong></td>
        </tr>
        <tr>
            <td>Baseline A (Fixed 5-Day)</td>
            <td>5+ Zero Days (FIFO)</td>
            <td>{m_ba['total_alerts']:,}</td>
            <td>{m_ba['total_true_positives']:,}</td>
            <td>{m_ba['raw_precision_at_20']*100:.2f}%</td>
            <td>{m_ba['norm_precision_at_20']*100:.2f}%</td>
            <td>{m_ba['episode_recall_7d']*100:.2f}%</td>
            <td>{m_ba['episode_recall_overall']*100:.2f}%</td>
            <td>{m_ba['mean_time_to_detect_days']:.1f} days</td>
            <td>{m_ba['value_weighted_recall']*100:.2f}%</td>
        </tr>
        <tr style="background-color: #fefce8;">
            <td>Baseline A (Matched Volume)</td>
            <td>Fixed 5-Day Truncated to 9,084 Alerts</td>
            <td>{m_bam['total_alerts']:,}</td>
            <td>{m_bam['total_true_positives']:,}</td>
            <td>{m_bam['raw_precision_at_20']*100:.2f}%</td>
            <td>{m_bam['norm_precision_at_20']*100:.2f}%</td>
            <td>{m_bam['episode_recall_7d']*100:.2f}%</td>
            <td>{m_bam['episode_recall_overall']*100:.2f}%</td>
            <td>{m_bam['mean_time_to_detect_days']:.1f} days</td>
            <td>{m_bam['value_weighted_recall']*100:.2f}%</td>
        </tr>
        <tr>
            <td>Baseline B (Streak Rank)</td>
            <td>Zero-Streak Length (Longest First)</td>
            <td>{m_bb['total_alerts']:,}</td>
            <td>{m_bb['total_true_positives']:,}</td>
            <td>{m_bb['raw_precision_at_20']*100:.2f}%</td>
            <td>{m_bb['norm_precision_at_20']*100:.2f}%</td>
            <td>{m_bb['episode_recall_7d']*100:.2f}%</td>
            <td>{m_bb['episode_recall_overall']*100:.2f}%</td>
            <td>{m_bb['mean_time_to_detect_days']:.1f} days</td>
            <td>{m_bb['value_weighted_recall']*100:.2f}%</td>
        </tr>
        <tr style="color: #64748b;">
            <td>Random Selection</td>
            <td>Random Uniform from positive stock</td>
            <td>{m_rnd['total_alerts']:,}</td>
            <td>{m_rnd['total_tp']:,}</td>
            <td>{m_rnd['raw_precision']*100:.2f}%</td>
            <td>{m_rnd['norm_precision']*100:.2f}%</td>
            <td>{m_rnd['recall_7d']*100:.2f}%</td>
            <td>{m_rnd['recall_overall']*100:.2f}%</td>
            <td>—</td>
            <td>{m_rnd['value_weighted_recall']*100:.2f}%</td>
        </tr>
        <tr style="color: #64748b;">
            <td>Theoretical Oracle</td>
            <td>Perfect Knowledge of Active Phantoms</td>
            <td>{m_orc['total_alerts']:,}</td>
            <td>{m_orc['total_tp']:,}</td>
            <td>{m_orc['raw_precision']*100:.2f}%</td>
            <td>100.00%</td>
            <td>{m_orc['recall_7d']*100:.2f}%</td>
            <td>100.00%</td>
            <td>1.0 day</td>
            <td>100.00%</td>
        </tr>
    </tbody>
</table>
"""
st.markdown(benchmark_html, unsafe_allow_html=True)

st.markdown(
    """
    <div class="callout-box">
        <strong>💡 Key Empirical Proof (Matched Audit Volume):</strong><br>
        When Baseline A is restricted to the <strong>exact same 9,084 audit budget</strong> that StockGuard used, its overall recall collapses from 98.55% to <strong>66.12%</strong> and its value-weighted recall drops to <strong>69.67%</strong>.  
        StockGuard catches <strong>+36.6% more phantoms (437 vs 320)</strong> and recovers <strong>+40.4% more margin (97.80% vs 69.67%)</strong> at the exact same labor budget!
    </div>
    """,
    unsafe_allow_html=True,
)

# Benchmark Diagnostic Charts
b_col1, b_col2 = st.columns(2)
with b_col1:
    st.markdown("#### 1. Precision & Recall vs Daily Audit Budget")
    b_sweep = metrics["budget_sweep"]
    budgets = [1, 3, 5, 10, 20]

    sg_p = [b_sweep["stockguard"][str(b)]["raw_precision"] * 100 for b in budgets]
    sg_vwr = [b_sweep["stockguard"][str(b)]["value_weighted_recall"] * 100 for b in budgets]
    ba_p = [b_sweep["baseline_a"][str(b)]["raw_precision"] * 100 for b in budgets]
    ba_vwr = [b_sweep["baseline_a"][str(b)]["value_weighted_recall"] * 100 for b in budgets]

    fig_budget = go.Figure()
    fig_budget.add_trace(go.Scatter(x=budgets, y=sg_p, mode="lines+markers", name="StockGuard Precision (%)", line=dict(color="#16a34a", width=2.5)))
    fig_budget.add_trace(go.Scatter(x=budgets, y=ba_p, mode="lines+markers", name="Baseline A Precision (%)", line=dict(color="#dc2626", width=2, dash="dash")))
    fig_budget.add_trace(go.Scatter(x=budgets, y=sg_vwr, mode="lines+markers", name="StockGuard Value Recall (%)", line=dict(color="#2563eb", width=2.5)))
    fig_budget.add_trace(go.Scatter(x=budgets, y=ba_vwr, mode="lines+markers", name="Baseline A Value Recall (%)", line=dict(color="#9333ea", width=2, dash="dash")))

    fig_budget.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=20),
        xaxis_title="Daily Audit Budget Cap (Checks/Store-Day)",
        yaxis_title="Percentage (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    st.plotly_chart(fig_budget, width="stretch")

with b_col2:
    st.markdown("#### 2. Probability Calibration (Reliability Diagram)")
    cal = metrics["calibration"]
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(color="#94a3b8", dash="dash")))
    fig_cal.add_trace(
        go.Scatter(
            x=cal["bin_mean_calibrated"][:3],
            y=cal["bin_fraction_positive"][:3],
            mode="lines+markers",
            name="StockGuard Platt-Calibrated",
            line=dict(color="#16a34a", width=2.5),
            marker=dict(size=8),
        )
    )
    fig_cal.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=20),
        xaxis_title="Predicted Probability P(Phantom)",
        yaxis_title="Observed Fraction of Phantoms",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        xaxis=dict(range=[0, 0.30]),
        yaxis=dict(range=[0, 0.30]),
    )
    st.plotly_chart(fig_cal, width="stretch")

# Confuser Breakdown & Cumulative Margin Charts
cf_col1, cf_col2 = st.columns(2)
with cf_col1:
    st.markdown("#### 3. False Positive Breakdown by Confuser Type")
    fp_data = m_sg["fp_breakdown"]
    labels = [
        f"Slow-Mover Zeros ({fp_data['slow_mover_share_pct']}%)",
        f"Post-Promo Lulls ({fp_data['post_promo_share_pct']}%)",
        f"Delisted Ghost Stock ({fp_data['delisted_share_pct']}%)",
        f"Demand Shocks ({fp_data['demand_shock_share_pct']}%)",
        f"Genuine Shelf Void / Misplaced ({fp_data['misplaced_share_pct']}%)",
    ]
    values = [
        fp_data["slow_mover_zeros"],
        fp_data["post_promo_lulls"],
        fp_data["delisted_ghost_inventory"],
        fp_data["demand_shocks"],
        fp_data["genuine_shelf_void_misplaced"],
    ]
    colors = ["#94a3b8", "#f59e0b", "#6366f1", "#ef4444", "#10b981"]

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="percent",
                hovertemplate="%{label}<br>Count: %{value:,} alerts<extra></extra>",
            )
        ]
    )
    fig_donut.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5, font=dict(size=9)),
    )
    st.plotly_chart(fig_donut, width="stretch")

with cf_col2:
    st.markdown("#### 4. Cumulative Margin Recovered vs Audit Volume")
    audit_vols = [b_sweep["stockguard"][str(b)]["total_alerts"] for b in budgets]
    sg_margin_eval = [b_sweep["stockguard"][str(b)]["value_weighted_recall"] * 576911.0 for b in budgets]
    ba_margin_eval = [b_sweep["baseline_a"][str(b)]["value_weighted_recall"] * 576911.0 for b in budgets]

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=audit_vols, y=sg_margin_eval, mode="lines+markers", name="StockGuard Margin (₹)", line=dict(color="#16a34a", width=2.5)))
    fig_cum.add_trace(go.Scatter(x=[b_sweep["baseline_a"][str(b)]["total_alerts"] for b in budgets], y=ba_margin_eval, mode="lines+markers", name="Baseline A Margin (₹)", line=dict(color="#dc2626", width=2, dash="dash")))

    fig_cum.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=10, b=20),
        xaxis_title="Cumulative Shelf Audits Performed",
        yaxis_title="Margin Recovered (₹ Eval Period)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    st.plotly_chart(fig_cum, width="stretch")

# Interactive Rupee Recovery Simulator
st.markdown("---")
st.markdown("### 💰 Chain-Wide Financial ROI Simulator")
st.markdown("Simulate net annual bottom-line returns under varying store network and labor cost assumptions:")

sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
with sim_c1:
    sim_stores = st.slider("Store Count", min_value=5, max_value=200, value=10, step=5)
with sim_c2:
    sim_skus = st.slider("Assortment per Store (SKUs)", min_value=100, max_value=1000, value=150, step=50)
with sim_c3:
    sim_wage = st.slider("Staff Wage (₹/hour)", min_value=100, max_value=300, value=150, step=10)
with sim_c4:
    sim_lost_share = st.slider("Share Truly Lost (%)", min_value=30, max_value=100, value=60, step=5) / 100.0

scale_factor = (sim_stores * sim_skus) / (10 * 150)
annual_episodes = 484 * (365 / 216) * scale_factor
caught_episodes = annual_episodes * (m_sg["episode_recall_overall"])
days_saved_ep = 15.9
margin_per_day = 87.1

sim_gross_margin = caught_episodes * days_saved_ep * margin_per_day * sim_lost_share
sim_checks_per_year = sim_stores * 20 * 365
sim_staff_hours = sim_checks_per_year * (2.0 / 60.0)
sim_audit_cost = sim_staff_hours * sim_wage
sim_net_roi = sim_gross_margin - sim_audit_cost

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("Gross Margin Recovered", f"₹{sim_gross_margin:,.0f}")
with r2:
    st.metric("Annual Staff Audit Cost", f"₹{sim_audit_cost:,.0f}")
with r3:
    st.metric("Net Annual Bottom-Line ROI", f"₹{sim_net_roi:,.0f}")
with r4:
    ratio = sim_gross_margin / max(1.0, sim_audit_cost)
    st.metric("Return on Audit Spend", f"{ratio:.1f}x ROI")
