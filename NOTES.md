# StockGuard Project Notes & Decision Log

## Initial Setup (Day 1 - 18 Oct 2026)
- **Problem:** Phantom Stock-Out Detector (Hackathon Round 1).
- **Core Plan Source of Truth:** `docs/STOCKGUARD_PLAN.md` (copied from research plan).
- **Git Remote:** `https://github.com/yogeswar142/StockGuard.git` configured on branch `main`.

## Key Decisions & Architectural Principles
1. **Explainable Statistics First:**
   - Expected demand $\lambda$ per SKU-store-day using rolling baseline + weekday/season/promo factors.
   - Zero-streak surprise: $P(\text{streak} \mid \text{in-stock}) = e^{-\lambda N}$ (or Negative Binomial for over-dispersed demand).
   - Per-SKU statistical threshold: avoids drowning in slow-mover false positives by computing $P(\text{streak})$ dynamically rather than using arbitrary global $N$.
   - Bayesian posterior: $P(\text{phantom} \mid \text{streak}, \text{stock} > 0)$ combining prior phantom rate (~3%) with streak likelihood.
   - Corroborating signals: peer-store sales, substitute SKU category lift, flat system stock, and recency of last sale.
2. **Budget-Aware Ranking:**
   - Operational reality: Store manager can only audit 20 shelves per day.
   - Ranking criterion: Expected Value of a Check $= P(\text{phantom}) \times \text{Daily Margin at Risk} \times \text{Expected Days Saved} - \text{Audit Cost}$.
   - Output hard cap: Exactly top 20 alerts per store per day.
3. **Synthetic Dataset Structure:**
   - Dimensions: MVP (10 stores x 150 SKUs x 540 days saved + 60 days discarded burn-in) -> Scaled (25 stores x 800 SKUs x 540 days).
   - Fast / medium / slow mover velocity mix (realistic long tail: ~60% slow, 30% medium, 10% fast).
   - Seasonality, day-of-week effects, promo calendar, and Indian festive peaks (Diwali, Holi, Dussehra, Republic Day).
   - Prior & Hazard Conversion:
     - 3% prevalence represents active state share $P \approx 0.03$.
     - By steady-state kinetics ($P \approx \lambda_{\text{onset}} \times \bar{D}$), with mean undetected duration $\bar{D} = 21$ days (cycle count interval), the daily onset hazard rate is $\lambda_{\text{onset}} = 0.03 / 21 \approx 0.001429$ per SKU-store-day.
     - Episode duration modeled with heavy-tailed skewed distribution (Geometric / Lognormal with mean 21 days).
   - Injected ground truth: phantom episodes with known onset, plus misplaced and delisted cases.
   - Hard Confusers (all with system_stock > 0):
     - Sustained genuine demand drops (season end, price hike) -> labeled `is_demand_shock`
     - Post-promo demand lulls (forward buying dip) -> labeled `is_post_promo_lull`
     - Intermittent slow-mover zero runs -> labeled `is_slow_mover_zero`
     - True stock-outs (system_stock = 0) -> labeled `is_true_stockout`
   - Ground truth labels stored separately from POS input data.
4. **Leakage Prevention & Calibration Split:**
   - Detector strictly does not access ground_truth_labels.
   - 60-day burn-in discarded before saving output files.
   - Calibration/tuning on first 60% of saved timeline (days 1 to 324); final out-of-sample evaluation on remaining 40% (days 325 to 540).
5. **Cooldown & Audit Simulation:**
   - Once a SKU is alerted and audited in the simulation:
     - If true phantom: physical stock corrected to system stock or system stock corrected to 0, triggering reorder.
     - If false positive (stock found): alert enters cooldown for $N_{\text{cooldown}} = 7$ days to prevent list starvation.
6. **Evaluation Metrics:**
   - Raw Precision@20.
   - Normalized Precision@20: $\text{raw\_p@20} / \min(20, \text{active phantoms that day})$.
   - Episode-level Recall within $K$ days: $K \in \{3, 7, 14\}$ days from onset.
   - Time-to-detect (mean days from onset to first alert).
   - Breakdown of false positives by confuser type (`is_demand_shock`, `is_post_promo_lull`, `is_slow_mover_zero`).
7. **Tooling & Tech Stack:**
   - Python, pandas, numpy, scipy, statsmodels/scikit-learn (lightweight), Streamlit, plotly/matplotlib.
   - No heavy uninterpretable deep learning. Every alert must have an English reason and visual proof.

## Open Questions & Assumptions Log
- [Assumption] Target runtime environment: Local Windows Python 3.11 with virtualenv.
- [Assumption] Currency: Indian Rupees (₹).
- [Assumption] Staff audit cost: ~₹150/hr (~₹5/check at 2 min per check).
- [Decision - Step 3b] Calibration Fixes:
  - Added `min_streak_for_peer_boost = 4` to suppress peer-store signal inflation on short 2-day streaks.
  - Reason explanation thresholding: only show peer signal if `peer_sales_7d >= 10`, and substitute lift only if `lift >= 15%` AND `sub_units_7d >= 5`.
  - Tested velocity-dependent thresholding strictly on calibration slice (days 160-323):
    - Lowering threshold for fast/medium movers does NOT improve recall@7d (fast movers already cross 0.50 within 3 days).
    - Lowering medium threshold allowed more false positives into the top-20 budget, dropping precision by 0.5-0.7 percentage points without recall gain.
    - Result: Uniform threshold (0.35) combined with Expected Value ranking is empirically optimal.
- [Decision - Step 4] Value & Benchmark Suite:
  - Implemented budget sweeps (1, 3, 5, 10, 20 checks/store-day) across StockGuard, Baseline A, Baseline A Matched-Volume, Baseline B, Random, and Oracle.
  - Value-Weighted Recall: evaluates share of total economic margin-at-risk caught (`price * margin * lambda * duration`).
  - Annual Rupee Calculation:
    $$\text{Net Annual Recovery} = \text{Caught Episodes} \times \bar{M}_{\text{daily}} \times \bar{D}_{\text{saved}} \times \frac{365}{D_{\text{eval}}} \times 0.60 - \text{Audit Cost}$$
    accounting for 60% truly lost demand and ₹5 audit cost per check.

