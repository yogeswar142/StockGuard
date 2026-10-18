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
   - Dimensions: MVP (10 stores x 150 SKUs x 6 months) -> Scaled (25 stores x 800 SKUs x 18 months).
   - Fast / medium / slow mover velocity mix (realistic long tail: ~60% slow, 30% medium, 10% fast).
   - Seasonality, day-of-week effects, promo calendar, and Indian festive peaks (Diwali, etc.).
   - Prior & Hazard Conversion:
     - 3% prevalence represents active state share $P \approx 0.03$.
     - By steady-state kinetics ($P \approx \lambda_{\text{onset}} \times \bar{D}$), with mean undetected duration $\bar{D} = 21$ days (cycle count interval), the daily onset hazard rate is $\lambda_{\text{onset}} = 0.03 / 21 \approx 0.001429$ per SKU-store-day (~1 episode per 700 SKU-store-days).
     - Episode duration modeled with mean 21 days (unless resolved earlier by audit/replenishment).
   - Injected ground truth: phantom episodes with known onset, plus misplaced and delisted cases.
   - Real-world confusers: true stock-outs (system stock = 0), natural slow-mover zero runs, post-promo demand lulls.
   - Ground truth labels stored separately from POS input data.
4. **Leakage Prevention & Calibration Split:**
   - Detector strictly does not access ground_truth_labels.
   - Calibration/tuning on first ~60% of timeline; final evaluation on remaining ~40%.
5. **Cooldown & Audit Simulation:**
   - Once a SKU is alerted and audited in the simulation:
     - If true phantom: physical stock corrected to system stock or system stock corrected to 0, triggering reorder.
     - If false positive (stock found): alert enters cooldown for $N_{\text{cooldown}} = 7$ days to prevent list starvation.
6. **Evaluation Metrics:**
   - Raw Precision@20.
   - Normalized Precision@20: $\text{raw\_p@20} / \min(20, \text{active phantoms that day})$.
   - Episode-level Recall within $K$ days: $K \in \{3, 7, 14\}$ days from onset.
   - Time-to-detect (mean days from onset to first alert).
7. **Tooling & Tech Stack:**
   - Python, pandas, numpy, scipy, statsmodels/scikit-learn (lightweight), Streamlit, plotly/matplotlib.
   - No heavy uninterpretable deep learning. Every alert must have an English reason and visual proof.

## Open Questions & Assumptions Log
- [Assumption] Target runtime environment: Local Windows Python 3.11 with virtualenv.
- [Assumption] Currency: Indian Rupees (₹).
- [Assumption] Staff audit cost: ~₹150/hr (~₹5/check at 2 min per check).
