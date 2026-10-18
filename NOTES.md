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
   - Dimensions: 800 SKUs × 25 stores × 18 months daily data (~1.1M records or optimized representation).
   - Fast / medium / slow mover velocity mix (realistic long tail: ~60% slow, 30% medium, 10% fast).
   - Seasonality, day-of-week effects, promo calendar, and Indian festive peaks (Diwali, etc.).
   - Injected ground truth: ~3% phantom stock-out episodes (shelf empty, system stock > 0, sales frozen).
   - Real-world confusers: true stock-outs (system stock = 0), natural zero sales runs on slow movers, post-promo demand lulls, and regional holiday dips.
   - Stretch cases: misplaced inventory (slow trickle / adjacent bay) and delisted/discontinued lines.
   - Ground truth labels stored separately from POS input data.
4. **Tooling & Tech Stack:**
   - Python, pandas, numpy, scipy, statsmodels/scikit-learn (lightweight), Streamlit, plotly/matplotlib.
   - No heavy uninterpretable deep learning. Every alert must have an English reason and visual proof.

## Open Questions & Assumptions Log
- [Assumption] Target runtime environment: Local Windows Python 3.11 with virtualenv.
- [Assumption] Currency: Indian Rupees (₹).
- [Assumption] Staff audit cost: ~₹150/hr (~₹5/check at 2 min per check).
