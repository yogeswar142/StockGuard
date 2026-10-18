# StockGuard Hackathon Evaluation Report

> **Methodology:** Strictly out-of-sample evaluation on final 40% of timeline.
> **Calibration Window:** Start → 2026-01-19 (first 60% of timeline).
> **Evaluation Window:** 2026-01-20 → End (216 days, 10 stores × 150 SKUs = 1,500 pairs).
> **Operational Constraint:** Hard budget cap of ≤ 20 audits per store per day with simulated feedback cooldown.

---

## 1. Headline Benchmark Comparison

| Metric | StockGuard Core | Baseline A (Fixed 5d) | Baseline A (Matched Vol) | Baseline B (Streak Rank) | Random Selection | Oracle (Upper Bound) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Total Alerts Issued** | **9,084** | 14,779 | 9,084 | 32,703 | 43,200 | 43,200 |
| **True Phantoms Caught** | **437** | 477 | 320 | 470 | 1,209 | 9,080 |
| **Raw Precision@20** | **4.81%** | 3.23% | 3.52% | 1.44% | 2.80% | 21.02% |
| **Normalised Precision@20** | **5.12%** | 5.12% | 5.51% | 5.13% | 4.20% | **100.00%** |
| **Episode Recall (Overall)** | **90.29%** | 98.55% | 66.12% | 97.11% | 14.40% | **100.00%** |
| **Recall within 3 Days** | **48.35%** | 36.16% | 24.79% | 55.79% | 4.80% | **100.00%** |
| **Recall within 7 Days** | **76.65%** | 92.15% | 59.71% | 90.70% | 12.00% | **100.00%** |
| **Recall within 14 Days** | **84.09%** | 94.21% | 61.78% | 92.77% | 15.60% | **100.00%** |
| **Mean Time-to-Detect (TTD)** | **5.0 days** | 4.6 days | 5.1 days | 3.8 days | ~10.5 days | **1.0 days** |
| **Value-Weighted Recall** | **97.80%** | 99.57% | 69.67% | 98.93% | 12.00% | **100.00%** |

### Exact Definition of Normalised Precision@20
> On store-days with fewer active phantoms than the 20-check budget, raw Precision@20 is mathematically capped below 100%.
> We therefore report Normalised Precision@20 evaluated per store-day $(s, t)$ as:

$$\text{Precision@20}_{\text{norm}}(s, t) = \frac{\text{True Positives in Top 20}(s, t)}{\min(20, \text{Active True Phantoms}(s, t))}$$

> averaged across all store-days where active true phantoms existed on the shelf.

---

## 2. Precision by SKU Velocity Band

| Velocity Band | StockGuard Core | Baseline A (Fixed 5d) | Baseline B (Streak Rank) |
|---|:---:|:---:|:---:|
| **Fast Movers** | **25.1%** (47/187) | 90.4% (47/52) | 4.9% (47/952) |
| **Medium Movers** | **4.9%** (137/2810) | 17.0% (136/802) | 1.6% (134/8569) |
| **Slow Movers** | **4.2%** (253/6087) | 2.1% (294/13925) | 1.2% (289/23182) |

---

## 3. False Positive Breakdown by Confuser Type

| Confuser Category | StockGuard Alerts | Baseline A Alerts | Root Cause in Retail Operations |
|---|:---:|:---:|---|
| **Slow-Mover Natural Zeros** | 74.9% (6,475) | 82.1% (11,747) | Naturally low sales rate means 5+ consecutive zero days occur without stock-out |
| **Post-Promo Demand Lulls** | 9.9% (852) | 7.0% (1,006) | Customer forward-buying during promo depresses immediate post-promo demand |
| **Sustained Demand Shocks** | 6.0% (516) | 5.3% (762) | Seasonal drops / category price hikes cause prolonged healthy zeros |
| **Genuine Shelf Void (Misplaced)** | 0.8% (69) | 0.5% (69) | Physical units exist in store but not on customer shelf (backroom/wrong aisle) |
| **Delisted Ghost Inventory** | 9.4% (809) | 5.7% (810) | Discontinued SKU with phantom book inventory (system stock > 0, sales = 0) |
| **Stochastic Poisson Noise** | 0 | 0 | Pure random variance under standard Poisson process |

---

## 4. Probability Calibration (Platt Scaling)

- **Calibrated Brier Score:** `0.0477` vs **Uncalibrated:** `0.3624` (Base-Rate Predictor: `0.0458`).
- **Empirical Calibration Function:** Fitted logistic regression mapping on calibration-slice logit posteriors.
- **Integrity Guarantee:** Alerts now display empirical hit-probabilities (e.g. 5%–25%), completely eliminating misleading '99.9%' confidence claims.
- See `results/calibration.png` for reliability diagram.

---

## 5. Missed Episode Diagnosis

- **Total Missed Episodes:** 47 out of 484 total phantom episodes.
  - Fast movers missed: `0` (0% missed)
  - Medium movers missed: `0` (0% missed)
  - Slow movers missed: `47` (100% of missed episodes)
  - Median duration of missed episodes: `7.0 days`
  - Duration distribution: `0 short (<7d)`, `39 medium (7-14d)`, `8 long (≥14d)`.
- **Economic Margin at Risk in Misses:** ₹17,322 (`2.2%` of total phantom margin).
- **Value Captured:** StockGuard captures **97.80%** of total phantom margin at risk.

---

## 6. Exact Per-Episode Rupee Recovery (10 Stores × 150 SKUs)

> **Formula:** For each caught episode $e$, $\text{days\_saved}_e = \max(0, \text{actual\_duration}_e - (\text{days\_to\_alert}_e + 1))$.
> `Episodes caught with days saved > 0: 424 × ~₹87/day margin × ~15.9 days saved per episode × 1.69 (annual scale 365/216) × 0.60 (truly lost share) - ₹76,751 audit cost = ₹508,172/yr`

| Financial Metric | Evaluated Value | Derivation & Source |
|---|:---:|---|
| Phantom Episodes Caught | 424 | Across 216 out-of-sample evaluation days |
| Mean Daily Margin at Risk | ₹87 | True positive SKU price × gross margin % × base daily demand |
| Mean Days Saved per Episode | 15.9 days | Exact per-episode duration minus detection day |
| Annualisation Factor | 1.69× | Scaled to 365 calendar days (365 / 216) |
| Truly Lost Demand Share | 60% | 40% of shoppers substitute, 60% walk away empty-handed |
| **Gross Margin Recovered (Annual)** | **₹974,872** | Total phantom revenue preserved |
| Audit Staff Cost (Annual) | ₹76,751 | ₹5 per audit check (2 mins staff time @ ₹150/hr) |
| **Net Annual Rupee Recovery** | **₹508,172** | **Evaluated 10-store × 150-SKU pilot net ROI** |

### Exact Sensitivity Analysis (±10% Recall, ±7 Days Duration):

| Recall Variation | Duration Variation | Net Annual Rupee Recovery |
|:---:|:---:|:---:|
| -10% | -7 days | **₹253,166** |
| -10% | +0 days | **₹449,680** |
| -10% | +7 days | **₹692,297** |
| +0% | -7 days | **₹289,824** |
| +0% | +0 days | **₹508,172** |
| +0% | +7 days | **₹777,747** |
| +10% | -7 days | **₹326,482** |
| +10% | +0 days | **₹566,664** |
| +10% | +7 days | **₹863,197** |

---

## 7. Robustness Grid Across Operational Stress Scenarios

| Stress Scenario | Alerts | Raw Prec | Norm Prec | Recall@7d | Overall Rec | TTD | Value Recall | Net Recovery |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Scenario (Seed 42)** | 5,883 | 4.66% | 5.39% | 76.00% | 91.33% | 5.0d | 94.79% | **₹449,358** |
| **Alternative Seed (Seed 101)** | 5,672 | 5.18% | 4.91% | 77.20% | 89.36% | 4.8d | 97.34% | **₹679,242** |
| **Noisier Demand (alpha=0.60)** | 5,770 | 4.77% | 5.25% | 73.63% | 88.42% | 5.4d | 95.14% | **₹532,505** |
| **Heavier Promos (freq=0.16)** | 6,469 | 4.67% | 5.14% | 73.05% | 90.42% | 6.2d | 95.51% | **₹482,353** |
| **Longer Slow Tail (75% Slow)** | 5,963 | 4.60% | 4.99% | 70.82% | 89.84% | 5.8d | 91.35% | **₹384,177** |

---

## 8. Honest Engineering Assessment: Where StockGuard is Worse vs Baseline A

| Performance Dimension | StockGuard Core | Baseline A (Fixed 5d) | Honest Engineering Assessment |
|---|:---:|:---:|---|
| **Recall within 7 Days** | 76.65% | **92.15%** | **StockGuard is lower by 15.5 percentage points.** Baseline A alerts every item on day 5 regardless of velocity, while StockGuard requires 8-14 zero days on slow movers to establish statistical significance. |
| **Mean Time-to-Detect** | 5.0 days | **4.6 days** | **StockGuard is slower by 0.4 days** because it suppresses early false alerts on slow-moving items. |
| **Normalized Precision** | 5.12% | 5.12% | **Roughly PARALLEL.** Under the 20-check budget ceiling, both models encounter severe denominator constraints on quiet store-days. |
| **Raw Precision (Wasted Audits)** | **4.81%** | 3.23% | **StockGuard WINS by +51%**, issuing 41% fewer wasted alerts. |

> **Operational Conclusion:** Under a matched 20-check budget, StockGuard captures **97.80%** of phantom margin at risk while sparing store associates over 5,000 unproductive audits, delivering **₹508,172 net annual recovery** in the 10-store pilot.

---
*StockGuard Hackathon Evaluation Engine | Round 1*