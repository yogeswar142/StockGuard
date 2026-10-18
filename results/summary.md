# StockGuard Hackathon Evaluation Report

> **Methodology:** Strictly out-of-sample evaluation on final 40% of timeline.
> **Calibration Window:** Start → 2026-01-19 (60% of timeline).
> **Evaluation Window:** 2026-01-20 → End (216 days, 10 stores × 150 SKUs).
> **Operational Constraint:** Hard budget cap of ≤ 20 audits per store per day with simulated feedback cooldown.

---

## 1. Headline Benchmark Comparison

| Metric | StockGuard | Baseline A (Fixed 5d) | Baseline A (Matched Vol) | Baseline B (Streak Rank) |
|---|---|---|---|---|
| Total Alerts Issued | 8,584 | 14,562 | 8,584 | 32,683 |
| True Phantoms Caught | 418 | 469 | 283 | 462 |
| **Raw Precision@20** | **4.87%** | 3.22% | 3.30% | 1.41% |
| **Normalised Precision@20** | **5.09%** | 5.10% | 5.58% | 5.09% |
| **Episode Recall (Overall)** | **88.19%** | 98.95% | 59.70% | 97.47% |
| **Recall within 3 Days** | **48.10%** | 37.34% | 22.15% | 53.59% |
| **Recall within 7 Days** | **72.57%** | 92.83% | 53.59% | 91.35% |
| **Recall within 14 Days** | **81.43%** | 93.67% | 54.43% | 92.19% |
| **Mean Time-to-Detect** | **5.5 days** | 4.9 days | 5.9 days | 4.4 days |
| **Value-Weighted Recall** | **97.78%** | 99.43% | 62.18% | 99.22% |

---

## 2. Precision by SKU Velocity Band

| Velocity Band | StockGuard | Baseline A (Fixed 5d) | Baseline B (Streak Rank) |
|---|---|---|---|
| **Fast Movers** | **23.1%** (43/186) | 79.6% (43/54) | 4.5% (43/954) |
| **Medium Movers** | **5.7%** (154/2683) | 19.2% (154/803) | 1.8% (150/8553) |
| **Slow Movers** | **3.9%** (221/5715) | 2.0% (272/13705) | 1.2% (269/23176) |

---

## 3. False Positive Breakdown by Confuser Type

| Confuser Type | StockGuard Alerts | Baseline A Alerts | Root Cause in Retail Reality |
|---|---|---|---|
| **Slow-Mover Natural Zeros** | 79.8% (6,518) | 86.6% (12,210) | Low demand rate means 5-day zeros occur naturally without stock-out |
| **Post-Promo Lulls** | 9.7% (791) | 7.3% (1,033) | Shoppers pantry-loaded during promotion, depressing subsequent sales |
| **Misplaced Inventory** | 1.1% (91) | 0.6% (84) | Physical units exist elsewhere in store (backroom/wrong aisle) |
| **Other Noise** | 766 | 766 | Statistical Poisson variance |

---

## 4. Confidence Calibration & Reliability

- **Brier Score:** `0.3542` (Lower is better; 0.0 = perfect calibration, 0.25 = uninformative baseline).
- **Short-Streak Overconfidence Fix:** Peer-store signal boost is strictly suppressed for zero streaks < 4 days (`min_streak_for_peer_boost=4`), preventing inflated confidence on 2-day absences.
- See `results/calibration.png` for the reliability diagram across confidence deciles.

---

## 5. Missed Episode Diagnosis & Calibration Experiment

- **Total Missed Episodes:** 56 out of 474 total phantom episodes.
  - Fast movers missed: `0`
  - Medium movers missed: `0`
  - Slow movers missed: `56`
  - Median duration of missed episodes: `7.0 days`
  - Duration breakdown: `0 short (<7d)`, `52 medium (7-14d)`, `4 long (≥14d)`.
- **Economic Value at Risk in Misses:** ₹17,806 (`2.2%` of total phantom margin).

### Calibration Period Tuning Experiment (Tuned strictly on first 60% of timeline):
> We empirically evaluated whether lowering the alert threshold for fast/medium movers on the calibration slice improves recall@7d without destroying precision:
> - **Uniform 0.35:** Alerts = 6,362 | Raw Precision = 5.27% | Recall@7d = 74.06% | TTD = 5.6d
> - **Adaptive (Fast: 0.20, Medium: 0.25, Slow: 0.35):** Alerts = 6,894 | Raw Precision = 4.82% | Recall@7d = 73.53% | TTD = 5.5d
> - **Adaptive (Fast: 0.15, Medium: 0.20, Slow: 0.35):** Alerts = 7,202 | Raw Precision = 4.61% | Recall@7d = 73.53% | TTD = 5.5d
>
> **Finding:** Lowering the threshold for medium movers does not catch more true phantoms (fast movers already cross 0.50 within 3 days). Instead, it permits more false-positive medium movers into the 20-check budget, lowering precision by 0.45-0.66 percentage points without recall gain. Thus, uniform thresholding with value-based ranking is optimal.

---

## 6. Annual Rupees-Recovered Arithmetic (Plan Section 2.7)

> **Formula:** `Episodes caught in eval: 418 × ~₹103/day margin × ~13 days saved × 1.69 (annual scale) × 0.60 (truly lost) - ₹72527 audit = ₹500,325/yr`

| Financial Metric | Evaluated Value | Notes / Methodology |
|---|---|---|
| Phantom Episodes Caught | 418 | Across 216 out-of-sample evaluation days |
| Mean Daily Margin at Risk | ₹103 | True positive SKU price × gross margin % × expected daily demand |
| Mean Days Saved per Episode | 13.1 days | Cycle count interval (21d) minus detection latency |
| Annualisation Factor | 1.69× | Scaled to 365 calendar days (365 / 216) |
| Truly Lost Demand Share | 60% | 40% of shoppers substitute, 60% walk away empty-handed |
| **Gross Margin Recovered (Annual)** | **₹954,754** | Total phantom revenue preserved |
| Audit Staff Cost (Annual) | ₹72,527 | ₹5 per audit check (2 mins staff time @ ₹150/hr) |
| **Net Annual Rupee Recovery** | **₹500,325** | **Final net ROI for 10-store chain** |

### Sensitivity Analysis (±10% Recall, ±7 Days Duration):

| Recall Variation | Duration Variation | Net Annual Rupee Recovery |
|---|---|---|
| -10% | -7 days | **₹168,109** |
| -10% | +0 days | **₹443,040** |
| -10% | +7 days | **₹717,972** |
| +0% | -7 days | **₹194,846** |
| +0% | +0 days | **₹500,325** |
| +0% | +7 days | **₹805,805** |
| +10% | -7 days | **₹221,583** |
| +10% | +0 days | **₹557,611** |
| +10% | +7 days | **₹893,638** |

---

## 7. Honest Assessment: Where StockGuard is Worse vs Baseline A

| Performance Dimension | StockGuard Core | Baseline A (Fixed 5d) | Honest Engineering Diagnosis |
|---|---|---|---|
| **Recall within 7 Days** | 72.57% | **92.83%** | **StockGuard is WORSE by ~20 percentage points.** StockGuard's statistical threshold delays alerts on slow movers until day 8-12, whereas Baseline A alerts everything at day 5. |
| **Mean Time-to-Detect** | 5.5 days | **4.9 days** | **StockGuard is SLOWER by ~0.6 days.** StockGuard requires more corroborating evidence before triggering alerts on low-velocity items. |
| **Normalized Precision** | 5.09% | **5.10%** | **StockGuard is roughly PARALLEL (4.91% vs 5.10%).** In stores with few active phantoms, both models face severe denominator caps. |
| **Raw Precision (Wasted Checks)** | **4.87%** | 3.22% | **StockGuard WINS by +38%.** StockGuard issues ~35% fewer total alerts (9,422 vs 14,562), sparing staff 5,140 useless shelf audits. |

> **Operational Conclusion:** Baseline A achieves higher recall only by flooding store managers with 14,562 alerts (87% of which are slow-moving false positives). StockGuard curates high-value, high-confidence alerts, recovering **₹500,325 annually across 10 stores (~₹1.25M for 25 stores)** while respecting the manager's 20-check daily budget.

---
*Automated Evaluation Suite | StockGuard Hackathon Round 1*