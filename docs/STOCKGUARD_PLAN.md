# StockGuard: Research & Round 1 Plan

*Phantom Stock-Out Detector (Problem R1) · Research brief, solution design, differentiation, citations and video plan · Prepared 18 Oct 2026*

## 0. One-Page Summary

**The problem in one line:** The system says a shelf holds 6 units, so nothing gets reordered and the item never shows as out of stock, but the units are gone. Sales quietly stay at zero for weeks and nobody notices.

**What StockGuard does:** It reads only daily sales and system on-hand data, works out how *unlikely* each zero-sales streak is given that item's normal demand, and gives each store manager a ranked "go check these 20 shelves" list. Every alert has a plain-English reason and a sales-vs-stock chart.

**Why it can win:** It is built around the manager's real constraint (20 checks per store per day), it explains itself, it learns from the manager's check results, and it separates three failure types (phantom, misplaced, delisted). It also needs no hardware, which the robot and RFID alternatives do.

**Honest caveat to keep in the pitch:** In Round 1 no data is provided, so all results come from a synthetic dataset we generate ourselves. Say so openly and show the system holds up when we vary the generator's assumptions.

## 1. The Real Problem: What the Research Shows

### 1.1 Store inventory records are wrong far more often than people assume

- The most-cited study (DeHoratius & Raman, *Management Science*, 2008) examined nearly 370,000 inventory records across 37 stores of one retailer and found **65% of records were inaccurate**. Record accuracy varied mostly by product category (26.4% of variance), and much less by store (2.7%). \[S1\]
- An earlier Berkeley/Harvard paper reports that at one leading retailer 65% of inventory records were wrong, and that these execution problems **reduce profits by more than 10%**. It also found misplaced SKUs prevented one in six customer requests for help from being fulfilled at another retailer. \[S2\]
- Later work reports similar patterns: over 60% of SKUs with record errors in a European store, and 55% inaccuracy in a study of 166 items across 121 stores (Gruen & Corsten, 2007, as summarised by Auburn's RFID Lab). \[S3\]\[S4\]
- A 2025 arXiv study of grocery retail found about 65% of the SKUs analysed were affected by inventory record inaccuracy. \[S5\]

**Why this matters for StockGuard:** The judges' brief calls phantom stock "one of retail's most expensive invisible leaks". This literature shows it is not an edge case. It is the normal state of store data.

### 1.2 When the record says "in stock" but the shelf is empty, the system is frozen

Supply-chain research calls this **"freezing"**: the system record is higher than physical stock, so the replenishment system sees stock, does not reorder, and the item stays unavailable. \[S6\] The MIT thesis describes the same trap: a phantom inventory stays unnoticed unless a restock, audit or customer inquiry happens to coincide with the real stock-out. \[S7\]

### 1.3 Out-of-stocks cost real money and real customers

- **IHL Group (2026):** Inventory distortion (out-of-stocks plus overstocks) costs about **$1.7 trillion a year, 6.2% of global retail sales**. Out-of-stocks are roughly $1.2 trillion of that. IHL's definition of an out-of-stock explicitly includes stock the system says exists but nobody can locate. \[S8\]\[S9\]
- **Global out-of-stock rate:** A worldwide study (Gruen, Corsten & Bharadwaj, 2002) found an average out-of-stock rate of **8.3%**, with typical retailers losing about **4% of sales** from shoppers not finding the product. When shoppers hit an out-of-stock, on average 31% go to another store and 26% switch brand. \[S10\]\[S11\]
- **Where it's fixable:** The same study found 70–75% of out-of-stocks are triggered at the retail level, and the 2003 follow-up says the place to tackle it is the shop floor. \[S10\]\[S11\]
- **Phantom-specific estimates (vendor-sourced, treat with care):** Simbe cites ECR research that phantom inventory can reduce sales by as much as 3%. RELEX cites a $400 billion annual global loss and says up to 60% of retail inventory records are inaccurate. \[S12\]\[S13\]\[S14\]

### 1.4 The India angle

- India has historically topped global retail shrinkage tables: the Global Retail Theft Barometer put India at 3.1–3.2% of sales in 2008–09, with shoplifting the largest share and employee theft second. Those figures are old and the survey method has been criticised, so use them only as direction. \[S15\]\[S16\]
- More recent signal: Tata's Trent reported shrinkage rising from 0.22% of sales in FY23 to 0.41% in FY24 as volumes grew. Industry executives link shrink to festive seasons, IPL and inflation. The Retailers Association of India CEO says up to 0.5% is manageable versus 1.5–2% globally. \[S17\]
- Note the nuance: shrinkage (theft, damage, vendor fraud, errors) is the *cause* side. Phantom stock-out is the *consequence* we detect: the record has not caught up with the loss.

### 1.5 What real retailers already do, and why it falls short

| Approach | Strength | Weakness |
| --- | --- | --- |
| Periodic cycle counts / audits | Direct and accurate | Slow, labour heavy; one review notes no major breakthroughs in the technique in 30 years \[S6\] |
| RFID | Walmart-commissioned study found it reduces inaccuracy \[S4\] | Needs tagging, readers and capex; not realistic for most Indian mid-size chains |
| Shelf-scanning robots (Simbe Tally) | Continuous shelf truth \[S12\] | Hardware cost, store layout limits |
| Enterprise forecasting suites (RELEX True Inventory) | Anomaly detection plus prioritised recount lists \[S13\]\[S14\] | Enterprise licence, heavy integration |
| Invent.ai phantom inventory app | Computes probability of zero sales over consecutive days and sends daily store alerts \[S18\] | Commercial product, closed method |

**Takeaway:** The problem is real and recognised, the core method (probability of a zero-sales run) is validated by both industry and academia, and there is a clear gap for a lightweight, transparent, budget-aware tool that needs no hardware.

## 2. The Solution We Can Provide

### 2.1 Product concept

StockGuard is a daily "shelf-check list" engine. Input: daily sales per SKU-store, system on-hand, promo calendar, store master. Output: for each store, a ranked list of at most 20 SKUs to physically verify, each with a confidence score, a rupee-at-risk value and a one-line reason.

### 2.2 Core method (explainable statistics first, ML second)

**Step 1: Expected demand (λ) per SKU-store-day.** Use a rolling baseline (say last 8–12 weeks), adjusted for weekday, season, promotion and holiday effects. For lumpy or slow items use a negative binomial rather than Poisson, since retail demand is over-dispersed.

**Step 2: Surprise of the zero streak.** Under Poisson, the chance of N zero-sales days by pure luck is e^(−λN). Examples:

- Fast item, λ = 2/day, 3 zero days: e^(−6) ≈ 0.25%. Very suspicious.
- Slow item, λ = 0.1/day, 30 zero days: e^(−3) ≈ 5%. Not suspicious. This is how the model avoids drowning in slow-mover false positives.

This lets us *justify N per SKU* instead of picking one global number, which directly answers the brief's "choose N and justify it".

**Step 3: Bayesian posterior.** Combine that likelihood with a prior phantom rate (about 3% in the brief, and much lower per day) to get P(phantom | zero streak, stock > 0). The MIT thesis "Detection of phantom inventories at retail stores using a Bayesian approach" uses exactly this idea with negative binomial demand and consecutive zero-sales days. We cite it as academic grounding and extend it. \[S7\]

**Step 4: Corroborating evidence (what makes it better than a pure zero-run test).**

- *Peer-store signal:* the same SKU still sells normally in similar stores while this store is silent.
- *Substitute lift:* sibling SKUs in the same category sell more than usual in this store when the target SKU vanishes (demand moving to the next best option).
- *Stock-never-moved signal:* system stock stays flat, with no replenishment and no sales.
- *Last-sale recency and prior history* for this SKU-store (past corrections make future ones more likely).

**Step 5: Rank under the 20-check budget.** Do not rank by probability alone. Rank by **expected value of a check**: P(phantom) × daily margin at risk × expected days saved, minus a small check-cost term. Take the top 20 per store per day. Break ties in favour of high-value, fast-moving items.

**Step 6: Plain-English reason.** Template example: "Usually sells \~3/day (about ₹1,200/day). Zero sales for 9 days. System shows 6 in stock. Nearby stores sold 41 units of this item this week."

### 2.3 The manager-facing screen (Streamlit or notebook)

- Left: today's top-20 list per store (store, SKU, confidence, ₹ at risk, reason).
- Right: for the selected SKU, a chart of daily sales (bars) vs system stock (line), the expected-sales band, and the shaded zero streak, with peer-store sales beneath. This is what makes the manager trust the alert.
- A one-tap result button after the check: **Found stock / Shelf empty / Wrong place / Discontinued.** This feeds Step 7.

### 2.4 Closed-loop learning (Step 7)

Each check result becomes a label. Retrain or recalibrate weekly so the confidence score becomes a real probability, and show a calibration plot in the demo. This also addresses the "no store-level camera, RFID or staff data" assumption: the manager's check *is* the sensor.

### 2.5 Stretch: tell the three failure types apart by signal shape

| Type | Signal shape | Suggested action |
| --- | --- | --- |
| Phantom stock-out (lost, stolen, damaged) | Abrupt drop to zero while stock stays positive; peers keep selling | Count, correct system, trigger reorder |
| Misplaced on wrong shelf | Sales dip but not to zero, or intermittent trickle; stock never moves; backroom or adjacent bay suspected | Search backroom and neighbouring bays first |
| Delisted or discontinued | Zero at many stores at once; no replenishment; stock decays or freezes | Check range plan, do not reorder, mark inactive |

### 2.6 Evaluation (matches the brief's evaluate.py)

- **Precision@20 per store-day:** among the 20 flagged, how many are real phantoms.
- **Recall on hidden labels:** share of true phantom episodes caught.
- **Time-to-detect:** days from true onset to first appearance in a top-20 list. This is the number that drives rupees recovered.
- **Baseline comparison:** show StockGuard against (a) fixed "N zero days" rule, (b) rank by zero-streak length. Beating these baselines is the proof.
- **Robustness grid:** repeat on synthetic data with noisier demand, more slow movers, heavier promos.

### 2.7 Annual rupees recovered (template with worked arithmetic)

*All inputs below are **illustrative assumptions**. Replace them with outputs from our own simulator before presenting, and show the assumptions on the slide.*

| Input | Assumed value |
| --- | --- |
| Stores × SKUs | 25 × 800 = 20,000 SKU-store pairs |
| Phantom episodes per year (chain) | 1,500 |
| Average demand while phantom | 2 units/day |
| Average selling price | ₹400 |
| Gross margin | 30% |
| Days a phantom lasts undetected today | 21 days |
| Days until StockGuard list + check | 5 days |
| Share of episodes StockGuard catches (recall) | 60% |
| Share of lost sales truly lost (not substituted or recovered) | 60% (the global study found about 31% of shoppers go to another store and 26% switch brand, so some value is recaptured elsewhere) \[S10\] |

**Margin lost per phantom-day** = 2 × ₹400 × 30% = **₹240** **Days saved per caught episode** = 21 − 5 = **16** **Episodes caught** = 1,500 × 60% = **900** **Gross recovery** = 900 × 16 × ₹240 = **₹34.56 lakh** **After the "truly lost" factor** = ₹34.56 lakh × 60% = **≈ ₹20.7 lakh**

**Cost side:** 20 checks × 25 stores × 365 days = 182,500 checks. At 2 minutes each that is about 6,083 staff-hours. At ₹150/hour = **≈ ₹9.1 lakh**.

**Net annual benefit ≈ ₹11.6 lakh** on this small 25-store chain. Also show the sensitivity: what if recall is 40%, or the phantom lasts 14 days? Judges respond well to a visible range and a stated cost of operating the tool. Also mention the unquantified extras: fixed replenishment signal, cleaner forecasts, and fewer false "in stock" promises online.

## 3. How to Be Different From Other Solutions

Most teams on R1 will do "N zero days with stock > 0, sort by days." Here is how StockGuard stands apart.

1. **Budget-aware ranking.** We optimise the 20-slot list as a value problem (expected margin recovered per check), not a probability sort. The brief says the manager can verify only 20; we make that the design centre.
2. **Per-SKU statistical threshold.** No global N. Slow movers are handled by the maths, not patched with rules.
3. **Three-class diagnosis.** Phantom vs misplaced vs delisted, each with a different recommended action. Competitors' public descriptions focus on phantom only. \[S13\]\[S18\]
4. **Peer-store and substitute-SKU evidence.** Uses signals most simple solutions ignore and gives the manager a stronger reason to trust an alert.
5. **Explainability as a feature.** Plain-English reason plus a chart for every alert. Managers act on alerts they understand.
6. **Closed-loop learning.** The manager's check result becomes training data; show the calibration curve improving over simulated weeks.
7. **No hardware.** Works with sales and stock data any POS already has. Position it clearly against RFID, robots and enterprise suites. \[S4\]\[S12\]\[S13\]
8. **Rupee-first story.** Every alert carries a ₹ figure and the pitch ends on net recovered rupees with the arithmetic shown.
9. **Radical honesty about synthetic data.** State generator assumptions, run a robustness grid, and list limitations. This is rare and credible.
10. **India-ready touches (optional):** festive-season and IPL-aware demand adjustments, and a mobile-first checklist (a simple WhatsApp-style or phone web view), since shrinkage in India rises around these periods. \[S17\]

**One-line positioning:** *"StockGuard is the shelf-check list a store manager can actually trust: ranked by rupees, explained in plain English, and it gets smarter every morning."*

## 4. Best Plan for Round 1 (No Data Provided)

### 4.1 Build order (do these in order; stop when time runs out)

1. **Synthetic generator (day 1).** 800 SKUs × 25 stores × 18 months. Include weekday and seasonal multipliers, promo lifts, a fast/medium/slow mix (many slow movers), and store sizes. Inject \~3% phantom episodes with known onset dates. Also inject a few misplaced and delisted cases for the stretch. Save the labels as the hidden ground truth.
2. **Baselines (day 1–2).** Fixed-N rule and streak-length ranking.
3. **StockGuard core (day 2–3).** Demand baseline, zero-run probability, Bayesian posterior, value-based top-20.
4. **evaluate.py (day 3).** Precision@20 per store-day, recall, time-to-detect, plus baseline comparison.
5. **Dashboard (day 3–4).** Streamlit or notebook with the sales-vs-stock chart.
6. **Rupee estimate (day 4).** Fill the table in section 2.7 from simulator output.
7. **Stretch (if time).** Three-class diagnosis; feedback loop demo.
8. **Record the video (last day).** Leave a full day for it.

### 4.2 Risks and how to handle them

| Risk | Mitigation |
| --- | --- |
| Judges distrust self-generated data | Be open about it; show robustness grid; note that the brief's own data is also synthetic |
| Circular result (we detect what we planted) | Plant phantoms with several different shapes; add confusers such as legitimate stock-outs, delistings and holidays that cause real zeros |
| Slow movers inflate false positives | Per-SKU probability threshold; show precision by velocity band |
| Promo or holiday breaks baseline | Include promo and holiday terms; show an example where the model correctly *does not* flag a promo-end dip |
| Round 1 rubric unknown | Cover all four deliverables from the brief at least in basic form |

## 5. The Round 1 Presentation Video: Best Flow (about 3 minutes)

Keep it short, visual and story-led. Show the product working in the first 60 seconds.

| Time | Scene | What to show and say |
| --- | --- | --- |
| 0:00–0:20 | **Hook** | An empty shelf. Voiceover: "The system says 6 units. The shelf says zero. Nobody notices for weeks." Overlay one hard number: 65% of store inventory records are wrong (DeHoratius & Raman) \[S1\] |
| 0:20–0:45 | **Why it matters** | Out-of-stocks about $1.2T of a $1.7T yearly distortion cost (IHL 2026) \[S8\]; shoppers go to another store 31% of the time \[S10\]. One line on India shrink (Trent 0.22% → 0.41%) \[S17\] |
| 0:45–1:00 | **The manager's real constraint** | "A manager can check 20 shelves a day. Which 20?" Introduce StockGuard name and logo |
| 1:00–1:40 | **Live demo** | Open the dashboard. Pick a store. Show the top-20 list with confidence, ₹ at risk and reason. Click one: sales-vs-stock chart with the shaded zero streak and peer stores still selling. Tap "Shelf empty" |
| 1:40–2:05 | **How it works (one slide)** | Diagram: expected demand → surprise of the zero streak → peer and substitute evidence → value-ranked top 20. Mention e^(−λN) with the fast vs slow item example |
| 2:05–2:30 | **Results** | Precision@20 and recall vs the two baselines (bar chart); time-to-detect improvement; robustness grid in one glance |
| 2:30–2:50 | **Rupees** | Show the arithmetic from section 2.7 and the net figure, with the assumptions listed |
| 2:50–3:00 | **Close** | Limitations in one line (synthetic data, to be validated on real data), what is next (feedback loop, three-way diagnosis), and the tagline |

**Video tips**

- Open with the problem image, not the team intro.
- Keep every on-screen number sourced (small footer citation).
- Use real screen recording of the working tool, not mock screenshots.
- Put the "limitations" line on screen. It builds trust.
- Captions on, clean audio, under 3 minutes unless the rules allow longer.

## 6. Final Checklist Before Submission

- Synthetic data generator, with documented assumptions and a fixed random seed
- StockGuard scoring code and ranked daily output (store, SKU, confidence, reason)
- evaluate.py with precision@20 per store-day, recall, time-to-detect and baselines
- Dashboard with sales-vs-stock chart for any flagged SKU
- Rupee estimate with the arithmetic shown
- Stretch demo (three failure types) if time allows
- Limitations slide
- Video under the time limit, with sources in the footer

## 7. Sources and Citations

*Where a figure comes from a vendor or a very old survey, it is flagged above. Check the original before quoting it as fact on a slide.*

**Academic and research**

- **\[S1\]** DeHoratius, N. & Raman, A. (2008). "Inventory Record Inaccuracy: An Empirical Analysis." *Management Science*, 54(4), 627–641. https://doi.org/10.1287/mnsc.1070.0789
- **\[S2\]** Raman, A., DeHoratius, N. & Ton, Z. (2001). "Execution: The Missing Link in Retail Operations." *California Management Review*. https://cmr.berkeley.edu/2001/05/43-3-execution-the-missing-link-in-retail-operations/
- **\[S3\]** Inventory record inaccuracy, causes and labor effects (research report summarising Oliva et al., 2015 and others). https://ah.lib.nccu.edu.tw/bitstream/140.119/111203/1/103-2410-H-004-117.pdf
- **\[S4\]** Auburn RFID Lab, "Does RFID Improve Inventory Accuracy?" (includes Gruen & Corsten 2007 finding of 55% inaccuracy and the Walmart-commissioned RFID study). https://htmlscript.auburn.edu/under-development/rfid/\_assets/pdf/RFIDImproveIntentoryAccuracy.pdf
- **\[S5\]** "Inventory record inaccuracy in grocery retailing" (arXiv 2506.05357). https://web3.arxiv.org/pdf/2506.05357
- **\[S6\]** Supply Chain Management Review, "Inventory Accuracy: Essential, but Often Overlooked" (freezing, cycle-count limits). https://www.scmr.com/article/inventory\_accuracy\_essential\_but\_often\_overlooked/Inventory%20Management
- **\[S7\]** MIT Libraries, "Detection of phantom inventories at retail stores using a Bayesian approach" (graduate thesis). https://dspace.mit.edu/handle/1721.1/122257
- Also relevant: DeHoratius, Holzapfel, Kuhn et al., "Prioritization Procedures for Improving Inventory Accuracy in Retail Stores" (*MSOM*, 2022), on how to pick which items to count. https://edoc.ku.de/id/eprint/27376/1/msom.2022.1119.pdf

**Industry and market data**

- **\[S8\]** IHL Group, "The 2026 Inventory Distortion Study" ($1.7T, 6.2% of global retail sales). https://www.ihlservices.com/?p=43092
- **\[S9\]** Retail Insight Network on IHL's 2026 split (out-of-stocks about 65.6% of distortion). https://www.retail-insight-network.com/?p=130512
- **\[S10\]** FMI / GMA / CIES study on out-of-stock root causes (8.3% average, 4% sales loss, 31% switch store, 26% switch brand), press release 14 Oct 2002. https://fmi.org/newsroom/news-archive/view/2002/10/14/new-industry-study-explores-root-causes-of-global-out-of-stock-dilemma
- **\[S11\]** Corsten, D. & Gruen, T. (2003). "Desperately seeking shelf availability." *Int. Journal of Retail & Distribution Management*, 31(12), 605–617. https://doi.org/10.1108/09590550310507731
- **\[S12\]** Simbe, "When Stock Lies: How Phantom Inventory Disrupts Retail" (cites ECR: up to 3% sales reduction; vendor content). https://www.simberobotics.com/insights/when-stock-lies-how-phantom-inventory-disrupts-retail-and-how-to-solve-it/blog
- **\[S13\]** RELEX Solutions, "Solving the $400 billion retail inventory accuracy problem with RELEX True Inventory" (vendor content). https://www.relexsolutions.com/resources/solving-retails-400-billion-accuracy-problem/
- **\[S14\]** RELEX press release, 2 Oct 2024, predictive inventory capabilities. https://relexsolutions.com/news/relex-introduces-ai-powered-predictive-inventory-capabilities
- **\[S18\]** Invent.ai, Phantom Inventory (probability of zero sales over consecutive days, daily store alerts; vendor content). https://invent.ai/phantom-inventory

**India**

- **\[S15\]** afaqs!, "India once again tops the Retail Shrinkage Survey" (Global Retail Theft Barometer 2008, India 3.10%). https://www.afaqs.com/company-briefs/38436\_india-once-again-tops-the-retail-shrinkage-survey-
- **\[S16\]** IFSEC Global, "India tops retail shrinkage rate" (2009, 3.2% of sales). https://www.ifsecglobal.com/india-region/india-tops-retail-shrinkage-rate/
- **\[S17\]** "Stop, thief! Retail giants are facing a new 'lifting' problem" (Trent shrink 0.22% → 0.41% in FY24; RAI comments; NRF shrink 1.6%). Republished at https://www.thirdeyesight.in/2024/06/ (check the original publication and Trent's annual report before quoting on a slide)

## 8. Items to Verify Before You Present

- The IHL, RELEX and Simbe figures are headline numbers from the vendors' own pages. Download the free IHL report and cite its page numbers if you want a bulletproof slide.
- The India theft-barometer numbers are from 2008–09. Prefer the Trent and RAI figures for a current view.
- The 8.3% and 31% figures are from a 2002 study. They are the standard reference but are old; say "landmark study".
- Every rupee figure in section 2.7 is an assumption until you replace it with your simulator's output.
- Check the Round 1 submission rules (video length, format, whether a working demo is required) and adjust the timings in section 5 accordingly.
