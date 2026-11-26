# Memo: Pre-Dispatch Return Risk Scoring & Pilot Plan

**To:** Ritu Deshpande, Head of D2C Operations  
**Cc:** Farhan Sheikh (Finance), Meenal Joshi (Service Desk), Kabir Nanda  
**Subject:** Recommendation on Return Risk Model, 95% Accuracy Ask, and Next-Week Pilot

---

## 1. Decision Summary

Do **not** automatically hold flagged orders, and do **not** target 95% classification accuracy. Instead, use the new pre-dispatch risk score (`0%–100%` probability of return) to prioritize **₹45 pre-dispatch confirmation calls** on the highest-risk non-Shield orders, and run a **2-week controlled pilot** starting next week.

## 2. Why 95% Accuracy and Auto-Holds Would Hurt Kestrel

- **Why 95% accuracy is misleading:** Only **11.5%** of recent orders are returned. A useless rule that predicts *"no order will ever be returned"* already gets **88.5% accuracy** while catching **zero returns**. Optimizing for accuracy treats a ₹1,150 return the same as a ₹45 phone call.
- **Why auto-holding is costly:** Under Policy §7, holding an order over 24 hours causes customers to cancel **~12%** of the time (on orders averaging **₹8,139**; median **₹4,440**). Worse, **Kestrel Shield members** make up **22.5%** of orders and account for a **20.5%** return rate (vs **8.9%** for non-Shield) because returns are free for them. Auto-holding Shield orders would frustrate Kestrel's highest-lifetime-value repeat buyers.

## 3. Model Performance (Time-Based Validation)

We trained a leakage-free model on 8,378 historical orders (`2025-04-01` to `2026-03-31`) and tested it on the next 3 months (`2026-04-01` to `2026-06-30`, **2,126 orders**, **11.52%** return rate):
- **Ranking score (ROC-AUC):** **0.7816** (vs `0.5000` random baseline; PR-AUC `0.4056` vs `0.1152` baseline).
- Each score comes with plain-English risk drivers (e.g., COD payment, prior returns, Robot Vacuum category, long promised delivery) so desk agents know *why* an order is flagged.

## 4. Cost Arithmetic (~700 Orders / Month)

Using Policy §4 and §7 figures (**₹1,150** per return, **₹45** per completed confirmation call, **~35%** return prevention on called orders from the spring pilot):
- **Break-even risk:** A ₹45 call pays for itself whenever expected return risk exceeds `45 ÷ (0.35 × 1,150) = 11.2%`.
- **Break-even threshold (`11.2%`):** Flags **29.3%** of orders (~**205 calls/month**), catching **64.9%** of returns.
  - Monthly call cost: `205.1 × ₹45 = ₹9,231`
  - Estimated gross savings (if 35% prevention holds): `₹21,072`
  - **Estimated net savings:** **~₹11,841 / month**
- **Focused pilot threshold (`15.0%`):** Flags **20.9%** of orders (~**147 calls/month**), catching **54.7%** of returns with **29% fewer calls**:
  - Monthly call cost: `146.5 × ₹45 = ₹6,593`
  - Estimated gross savings: `₹17,758`
  - **Estimated net savings:** **~₹11,165 / month**
- **AI / Model API Cost:** **₹0 per prediction** and **₹0 / month for 700 orders** (runs locally with no paid external API; standard app hosting/engineering time is separate).

## 5. Next-Week Pilot & What to Monitor

1. **Scope (2 weeks):** Score all incoming orders at dispatch. Route **non-Shield** orders scoring **≥ 15%** (~35 calls/week) to the confirmation call queue; treat **Shield** orders as courtesy check-ins only (never hold).
2. **Holdout check:** Randomly skip calling 20% of flagged orders during the pilot so Finance and Ops can measure the **true** causal return reduction rather than assuming the 35% spring pilot figure holds everywhere.
3. **Monitor weekly:** (a) Call completion rate, (b) Return rate in called vs. uncalled flagged orders, (c) Any customer cancellation spike, and (d) Shield customer feedback.
