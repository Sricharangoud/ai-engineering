# Submission Artifacts: Recording Script & Form Answers

## Part A: 3-Minute Screen Recording Script (No Slides, Synthetic Data Only)

> **Recording link:** `[PLACEHOLDER: INSERT PUBLIC RECORDING URL]`  
> **Rules:** Keep under 3:00. Use only synthetic/redacted records in the UI and terminal. Do not show raw `train.csv`, `customers.csv`, or real delivery notes.

1. **0:00–0:35 — Business framing & what was changed:** Open `MEMO_TO_RITU.md`. Explain why we pushed back on the 95% accuracy / auto-hold ask (88.5% majority baseline, 12% hold-cancellation risk, Shield LTV) and pivoted to a ₹45 confirmation-call triage score.
2. **0:35–1:15 — Data bugs & leakage caught:** Show `src/data.py` and `EVIDENCE.md` (no raw CSVs). Point out the 651 `partner_feed` duplicates, the October 2025 100x paise bug, and why `last_service_event_type` (`REVERSE_PICKUP` = 100% return in train, absent in test) and `pickup_scheduled_at` were excluded.
3. **1:15–1:55 — Validation & discarded models:** Show `reports/validation_metrics.json`. Explain the `2026-04-01` time split (`n=2,126` val) and why we discarded XGBoost (`0.7682` ROC-AUC) and HistGradientBoosting (`0.7580`) in favor of Logistic Regression (`0.7816` ROC-AUC, `0.4056` PR-AUC, better calibration, exact reason weights).
4. **1:55–2:40 — Live UI & API demo:** Run `uvicorn src.service:app --port 8000` in terminal and open `http://127.0.0.1:8000/`. Click **Load High-Risk Sample** -> **Score Return Risk** (show score, action, and readable reasons), then **Load Shield Sample** (show courtesy-check-only guidance), then trigger a validation error (e.g. `prior_returns > prior_orders`) to show polite error handling.
5. **2:40–3:00 — Tests & closing:** Run `python -m pytest -v` (9/9 passed) and mention the next-week 2-week pilot with a 20% uncalled holdout.

---

## Part B: Draft Submission-Form Answers

1. **What did you build, and what business outcome does it move? State the actual number and money.**  
   Built a leakage-free pre-dispatch return risk pipeline, batch scorer (`predictions.csv`), and local FastAPI + HTML operations UI (`src/service.py`, `src/static/index.html`) that scores orders and surfaces employee-readable drivers. Instead of auto-holding orders, it triages ₹45 pre-dispatch confirmation calls. On the 2,126-order time-based validation fold (`0.7816` ROC-AUC), flagging orders at the `11.18%` break-even threshold (29.3% of orders) projects **₹11,841/month** net savings at 700 orders/month (₹21,072 gross saved minus ₹9,231 call cost), or **₹11,165/month** at a `15.0%` pilot threshold with 29% fewer calls (assuming the spring pilot's 35% prevention rate holds).

2. **Expected hidden-outcome score, metric, rationale, and how it was estimated.**  
   Expected hidden-test (`test_unlabelled.csv`, Jul–Sep 2026, `n=2,096`) **ROC-AUC of ~0.75–0.79** (point estimate **0.7816**, **PR-AUC ~0.4056**). Estimated from a strict 3-month out-of-time validation split (`2026-04-01` to `2026-06-30`, `n=2,126`, `11.52%` return rate) after removing all post-dispatch leakage. Uncertainty comes from seasonal/monsoon-to-festive shifts between Q2 and Q3 2026.

3. **How do you know it works? Sample size, checks, error rate, and typical failure case.**  
   Validated on `n=2,126` future orders (`0.7816` ROC-AUC, `0.0855` Brier score; at `0.1118` threshold: `64.9%` recall, `25.5%` precision, `74.1%` accuracy). Verified `predictions.csv` (2,096 rows) against `sample_submission.csv` and ran `pytest -v` (**9/9 tests passed** for dedup, paise fix, leakage exclusion, API validation, and UI). Typical failures: false positives on COD/Robot Vacuum/Shield orders that customers keep, and false negatives on prepaid low-risk-category orders returned due to post-dispatch transit damage or installation issues.

4. **Did you change, narrow, or push back on the client’s ask? What, when, and why?**  
   Yes—pushed back upfront on both **95% accuracy** and **automatic holds**: (a) with an 11.4% return rate, predicting "never return" already hits 88.5% accuracy with zero business value; (b) Policy §7 shows >24h holds cause ~12% cancellations on ~₹8,139 average orders and risk churning high-LTV Shield members (20.5% return rate). Narrowed the action to ₹45 pre-dispatch confirmation calls on non-Shield high-risk orders, with a 2-week pilot + 20% holdout.

5. **What is wrong with the handoff? Be specific about known bugs, shortcuts, and gaps.**  
   (a) The 35% call-prevention rate is an unverified observational pilot average, not proven causal across high-risk segments. (b) `delivery_note` uses fixed keyword indicators rather than full NLP/embeddings. (c) `predictions.csv` is excluded from Git via `.gitignore` per Policy §10 and must be shared via private Drive, not public GitHub. (d) Clean-machine `pip install` was not separately tested in a fresh container.

6. **What did you deliberately leave out, and why?**  
   Left out `last_service_event_type` and `pickup_scheduled_at` (post-dispatch leakage), automated hold triggers (12% cancellation risk), external paid LLM APIs (unnecessary cost and latency; Finance asked for no surprise per-order bill), and complex ensembles/neural nets (XGBoost and HistGradientBoosting underperformed Logistic Regression on validation).

7. **Anything built or found that nobody asked for?**  
   Found and fixed an undocumented **100x paise bug** in `order_value_inr` across all 700 October 2025 orders (`2025-10`) from the new payment gateway, and documented the Policy §10 restriction blocking `predictions.csv` from public Git tracking.

8. **What AI tools/models were used, where they helped or wasted time, what was discarded, their cost, and the recording link.**  
   Used **Gemini 4 Hydrogen (Medium)** via Antigravity for EDA, pipeline scaffolding, FastAPI/UI implementation, and documentation. Helped rapidly diagnose the October paise bug and leakage columns. Discarded XGBoost (`0.7682` ROC-AUC) and HistGradientBoosting (`0.7580` ROC-AUC) after validation showed Logistic Regression (`0.7816`) performed better and was simpler. Recording link: `[PLACEHOLDER: INSERT PUBLIC RECORDING URL]`.

9. **Public Google Drive link.**  
   `[PLACEHOLDER: INSERT GOOGLE DRIVE LINK]` *(Note: Per Policy §10, restrict raw data/predictions sharing to the engagement team only.)*

10. **Three things a Monday handoff owner needs to know.**  
    1. Never re-enable `last_service_event_type` or `pickup_scheduled_at`—they are post-dispatch leakage.  
    2. Do not auto-hold Shield members; route non-Shield `>= 0.15` orders to ₹45 confirmation calls with a 20% uncalled control group for 2 weeks.  
    3. Keep `train.csv`, `customers.csv`, and `predictions.csv` out of public GitHub per Policy §10.

11. **Honest hours spent, as one number.**  
    `[PLACEHOLDER: INSERT HONEST HOURS SPENT]`

12. **Public GitHub repository URL.**  
    `[PLACEHOLDER: INSERT PUBLIC GITHUB REPOSITORY URL]`

13. **Cost per prediction and monthly cost at about 700 orders, with arithmetic.**  
    - **Model API cost:** **₹0 per prediction**, **₹0/month** at 700 orders (`700 × ₹0 = ₹0`), because inference runs locally in Python/FastAPI with zero external paid API calls (excluding standard server hosting/engineering).  
    - **Operational call cost (if acting on predictions):** At `0.15` pilot threshold (20.93% flagged), `700 × 0.2093 = 146.5` calls/month × ₹45/call = **₹6,593/month** (or at `0.1118` break-even, `700 × 0.2930 = 205.1` calls × ₹45 = **₹9,231/month**).
