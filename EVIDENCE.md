# Model Evidence & Evaluation — Kestrel Home Returns Risk (Variant A)

## 1. Data Sizes, Deduplication & Label Prevalence

| Dataset | Raw Rows | Unique `order_id`s | Date Range (IST) | Return Rate | Notes |
| --- | --- | --- | --- | --- | --- |
| `train.csv` (raw) | 11,155 | 10,504 | 2025-04-01 to 2026-06-30 | 11.36% (1,267/11,155) | 651 partner-feed re-imports (`source=partner_feed`) |
| `train.csv` (dedup) | 10,504 | 10,504 | 2025-04-01 to 2026-06-30 | 11.42% (1,200/10,504) | Deduplicated preferring `source=crm`; 0 label conflicts |
| `test_unlabelled.csv` | 2,096 | 2,096 | 2026-07-01 to 2026-09-30 | Unlabelled | All `source=crm`, exact `order_id` match with `sample_submission.csv` |
| `customers.csv` | 9,000 | 9,000 | — | — | 100% join coverage on `customer_id` |
| `products.csv` | 21 | 21 | — | — | 100% join coverage on `sku` |

## 2. Time-Based Validation Split

To mirror the 3-month future dispatch horizon (`2026-07-01` to `2026-09-30`, 2,096 orders), we split deduplicated `train.csv` at **`2026-04-01 00:00` IST**:
- **Train fold:** `2025-04-01` to `2026-03-31` — **8,378 orders** (11.40% return rate).
- **Validation fold:** `2026-04-01` to `2026-06-30` — **2,126 orders** (11.52% return rate; mean order value ₹8,139.00, median ₹4,439.62).

All preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) is fit **only** on the training fold during validation, and then the selected pipeline is refit on all 10,504 deduplicated training orders for batch scoring and serving.

## 3. Data Quality & Leakage Checks

### Data Quality Fixes Applied
1. **Duplicate partner-feed re-imports:** 651 orders appeared twice (`crm` + `partner_feed`) with identical labels and features except `source`. Deduplicated by `order_id` preferring `crm`.
2. **October 2025 payment-gateway paise bug:** All 700 orders in `2025-10` had `order_value_inr` stored as `100 ×` expected INR (`list_price_inr * qty * (1 - discount_pct / 100)`). Any row with `order_value_inr / expected > 10` is divided by `100.0`.
3. **Default walk-in pincode (`000000`):** 848 deduplicated train orders and 176 test orders have `delivery_pincode = 000000`. Captured via an explicit `is_default_pincode` flag and `00` pincode prefix.

### Excluded Post-Dispatch / Leaky Fields
The following columns are **strictly excluded** from features:
- `last_service_event_type`: In `train.csv` (exported later), `REVERSE_PICKUP` has **750/750 (100.0%)** return rate, `TECH_VISIT` has **172/447 (38.5%)**, and `DEMO_DONE` / `INSTALL_DONE` have **0/2,771 (0.0%)**. In `test_unlabelled.csv` (dispatch snapshot), only `NONE` (1,553) and `INSTALL_BOOKED` (543) appear—and `INSTALL_BOOKED` appears 0 times in `train.csv`.
- `pickup_scheduled_at`: In `train.csv`, non-empty `pickup_scheduled_at` has **1,130/1,231 (91.8%)** return rate vs **70/9,273 (0.8%)** when empty. In `test_unlabelled.csv`, it is **100% empty (2,096/2,096)**.
- `source`: Constant `crm` after deduplication and in `test_unlabelled.csv`.
- `returned`: Target label.

## 4. Model Comparison (Validation Fold, n = 2,126)

| Model | ROC-AUC | PR-AUC | Brier Score | Log-Loss | Accuracy @ 0.50 |
| --- | --- | --- | --- | --- | --- |
| `majority_baseline` (constant) | 0.5000 | 0.1152 | 0.1020 | 0.3573 | 0.8848 |
| **`logistic_regression`** (selected) | **0.7816** | **0.4056** | **0.0855** | **0.2957** | 0.8937 |
| `xgboost` | 0.7682 | 0.3941 | 0.0865 | 0.3004 | 0.8951 |
| `hist_gradient_boosting` | 0.7580 | 0.3832 | 0.0873 | 0.3051 | 0.8937 |

**Metric rationale:** With an 11.52% return rate, accuracy is misleading—predicting "never return" already achieves **88.48% accuracy** with 0% recall. **ROC-AUC** (`0.7816`) measures ranking quality across all thresholds for prioritizing a pre-dispatch call queue, while **PR-AUC** (`0.4056` vs `0.1152` baseline) and **Brier score** (`0.0855`) verify positive-class precision and probability calibration.

## 5. Threshold & Cost Analysis (Validation Fold, n = 2,126)

Policy assumptions (§4, §7): Return cost = ₹1,150; Confirmation call = ₹45; Pilot call prevention = 35% of otherwise-expected returns. Break-even probability = `45 / (0.35 × 1150) = 0.1118`.

| Threshold | Flagged (%) | TP | FP | TN | FN | Precision | Recall | Accuracy | Est. Calls / 700 Orders | Est. Net Savings / 700 Orders |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0.1118** (break-even) | 623 (29.3%) | 159 | 464 | 1,417 | 86 | 25.5% | 64.9% | 74.1% | 205.1 | **₹11,840.90** |
| **0.1500** (pilot alt.) | 445 (20.9%) | 134 | 311 | 1,570 | 111 | 30.1% | 54.7% | 80.2% | 146.5 | **₹11,165.10** |
| 0.2000 | 300 (14.1%) | 106 | 194 | 1,687 | 139 | 35.3% | 43.3% | 84.3% | 98.8 | ₹9,602.78 |
| 0.2500 | 212 (10.0%) | 87 | 125 | 1,756 | 158 | 41.0% | 35.5% | 86.7% | 69.8 | ₹8,388.64 |
| 0.3000 | 154 (7.2%) | 73 | 81 | 1,800 | 172 | 47.4% | 29.8% | 88.1% | 50.7 | ₹7,392.64 |
| 0.3500 | 114 (5.4%) | 66 | 48 | 1,833 | 179 | 57.9% | 26.9% | 89.3% | 37.5 | ₹7,057.62 |

**Shield Segment Note (Validation):** Shield members account for **478 / 2,126 (22.5%)** of validation orders and have a **20.50%** return rate vs **8.92%** for non-Shield orders.

## 6. What the Model Gets Wrong (Synthetic / Anonymized Failure Modes)

- **False Positives (FP = 464 at 0.1118):** Orders with risk-correlated traits (e.g., COD payment, Robot Vacuum or Water Purifier SKU, prior customer returns, or gift flag) where the customer ultimately keeps the product. Calling these orders costs ₹45/call (acceptable if not held), whereas auto-holding them risks a 12% cancellation rate.
- **False Negatives (FN = 86 at 0.1118):** Prepaid UPI/card orders from first-time customers for lower-return categories (e.g., Mixer Grinder or Ceiling Fan) that are returned due to post-dispatch events unobservable at checkout (transit damage, installation issue, or sudden buyer's remorse).

## 7. Software Verification (`pytest`)

Run `python -m pytest -v` (last run result: **9 passed in 8.25s**):
- `tests/test_pipeline.py`: deduplication (`crm` preference), October paise normalization, leaky-column exclusion, `predictions.csv` exact match with `sample_submission.csv`.
- `tests/test_service.py`: `/health`, `/` UI HTML, `/predict` valid synthetic payload (score + faithful log-odds reasons), `/predict` 422 validation errors for unknown SKU and `customer_prior_returns > customer_prior_orders`.

## 8. Limitations & Uncertainty

- **Unverified causal effect:** The 35% return reduction from confirmation calls comes from a spring pilot average (§7), not a randomized trial on model-flagged orders. Actual savings could be lower if high-risk orders (e.g., Shield free returns or defective units) are less responsive to calls.
- **Seasonality / shift:** Validation covers Apr–Jun 2026; test covers Jul–Sep 2026 (monsoon/festive ramp), where return rates or product mix may shift.
- **No post-dispatch visibility:** Transit delays or installation failures happen after dispatch and cannot be seen at prediction time.
