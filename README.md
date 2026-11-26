# Kestrel Home — Pre-Dispatch Returns Risk (Variant A)

Reproducible, leakage-aware ML pipeline, batch scorer, and local FastAPI + HTML operations UI for predicting pre-dispatch return risk at Kestrel Home.

## Business Decision

- **Do not auto-hold flagged orders** (>24h holds cause ~12% customer cancellations per Policy §7) and **do not optimize for 95% accuracy** (with an 11.4% return rate, a constant "no-return" baseline already gets 88.5% accuracy while catching 0 returns).
- **Use the risk score to triage ₹45 pre-dispatch confirmation calls** (break-even at `11.2%` risk assuming 35% prevention against a ₹1,150 return cost; ₹0 external API cost per prediction).

## Data Placement & Privacy

Per Kestrel Operations Policy §10, customer and operational data **must not be uploaded to public repositories**.
- `.gitignore` excludes all raw task-pack files (`*train.csv`, `*test_unlabelled.csv`, `*customers.csv`, `*products.csv`, `*sample_submission.csv`, `*ops-policy.pdf`, `*email-thread.txt`, `*README.txt`) **and** `predictions.csv`.
- To run locally, place the task-pack files in the repository root (either plain names like `train.csv` or UUID-prefixed names like `*-train.csv`).
- **Submission note:** `predictions.csv` is generated locally in the repository root for grading/submission, but is excluded from Git tracking to avoid publishing `order_id`s and risk scores to a public repository.

## Setup (Clean Machine)

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

*(Clean-machine installation from scratch has not been separately tested in this environment; dependencies are constrained in `requirements.txt` to the verified local versions.)*

## Commands

```bash
# 1. Train, run time-based validation (cutoff 2026-04-01), and save models/model_bundle.joblib
python -m src.train

# 2. Generate and verify predictions.csv against sample_submission.csv
python -m src.predict

# 3. Run unit and API/UI integration tests
python -m pytest -v

# 4. Start the local prediction service & operations UI (no paid API key required)
uvicorn src.service:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser for the single-order scoring screen.

## Endpoint Example (Synthetic Data)

`POST /predict`

```json
{
  "order_placed_at": "2026-08-15 14:30",
  "sku": "KH-RV-03",
  "sales_channel": "marketplace",
  "payment_mode": "cod",
  "discount_pct": 5.0,
  "qty": 1,
  "order_value_inr": 28213.10,
  "promised_delivery_days": 9,
  "delivery_pincode": "560001",
  "is_gift": "Y",
  "customer_prior_orders": 2,
  "customer_prior_returns": 2,
  "delivery_note": "Do not call, WhatsApp only",
  "shield_member": "N",
  "state": "KA",
  "signup_date": "2025-11-10"
}
```

Returns `score`, `risk_pct`, `risk_band`, `threshold_used`, `recommended_action`, `reasons` (faithful model log-odds drivers), and a correlation-vs-causation `caveat`.

## Model, Validation & Limitations Summary

- **Deduplication & Cleaning:** Deduplicates 651 `partner_feed` re-imports in `train.csv` (preferring `crm`), normalizes October 2025 `order_value_inr` paise values (`100x` INR), and flags `000000` walk-in pincodes.
- **Leakage Exclusion:** Excludes `last_service_event_type`, `pickup_scheduled_at`, `source`, and `returned`.
- **Validation:** Time-based split at `2026-04-01` (`n=8,378` train, `n=2,126` val). Selected **Logistic Regression** (`ROC-AUC 0.7816`, `PR-AUC 0.4056`, `Brier 0.0855`) over XGBoost (`0.7682`) and HistGradientBoosting (`0.7580`).
- **Details:** See [EVIDENCE.md](EVIDENCE.md) and [MEMO_TO_RITU.md](MEMO_TO_RITU.md).
