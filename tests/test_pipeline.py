"""Unit and integration tests for the data, feature, and prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import LEAKY_OR_EXCLUDED_COLUMNS, deduplicate_orders, fix_paise_order_values, resolve_pack_file
from src.features import MODEL_FEATURES, build_feature_frame

ROOT = Path(__file__).resolve().parent.parent


def test_deduplicate_orders_prefers_crm() -> None:
    df = pd.DataFrame(
        [
            {"order_id": "O1", "source": "partner_feed", "val": 1},
            {"order_id": "O1", "source": "crm", "val": 1},
            {"order_id": "O2", "source": "crm", "val": 2},
        ]
    )
    out = deduplicate_orders(df)
    assert len(out) == 2
    assert out["order_id"].is_unique
    assert out.loc[out["order_id"] == "O1", "source"].iloc[0] == "crm"


def test_fix_paise_order_values_normalizes_100x_outliers() -> None:
    df = pd.DataFrame(
        [
            {"list_price_inr": 5000.0, "qty": 1, "discount_pct": 10.0, "order_value_inr": 450000.0},
            {"list_price_inr": 5000.0, "qty": 1, "discount_pct": 10.0, "order_value_inr": 4500.0},
        ]
    )
    out = fix_paise_order_values(df)
    assert out["order_value_inr"].tolist() == [4500.0, 4500.0]


def test_build_feature_frame_excludes_leaky_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "order_placed_at": "2026-08-01 10:00",
                "signup_date": "2025-01-01",
                "launch_date": "2024-01-01",
                "discount_pct": 10.0,
                "qty": 1,
                "order_value_inr": 4500.0,
                "list_price_inr": 5000.0,
                "warranty_months": 12,
                "promised_delivery_days": 5,
                "customer_prior_orders": 2,
                "customer_prior_returns": 1,
                "is_gift": "N",
                "shield_member": "Y",
                "delivery_pincode": "560001",
                "delivery_note": "Call before delivery",
                "sales_channel": "app",
                "payment_mode": "cod",
                "family": "Air Fryer",
                "sku": "KH-AF-01",
                "state": "KA",
                "last_service_event_type": "REVERSE_PICKUP",
                "pickup_scheduled_at": "2026-08-05 12:00",
                "source": "crm",
                "returned": 1,
            }
        ]
    )
    feats = build_feature_frame(df)
    assert list(feats.columns) == MODEL_FEATURES
    for col in LEAKY_OR_EXCLUDED_COLUMNS:
        assert col not in feats.columns
    assert feats["customer_prior_return_rate"].iloc[0] == 0.5


def test_predictions_csv_matches_sample_submission() -> None:
    pred_path = ROOT / "predictions.csv"
    sample_path = resolve_pack_file(ROOT, "sample_submission.csv")
    pred_df = pd.read_csv(pred_path, dtype={"order_id": str})
    sample_df = pd.read_csv(sample_path, dtype={"order_id": str})

    assert list(pred_df.columns) == list(sample_df.columns)
    assert len(pred_df) == len(sample_df)
    assert pred_df["order_id"].equals(sample_df["order_id"])
    assert not pred_df["score"].isna().any()
    assert ((pred_df["score"] >= 0.0) & (pred_df["score"] <= 1.0)).all()
