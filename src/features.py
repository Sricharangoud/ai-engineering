"""Leakage-free feature engineering and preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "discount_pct",
    "qty",
    "order_value_inr",
    "list_price_inr",
    "warranty_months",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "customer_prior_return_rate",
    "is_repeat_customer",
    "is_gift_flag",
    "shield_member_flag",
    "is_default_pincode",
    "has_delivery_note",
    "delivery_note_len",
    "note_whatsapp",
    "note_do_not_call",
    "note_fragile",
    "note_third_floor",
    "note_office",
    "note_landmark",
    "order_hour",
    "order_dayofweek",
    "customer_tenure_days",
    "product_age_days",
]

CATEGORICAL_FEATURES = [
    "sales_channel",
    "payment_mode",
    "family",
    "sku",
    "state",
    "pincode_prefix2",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pre-dispatch features without target or post-dispatch columns."""
    out = pd.DataFrame(index=df.index)

    order_dt = pd.to_datetime(df["order_placed_at"], errors="coerce")
    signup_dt = pd.to_datetime(df["signup_date"], errors="coerce")
    launch_dt = pd.to_datetime(df["launch_date"], errors="coerce")

    out["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    out["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    out["order_value_inr"] = pd.to_numeric(df["order_value_inr"], errors="coerce")
    out["list_price_inr"] = pd.to_numeric(df["list_price_inr"], errors="coerce")
    out["warranty_months"] = pd.to_numeric(df["warranty_months"], errors="coerce")
    out["promised_delivery_days"] = pd.to_numeric(df["promised_delivery_days"], errors="coerce")

    prior_orders = pd.to_numeric(df["customer_prior_orders"], errors="coerce").fillna(0.0)
    prior_returns = pd.to_numeric(df["customer_prior_returns"], errors="coerce").fillna(0.0)
    out["customer_prior_orders"] = prior_orders
    out["customer_prior_returns"] = prior_returns
    out["customer_prior_return_rate"] = np.where(prior_orders > 0, prior_returns / prior_orders, 0.0)
    out["is_repeat_customer"] = (prior_orders > 0).astype(float)

    out["is_gift_flag"] = (df["is_gift"].astype(str).str.upper() == "Y").astype(float)
    out["shield_member_flag"] = (df["shield_member"].astype(str).str.upper() == "Y").astype(float)

    pincode = df["delivery_pincode"].astype(str).str.zfill(6)
    out["is_default_pincode"] = (pincode == "000000").astype(float)
    out["pincode_prefix2"] = np.where(pincode == "000000", "00", pincode.str[:2])

    note = df["delivery_note"].fillna("").astype(str).str.strip()
    note_lower = note.str.lower()
    out["has_delivery_note"] = (note.str.len() > 0).astype(float)
    out["delivery_note_len"] = note.str.len().astype(float)
    out["note_whatsapp"] = note_lower.str.contains("whatsapp", regex=False).astype(float)
    out["note_do_not_call"] = note_lower.str.contains("do not call", regex=False).astype(float)
    out["note_fragile"] = note_lower.str.contains("fragile", regex=False).astype(float)
    out["note_third_floor"] = note_lower.str.contains("third floor", regex=False).astype(float)
    out["note_office"] = note_lower.str.contains("office", regex=False).astype(float)
    out["note_landmark"] = note_lower.str.contains("landmark:", regex=False).astype(float)

    out["order_hour"] = order_dt.dt.hour.astype(float)
    out["order_dayofweek"] = order_dt.dt.dayofweek.astype(float)
    out["customer_tenure_days"] = (order_dt - signup_dt).dt.total_seconds().div(86400.0).clip(lower=0.0)
    out["product_age_days"] = (order_dt - launch_dt).dt.total_seconds().div(86400.0).clip(lower=0.0)

    out["sales_channel"] = df["sales_channel"].astype(str)
    out["payment_mode"] = df["payment_mode"].astype(str)
    out["family"] = df["family"].astype(str)
    out["sku"] = df["sku"].astype(str)
    out["state"] = df["state"].astype(str)

    return out[MODEL_FEATURES]


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create a scikit-learn preprocessor to be fit exclusively on training folds."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
