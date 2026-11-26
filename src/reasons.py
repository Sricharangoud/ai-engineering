"""Faithful, employee-readable reason explanations from the fitted model."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


def _humanize_feature(transformed_name: str, raw_row: pd.Series) -> tuple[str, str]:
    """Map a transformed feature name to a (group_key, readable_label) pair."""
    name = transformed_name.split("__", 1)[-1]

    if name == "payment_mode_cod":
        return "payment_mode", "Cash-on-delivery (COD) payment"
    if name.startswith("payment_mode_"):
        return "payment_mode", f"Payment mode: {raw_row['payment_mode']}"
    if name.startswith("sales_channel_"):
        return "sales_channel", f"Sales channel: {raw_row['sales_channel']}"
    if name.startswith("family_"):
        return "family", f"Product family: {raw_row['family']}"
    if name.startswith("sku_"):
        return "sku", f"SKU: {raw_row['sku']}"
    if name.startswith("state_"):
        return "state", f"Customer state: {raw_row['state']}"
    if name.startswith("pincode_prefix2_"):
        return "pincode_prefix2", "Delivery pincode region"

    mapping = {
        "customer_prior_returns": ("prior_returns", f"Prior returns count ({int(raw_row['customer_prior_returns'])})"),
        "customer_prior_return_rate": ("prior_returns", "Customer historical return rate"),
        "customer_prior_orders": ("prior_orders", f"Prior orders count ({int(raw_row['customer_prior_orders'])})"),
        "is_repeat_customer": ("prior_orders", "Repeat customer history"),
        "shield_member_flag": ("shield_member", f"Kestrel Shield member ({raw_row['shield_member']})"),
        "promised_delivery_days": ("promised_delivery_days", f"Promised delivery window ({int(raw_row['promised_delivery_days'])} days)"),
        "is_gift_flag": ("is_gift", f"Gift order ({raw_row['is_gift']})"),
        "discount_pct": ("discount_pct", f"Checkout discount ({float(raw_row['discount_pct']):.0f}%)"),
        "order_value_inr": ("order_value_inr", f"Order value (Rs {float(raw_row['order_value_inr']):,.0f})"),
        "list_price_inr": ("list_price_inr", f"Product list price (Rs {float(raw_row['list_price_inr']):,.0f})"),
        "warranty_months": ("warranty_months", f"Warranty length ({int(raw_row['warranty_months'])} months)"),
        "qty": ("qty", f"Quantity ({int(raw_row['qty'])})"),
        "is_default_pincode": ("is_default_pincode", "Default 000000 pincode (no address captured)"),
        "has_delivery_note": ("delivery_note", "Presence of delivery note"),
        "delivery_note_len": ("delivery_note", "Delivery note length"),
        "note_whatsapp": ("delivery_note", "Delivery note mentions WhatsApp"),
        "note_do_not_call": ("delivery_note", "Delivery note requests no phone calls"),
        "note_fragile": ("delivery_note", "Delivery note mentions fragile"),
        "note_third_floor": ("delivery_note", "Delivery note mentions third floor / no lift"),
        "note_office": ("delivery_note", "Delivery note mentions office address"),
        "note_landmark": ("delivery_note", "Delivery note includes landmark"),
        "order_hour": ("order_timing", "Order placement hour (IST)"),
        "order_dayofweek": ("order_timing", "Order day of week"),
        "customer_tenure_days": ("customer_tenure", "Customer account tenure"),
        "product_age_days": ("product_age", "Product age since launch"),
    }
    return mapping.get(name, (name, name))


def compute_model_contributions(
    model: Any,
    model_name: str,
    X_trans: np.ndarray,
) -> np.ndarray:
    """Return per-feature log-odds contributions matching the fitted model."""
    if model_name == "xgboost":
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_trans)
        contribs = booster.predict(dmat, pred_contribs=True)
        return contribs[:, :-1]  # drop bias term
    if model_name == "logistic_regression":
        coef = model.coef_[0]
        return X_trans * coef
    return np.zeros_like(X_trans)


def explain_single_prediction(
    model: Any,
    model_name: str,
    preprocessor: Any,
    feature_row_df: pd.DataFrame,
    raw_row: pd.Series,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Produce top employee-readable risk factors faithful to the fitted model."""
    X_trans = preprocessor.transform(feature_row_df)
    contribs = compute_model_contributions(model, model_name, X_trans)[0]
    feat_names = preprocessor.get_feature_names_out()

    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"impact": 0.0, "label": ""})
    for fname, val in zip(feat_names, contribs):
        group_key, label = _humanize_feature(fname, raw_row)
        grouped[group_key]["impact"] += float(val)
        if not grouped[group_key]["label"] or abs(val) > 0:
            grouped[group_key]["label"] = label

    items = [
        {
            "factor": v["label"],
            "direction": "increases_risk" if v["impact"] > 0 else "decreases_risk",
            "log_odds_impact": round(v["impact"], 4),
        }
        for v in grouped.values()
        if abs(v["impact"]) >= 0.02
    ]

    positives = sorted([x for x in items if x["direction"] == "increases_risk"], key=lambda x: -x["log_odds_impact"])
    negatives = sorted([x for x in items if x["direction"] == "decreases_risk"], key=lambda x: x["log_odds_impact"])

    selected = positives[:top_k]
    if len(selected) < top_k and negatives:
        selected.extend(negatives[: top_k - len(selected)])
    elif negatives:
        selected.append(negatives[0])

    return selected
