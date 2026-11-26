"""FastAPI prediction service for single-order pre-dispatch return risk scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.data import fix_paise_order_values
from src.features import build_feature_frame
from src.reasons import explain_single_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kestrel_service")

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = BASE_DIR / "models" / "model_bundle.joblib"
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class OrderPredictionRequest(BaseModel):
    """Validated single-order payload for pre-dispatch return risk scoring."""

    order_placed_at: str = Field(..., description="Order timestamp in IST, e.g. '2026-08-15 14:30'")
    sku: str = Field(..., description="Product SKU from products.csv, e.g. 'KH-RV-03'")
    sales_channel: Literal["app", "web", "marketplace", "partner_outlet"]
    payment_mode: Literal["prepaid_upi", "prepaid_card", "cod", "emi"]
    discount_pct: float = Field(..., ge=0.0, le=100.0)
    qty: int = Field(..., ge=1, le=50)
    order_value_inr: float = Field(..., gt=0.0)
    promised_delivery_days: int = Field(..., ge=1, le=60)
    delivery_pincode: str = Field(..., min_length=1, max_length=10)
    is_gift: Literal["Y", "N"]
    customer_prior_orders: int = Field(..., ge=0, le=1000)
    customer_prior_returns: int = Field(..., ge=0, le=1000)
    delivery_note: Optional[str] = Field(default="", max_length=500)
    shield_member: Literal["Y", "N"]
    state: str = Field(..., min_length=2, max_length=10)
    signup_date: str = Field(..., description="Customer signup date YYYY-MM-DD")

    @field_validator("order_placed_at", "signup_date")
    @classmethod
    def validate_datetime_str(cls, v: str) -> str:
        parsed = pd.to_datetime(v, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid date/timestamp format: '{v}'")
        return v

    @field_validator("customer_prior_returns")
    @classmethod
    def validate_prior_returns(cls, v: int, info: Any) -> int:
        prior_orders = info.data.get("customer_prior_orders")
        if prior_orders is not None and v > prior_orders:
            raise ValueError("customer_prior_returns cannot exceed customer_prior_orders")
        return v


def _load_bundle() -> dict[str, Any]:
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Model bundle missing at {BUNDLE_PATH}. Run 'python -m src.train' first.")
    return joblib.load(BUNDLE_PATH)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kestrel Home Pre-Dispatch Returns Risk API",
        version="1.0.0",
        description="Local, zero-API-cost return risk scoring and operations guidance.",
    )
    bundle = _load_bundle()
    products_by_sku = {p["sku"]: p for p in bundle["products"]}

    @app.get("/")
    def serve_ui() -> FileResponse:
        if not STATIC_INDEX.exists():
            raise HTTPException(status_code=404, detail="UI file not found")
        return FileResponse(STATIC_INDEX)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_name": bundle["model_name"],
            "recommended_threshold": bundle["recommended_threshold"],
            "api_cost_per_prediction_inr": 0.0,
        }

    @app.post("/predict")
    def predict(payload: OrderPredictionRequest) -> dict[str, Any]:
        product = products_by_sku.get(payload.sku)
        if product is None:
            valid_skus = ", ".join(sorted(products_by_sku.keys()))
            raise HTTPException(
                status_code=422,
                detail=f"Unknown sku '{payload.sku}'. Valid SKUs: {valid_skus}",
            )

        row_dict = {
            "order_placed_at": payload.order_placed_at,
            "sku": payload.sku,
            "sales_channel": payload.sales_channel,
            "payment_mode": payload.payment_mode,
            "discount_pct": payload.discount_pct,
            "qty": payload.qty,
            "order_value_inr": payload.order_value_inr,
            "promised_delivery_days": payload.promised_delivery_days,
            "delivery_pincode": str(payload.delivery_pincode).zfill(6),
            "is_gift": payload.is_gift,
            "customer_prior_orders": payload.customer_prior_orders,
            "customer_prior_returns": payload.customer_prior_returns,
            "delivery_note": payload.delivery_note or "",
            "shield_member": payload.shield_member,
            "state": payload.state.upper(),
            "signup_date": payload.signup_date,
            "family": product["family"],
            "model_name": product["model_name"],
            "list_price_inr": product["list_price_inr"],
            "warranty_months": product["warranty_months"],
            "launch_date": product["launch_date"],
        }

        df = fix_paise_order_values(pd.DataFrame([row_dict]))
        feat_df = build_feature_frame(df)
        X = bundle["preprocessor"].transform(feat_df)
        score = float(bundle["model"].predict_proba(X)[0, 1])

        threshold = float(bundle["recommended_threshold"])
        if score >= 0.25:
            risk_band = "high"
        elif score >= threshold:
            risk_band = "medium"
        else:
            risk_band = "low"

        if score >= threshold:
            if payload.shield_member == "Y":
                action = "Courtesy confirmation check (Shield member - do not auto-hold)"
            else:
                action = "Pre-dispatch confirmation call (Rs 45) - do not auto-hold"
        else:
            action = "Standard dispatch (no call needed)"

        reasons = explain_single_prediction(
            model=bundle["model"],
            model_name=bundle["model_name"],
            preprocessor=bundle["preprocessor"],
            feature_row_df=feat_df,
            raw_row=df.iloc[0],
            top_k=4,
        )

        # Log only non-PII operational telemetry
        logger.info(
            "Scored order: sku=%s channel=%s payment=%s score=%.4f band=%s",
            payload.sku,
            payload.sales_channel,
            payload.payment_mode,
            score,
            risk_band,
        )

        return {
            "score": round(score, 6),
            "risk_pct": round(score * 100.0, 2),
            "risk_band": risk_band,
            "threshold_used": threshold,
            "recommended_action": action,
            "reasons": reasons,
            "caveat": "Model risk drivers reflect historical associations, not proven causal effects.",
        }

    return app


app = create_app()
