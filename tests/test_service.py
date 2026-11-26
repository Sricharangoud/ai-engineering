"""API and UI integration tests for the FastAPI prediction service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.service import create_app

client = TestClient(create_app())

VALID_PAYLOAD = {
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
    "signup_date": "2025-11-10",
}


def test_health_endpoint() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["api_cost_per_prediction_inr"] == 0.0


def test_ui_route_serves_html() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Kestrel Home" in res.text


def test_predict_valid_payload_returns_score_and_reasons() -> None:
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert isinstance(body["reasons"], list) and len(body["reasons"]) > 0
    assert "recommended_action" in body


def test_predict_rejects_unknown_sku() -> None:
    bad = {**VALID_PAYLOAD, "sku": "INVALID-SKU"}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422
    assert "Unknown sku" in res.json()["detail"]


def test_predict_rejects_invalid_prior_returns() -> None:
    bad = {**VALID_PAYLOAD, "customer_prior_orders": 1, "customer_prior_returns": 3}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422
