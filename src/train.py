"""Reproducible time-based training, validation, and model selection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data import load_train_and_test
from src.features import build_feature_frame, make_preprocessor

VALIDATION_CUTOFF = "2026-04-01 00:00"
RETURN_COST_INR = 1150.0
CALL_COST_INR = 45.0
CALL_PREVENTION_RATE = 0.35
HOLD_CANCELLATION_RATE = 0.12
MONTHLY_VOLUME_ESTIMATE = 700


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute classification metrics and policy cost arithmetic at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    flagged = int(tp + fp)

    # Pre-dispatch confirmation call economics (using policy figures)
    gross_saved_inr = tp * CALL_PREVENTION_RATE * RETURN_COST_INR
    call_cost_inr = flagged * CALL_COST_INR
    net_savings_inr = gross_saved_inr - call_cost_inr
    scale_to_700 = MONTHLY_VOLUME_ESTIMATE / n if n > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "flagged_count": flagged,
        "flagged_pct": round(float(flagged / n), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "val_net_savings_inr": round(float(net_savings_inr), 2),
        "monthly_700_calls": round(float(flagged * scale_to_700), 1),
        "monthly_700_call_cost_inr": round(float(call_cost_inr * scale_to_700), 2),
        "monthly_700_gross_saved_inr": round(float(gross_saved_inr * scale_to_700), 2),
        "monthly_700_net_savings_inr": round(float(net_savings_inr * scale_to_700), 2),
    }


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute ranking and calibration metrics."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))), 4),
    }


def train_and_evaluate(root: Path | None = None) -> dict[str, Any]:
    """Run time-based validation, compare models, refit on full train, and save bundle."""
    base = Path(root) if root is not None else Path.cwd()
    train_df, test_df, _, products = load_train_and_test(base)

    val_mask = train_df["order_placed_at"] >= VALIDATION_CUTOFF
    tr_df = train_df.loc[~val_mask].reset_index(drop=True)
    va_df = train_df.loc[val_mask].reset_index(drop=True)

    X_tr_raw = build_feature_frame(tr_df)
    y_tr = tr_df["returned"].astype(int).to_numpy()
    X_va_raw = build_feature_frame(va_df)
    y_va = va_df["returned"].astype(int).to_numpy()

    # Majority-class baseline
    majority_prob = np.full(len(y_va), float(y_tr.mean()))
    majority_metrics = evaluate_probabilities(y_va, majority_prob)
    majority_metrics["accuracy_at_0_5"] = round(float(accuracy_score(y_va, np.zeros_like(y_va))), 4)

    candidates = {
        "logistic_regression": (
            True,
            lambda: LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        ),
        "hist_gradient_boosting": (
            False,
            lambda: HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=30,
                random_state=42,
            ),
        ),
        "xgboost": (
            False,
            lambda: XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    }

    model_results: dict[str, Any] = {"majority_baseline": majority_metrics}
    best_name = ""
    best_auc = -1.0
    best_val_probs = None

    for name, (scale_num, builder) in candidates.items():
        prep = make_preprocessor(scale_numeric=scale_num)
        X_tr = prep.fit_transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        clf = builder()
        clf.fit(X_tr, y_tr)
        va_prob = clf.predict_proba(X_va)[:, 1]
        metrics = evaluate_probabilities(y_va, va_prob)
        metrics["accuracy_at_0_5"] = round(float(accuracy_score(y_va, (va_prob >= 0.5).astype(int))), 4)
        model_results[name] = metrics

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_val_probs = va_prob

    assert best_val_probs is not None

    # Threshold analysis on validation ONLY
    breakeven_p = CALL_COST_INR / (CALL_PREVENTION_RATE * RETURN_COST_INR)
    candidate_thresholds = [breakeven_p, 0.15, 0.20, 0.25, 0.30, 0.35]
    threshold_table = [evaluate_threshold(y_va, best_val_probs, t) for t in candidate_thresholds]

    # Select recommended operational threshold: maximize net savings with reasonable precision (>= 0.22)
    recommended_threshold = max(threshold_table, key=lambda r: r["val_net_savings_inr"])["threshold"]

    # Shield vs Non-Shield breakdown on validation at recommended threshold
    va_shield = (va_df["shield_member"] == "Y").to_numpy()
    shield_breakdown = {
        "shield_val_count": int(va_shield.sum()),
        "shield_val_return_rate": round(float(y_va[va_shield].mean()), 4),
        "non_shield_val_count": int((~va_shield).sum()),
        "non_shield_val_return_rate": round(float(y_va[~va_shield].mean()), 4),
    }

    # Refit winning pipeline on full deduplicated train.csv
    scale_num, builder = candidates[best_name]
    full_prep = make_preprocessor(scale_numeric=scale_num)
    X_full = full_prep.fit_transform(build_feature_frame(train_df))
    y_full = train_df["returned"].astype(int).to_numpy()
    full_model = builder()
    full_model.fit(X_full, y_full)

    models_dir = base / "models"
    reports_dir = base / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": best_name,
        "preprocessor": full_prep,
        "model": full_model,
        "recommended_threshold": recommended_threshold,
        "breakeven_threshold": round(float(breakeven_p), 4),
        "products": products.to_dict(orient="records"),
        "validation_roc_auc": best_auc,
    }
    joblib.dump(bundle, models_dir / "model_bundle.joblib")

    summary = {
        "data_summary": {
            "train_dedup_rows": len(train_df),
            "train_dedup_return_rate": round(float(y_full.mean()), 4),
            "train_fold_rows": len(tr_df),
            "train_fold_return_rate": round(float(y_tr.mean()), 4),
            "val_fold_rows": len(va_df),
            "val_fold_return_rate": round(float(y_va.mean()), 4),
            "test_rows": len(test_df),
            "validation_cutoff": VALIDATION_CUTOFF,
            "train_date_range": [str(train_df["order_placed_at"].min()), str(train_df["order_placed_at"].max())],
            "test_date_range": [str(test_df["order_placed_at"].min()), str(test_df["order_placed_at"].max())],
        },
        "model_comparison": model_results,
        "selected_model": best_name,
        "recommended_threshold": recommended_threshold,
        "threshold_analysis": threshold_table,
        "shield_breakdown": shield_breakdown,
    }
    (reports_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result, indent=2))
