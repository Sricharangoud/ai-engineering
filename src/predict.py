"""Batch prediction script to generate and verify predictions.csv."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_train_and_test, resolve_pack_file
from src.features import build_feature_frame


def generate_predictions(root: Path | None = None) -> Path:
    """Score test_unlabelled.csv and write predictions.csv matching sample_submission.csv."""
    base = Path(root) if root is not None else Path.cwd()
    bundle_path = base / "models" / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Run python -m src.train first.")

    bundle = joblib.load(bundle_path)
    _, test_df, _, _ = load_train_and_test(base)

    X_test = bundle["preprocessor"].transform(build_feature_frame(test_df))
    scores = bundle["model"].predict_proba(X_test)[:, 1]

    out_df = pd.DataFrame(
        {
            "order_id": test_df["order_id"],
            "score": pd.Series(scores).round(6),
        }
    )

    sample_path = resolve_pack_file(base, "sample_submission.csv")
    sample_df = pd.read_csv(sample_path, dtype={"order_id": str})

    # Strict format verification against sample_submission.csv
    if list(out_df.columns) != list(sample_df.columns):
        raise ValueError(f"Column mismatch: {list(out_df.columns)} != {list(sample_df.columns)}")
    if len(out_df) != len(sample_df):
        raise ValueError(f"Row count mismatch: {len(out_df)} != {len(sample_df)}")
    if not out_df["order_id"].equals(sample_df["order_id"]):
        raise ValueError("order_id sequence does not match sample_submission.csv")
    if out_df["score"].isna().any():
        raise ValueError("predictions.csv contains NaN scores")
    if not ((out_df["score"] >= 0.0) & (out_df["score"] <= 1.0)).all():
        raise ValueError("Scores must be in [0, 1]")

    out_path = base / "predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} ({len(out_df)} rows). "
        f"Score min={out_df['score'].min():.4f}, median={out_df['score'].median():.4f}, max={out_df['score'].max():.4f}"
    )
    return out_path


if __name__ == "__main__":
    generate_predictions()
