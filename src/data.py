"""Data loading, deduplication, reference joining, and data-quality fixes."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pandas as pd

LEAKY_OR_EXCLUDED_COLUMNS = (
    "last_service_event_type",
    "pickup_scheduled_at",
    "source",
    "returned",
)


def resolve_pack_file(root: Path, suffix: str) -> Path:
    """Locate a task-pack file either by exact name or UUID-prefixed name."""
    exact = root / suffix
    if exact.exists():
        return exact
    matches = sorted(root.glob(f"*-{suffix}"))
    if not matches:
        matches = sorted(root.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Could not find task-pack file ending in '{suffix}' under {root}")
    return matches[0]


def load_reference_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customers and products reference tables with validated unique keys."""
    customers_path = resolve_pack_file(root, "customers.csv")
    products_path = resolve_pack_file(root, "products.csv")

    customers = pd.read_csv(customers_path, dtype={"customer_id": str, "shield_member": str})
    products = pd.read_csv(products_path, dtype={"sku": str})

    if not customers["customer_id"].is_unique:
        raise ValueError("customers.csv has duplicate customer_id keys")
    if not products["sku"].is_unique:
        raise ValueError("products.csv has duplicate sku keys")

    return customers, products


def deduplicate_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate partner-feed re-imports by order_id, preferring crm records."""
    if df["order_id"].is_unique:
        return df.copy()

    out = df.copy()
    out["_source_rank"] = (out["source"] != "crm").astype(int)
    out = (
        out.sort_values(["order_id", "_source_rank"], kind="mergesort")
        .drop_duplicates(subset=["order_id"], keep="first")
        .drop(columns=["_source_rank"])
        .reset_index(drop=True)
    )
    return out


def fix_paise_order_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize October 2025 payment-gateway values stored in paise (100x INR)."""
    out = df.copy()
    expected = out["list_price_inr"] * out["qty"] * (1.0 - out["discount_pct"] / 100.0)
    ratio = out["order_value_inr"] / expected.clip(lower=1.0)
    is_paise = ratio > 10.0
    out["order_value_inr"] = out["order_value_inr"].where(~is_paise, out["order_value_inr"] / 100.0)
    return out


def prepare_orders_dataframe(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Join reference tables and apply deterministic pre-dispatch data-quality fixes."""
    df = deduplicate_orders(orders) if deduplicate else orders.copy()

    df["delivery_pincode"] = df["delivery_pincode"].astype(str).str.zfill(6)
    df["delivery_note"] = df["delivery_note"].fillna("")

    df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(products, on="sku", how="left", validate="many_to_one")

    if df["shield_member"].isna().any():
        missing_count = int(df["shield_member"].isna().sum())
        raise ValueError(f"{missing_count} orders failed to match customer_id in customers.csv")
    if df["list_price_inr"].isna().any():
        missing_count = int(df["list_price_inr"].isna().sum())
        raise ValueError(f"{missing_count} orders failed to match sku in products.csv")

    df = fix_paise_order_values(df)
    return df


def load_train_and_test(root: Optional[Path] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare deduplicated train and test DataFrames plus reference tables."""
    base = Path(root) if root is not None else Path.cwd()
    customers, products = load_reference_tables(base)

    train_raw = pd.read_csv(
        resolve_pack_file(base, "train.csv"),
        dtype={"order_id": str, "customer_id": str, "sku": str, "delivery_pincode": str},
    )
    test_raw = pd.read_csv(
        resolve_pack_file(base, "test_unlabelled.csv"),
        dtype={"order_id": str, "customer_id": str, "sku": str, "delivery_pincode": str},
    )

    train_df = prepare_orders_dataframe(train_raw, customers, products, deduplicate=True)
    test_df = prepare_orders_dataframe(test_raw, customers, products, deduplicate=False)
    return train_df, test_df, customers, products
