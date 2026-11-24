from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"


def read_sales(path=DATA_PATH):
    """Read and validate the original department-level training data."""
    raw = pd.read_csv(path)
    required = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    raw = raw[required].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    raw["Weekly_Sales"] = pd.to_numeric(raw["Weekly_Sales"], errors="raise")
    if raw.isna().any().any() or not np.isfinite(raw["Weekly_Sales"]).all():
        raise ValueError("Missing or non-finite values in required fields")
    if not raw["IsHoliday"].isin([True, False, 0, 1]).all():
        raise ValueError("Invalid holiday flag")
    if raw.duplicated(["Store", "Dept", "Date"]).any():
        raise ValueError("Duplicate store/department/week records")
    if (raw["Date"].dt.dayofweek != 4).any():
        raise ValueError("Expected Friday-ending weekly observations")
    return raw


def weekly_sales(raw):
    """Aggregate department sales without discarding legitimate returns."""
    keys = ["Store", "Date"]
    if raw.groupby(keys)["IsHoliday"].nunique().gt(1).any():
        raise ValueError("Inconsistent holiday flags within a store/week")
    weekly = raw.groupby(keys, as_index=False).agg(
        Weekly_Sales=("Weekly_Sales", "sum"),
        IsHoliday=("IsHoliday", "first"),
    )
    weekly = weekly.sort_values(keys).reset_index(drop=True)
    dates = pd.date_range(weekly["Date"].min(), weekly["Date"].max(), freq="W-FRI")
    expected = pd.MultiIndex.from_product([weekly["Store"].unique(), dates])
    observed = pd.MultiIndex.from_frame(weekly[keys])
    if len(expected.difference(observed)):
        raise ValueError("Missing store/week observations; investigate before forecasting")
    return weekly


def load_sales(path=DATA_PATH):
    return weekly_sales(read_sales(path))
