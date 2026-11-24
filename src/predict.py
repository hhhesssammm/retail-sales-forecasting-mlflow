import argparse
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.data import DATA_PATH, ROOT, load_sales
from src.features import FEATURE_COLUMNS, make_features
from src.tracking import REGISTERED_MODEL, configure_tracking


def next_week_features(weekly, holiday):
    next_date = weekly["Date"].max() + pd.Timedelta(weeks=1)
    future = pd.DataFrame({
        "Store": sorted(weekly["Store"].unique()),
        "Date": next_date,
        "IsHoliday": holiday,
        "Weekly_Sales": np.nan,
    })
    features = make_features(pd.concat([weekly, future], ignore_index=True))
    rows = features[features["Date"] == next_date].copy()
    if len(rows) != len(future):
        raise ValueError("Not enough past sales for every store")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Predict the next week using a registered model")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--holiday", choices=["yes", "no"], required=True,
                        help="Whether the upcoming week is a holiday week")
    parser.add_argument("--alias", choices=["candidate", "champion"], default="candidate")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "forecasts" / "next_week.csv")
    args = parser.parse_args()
    configure_tracking()
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@{args.alias}")
    rows = next_week_features(load_sales(args.data), args.holiday == "yes")
    result = rows[["Store", "Date"]].copy()
    result["forecast"] = model.predict(rows[FEATURE_COLUMNS])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
