import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import DATA_PATH, ROOT, load_sales

OUTPUT = ROOT / "reports" / "eda"


def analyze(weekly, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dates = sorted(weekly["Date"].unique())
    if len(dates) <= 26:
        raise ValueError("Need more than 26 weeks for EDA and holdout windows")
    # Leave validation and test sales out of exploratory model decisions.
    train = weekly[weekly["Date"] < dates[-26]].copy()
    totals = train.groupby("Date")["Weekly_Sales"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    totals.plot(ax=ax, label="Weekly total")
    totals.rolling(4).mean().plot(ax=ax, label="4-week mean")
    ax.set(title="Weekly sales over time (training period)", ylabel="Sales (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "sales_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    train["Weekly_Sales"].hist(bins=40, ax=ax)
    ax.set(title="Store-week sales distribution", xlabel="Sales (USD)", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(output / "sales_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    totals.groupby(totals.index.month).mean().plot.bar(ax=ax)
    ax.set(title="Mean weekly chain sales by month", xlabel="Month", ylabel="Sales (USD)")
    fig.tight_layout()
    fig.savefig(output / "monthly_seasonality.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    train.boxplot(column="Weekly_Sales", by="IsHoliday", ax=ax)
    ax.set(title="Holiday versus regular weeks", ylabel="Sales (USD)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output / "holiday_sales.png", dpi=150)
    plt.close(fig)

    group = train.groupby("Store")["Weekly_Sales"]
    q1 = group.transform(lambda s: s.quantile(.25))
    q3 = group.transform(lambda s: s.quantile(.75))
    iqr = q3 - q1
    outliers = train[(train["Weekly_Sales"] < q1 - 1.5 * iqr) |
                     (train["Weekly_Sales"] > q3 + 1.5 * iqr)]
    outliers.to_csv(output / "outliers.csv", index=False)
    summary = {
        "store_weeks": len(weekly),
        "stores": int(weekly["Store"].nunique()),
        "first_date": str(weekly["Date"].min().date()),
        "last_date": str(weekly["Date"].max().date()),
        "missing_values": weekly.isna().sum().to_dict(),
        "eda_training_end": str(train["Date"].max().date()),
        "training_iqr_outliers": len(outliers),
        "note": "IQR flags are diagnostic only; holiday peaks are not removed.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    print(json.dumps(analyze(load_sales(args.data)), indent=2))
