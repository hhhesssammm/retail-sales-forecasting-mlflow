from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Expected matching, non-empty one-dimensional forecasts")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Forecasts and actuals must be finite")
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined for an all-zero target")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / denominator * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def save_forecasts(frame, predictions, directory, split):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = frame[["Store", "Date", "Weekly_Sales"]].rename(
        columns={"Weekly_Sales": "actual"}
    ).copy()
    result["predicted"] = np.asarray(predictions)
    result["error"] = result["actual"] - result["predicted"]
    result.sort_values(["Date", "Store"]).to_csv(directory / f"{split}_forecasts.csv", index=False)

    totals = result.groupby("Date")[["actual", "predicted"]].sum()
    store_id = result["Store"].min()
    store = result[result["Store"] == store_id].set_index("Date").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, values, title in [
        (axes[0], totals, "All stores (total)"),
        (axes[1], store[["actual", "predicted"]], f"Store {store_id}"),
    ]:
        ax.plot(values.index, values["actual"], label="Actual", marker="o", markersize=3)
        ax.plot(values.index, values["predicted"], label="Forecast", linestyle="--")
        ax.set(title=title, ylabel="Sales (USD)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.suptitle(f"{split.capitalize()}: actual vs forecast")
    fig.tight_layout()
    fig.savefig(directory / f"{split}_forecast.png", dpi=150)
    plt.close(fig)


def save_comparison(table, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = table.sort_values("rmse", ascending=False)
    ax.barh(ordered["run_name"], ordered["rmse"])
    ax.set(xlabel="Validation RMSE (USD)", title="Model comparison (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
