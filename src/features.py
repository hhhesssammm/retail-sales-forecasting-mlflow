import pandas as pd

LAGS = (1, 2, 4, 13, 52)
WINDOWS = (4, 13)
FEATURE_COLUMNS = [
    "Store", "IsHoliday", "year", "month", "week", "day_of_week", "trend",
    *[f"lag_{n}" for n in LAGS],
    *[f"rolling_mean_{n}" for n in WINDOWS],
    "trend_4w",
]


def make_features(weekly):
    """All sales-derived features use information from earlier weeks."""
    frame = weekly.sort_values(["Store", "Date"]).copy()
    date = frame["Date"].dt
    frame["year"] = date.year
    frame["month"] = date.month
    frame["week"] = date.isocalendar().week.astype(int)
    frame["day_of_week"] = date.dayofweek
    frame["trend"] = (frame["Date"] - frame["Date"].min()).dt.days // 7
    frame["IsHoliday"] = frame["IsHoliday"].astype(int)
    sales = frame.groupby("Store")["Weekly_Sales"]
    for n in LAGS:
        frame[f"lag_{n}"] = sales.shift(n)
    for n in WINDOWS:
        frame[f"rolling_mean_{n}"] = sales.transform(
            lambda s: s.shift(1).rolling(n).mean()
        )
    frame["trend_4w"] = frame["lag_1"] - frame["lag_4"]
    return frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def split_dates(frame, validation_weeks=13, test_weeks=13):
    if validation_weeks < 1 or test_weeks < 1:
        raise ValueError("Validation and test windows must be positive")
    dates = pd.DatetimeIndex(frame["Date"].unique()).sort_values()
    if len(dates) <= validation_weeks + test_weeks:
        raise ValueError("Not enough weeks for training, validation and test")
    validation_start = dates[-(validation_weeks + test_weeks)]
    test_start = dates[-test_weeks]
    return (
        frame.loc[frame["Date"] < validation_start].copy(),
        frame.loc[(frame["Date"] >= validation_start) & (frame["Date"] < test_start)].copy(),
        frame.loc[frame["Date"] >= test_start].copy(),
    )
