import pandas as pd
import pytest

from src.data import read_sales, weekly_sales
from src.features import make_features, split_dates


def sample_raw():
    dates = pd.date_range("2020-01-03", periods=90, freq="W-FRI")
    return pd.DataFrame([
        {"Store": store, "Dept": dept, "Date": date,
         "Weekly_Sales": store * 1000 + i * 10 + dept,
         "IsHoliday": False}
        for store in [1, 2] for i, date in enumerate(dates) for dept in [1, 2]
    ])


def test_aggregation():
    raw = sample_raw()
    weekly = weekly_sales(raw)
    assert len(weekly) == 180
    assert weekly.iloc[0]["Weekly_Sales"] == 2003
    assert weekly["Weekly_Sales"].sum() == raw["Weekly_Sales"].sum()


def test_missing_week_rejected():
    raw = sample_raw()
    bad = raw.drop(index=[0, 1])
    with pytest.raises(ValueError, match="Missing store/week"):
        weekly_sales(bad)


def test_duplicate_rejected(tmp_path):
    raw = sample_raw()
    path = tmp_path / "train.csv"
    pd.concat([raw, raw.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_sales(path)


def test_features_use_only_past_within_store():
    weekly = weekly_sales(sample_raw())
    frame = make_features(weekly)
    for store in [1, 2]:
        row = frame[frame["Store"] == store].iloc[0]
        history = weekly[weekly["Store"] == store]["Weekly_Sales"].reset_index(drop=True)
        assert row["lag_1"] == history.iloc[51]
        assert row["lag_52"] == history.iloc[0]
        assert row["rolling_mean_4"] == history.iloc[48:52].mean()

    changed = weekly.copy()
    date = frame["Date"].min()
    changed.loc[changed["Date"] >= date, "Weekly_Sales"] += 99999
    before = frame[frame["Date"] == date].drop(columns="Weekly_Sales")
    after = make_features(changed).query("Date == @date").drop(columns="Weekly_Sales")
    pd.testing.assert_frame_equal(before, after)


def test_chronological_split():
    frame = make_features(weekly_sales(sample_raw()))
    train, val, test = split_dates(frame)
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()
    assert val["Date"].nunique() == test["Date"].nunique() == 13
