import numpy as np
import pandas as pd
import pytest

from src.evaluation import forecast_metrics
from src.features import FEATURE_COLUMNS, make_features, split_dates
from src.models import build_model
from src.predict import next_week_features


@pytest.fixture
def weekly():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    return pd.DataFrame([
        {"Store": store, "Date": date, "IsHoliday": False,
         "Weekly_Sales": 1000 * store + i * 3 + rng.normal(0, 20)}
        for store in [1, 2] for i, date in enumerate(dates)
    ])


def test_metrics():
    metrics = forecast_metrics([10, 20], [12, 18])
    assert metrics["mae"] == 2
    assert metrics["rmse"] == 2
    assert metrics["wape"] == pytest.approx(400 / 30)
    assert metrics["r2"] == pytest.approx(0.84)


@pytest.mark.parametrize("family,params", [
    ("Naive", {"lag": 1}),
    ("LinearRegression", {"fit_intercept": True}),
    ("RandomForest", {"n_estimators": 3, "max_depth": 3}),
    ("XGBoost", {"n_estimators": 3, "max_depth": 2}),
    ("SARIMAX", {"p": 1}),
    ("SARIMAX", {"p": 2}),
])
def test_model_predicts_finite_values(weekly, family, params):
    train, val, _ = split_dates(make_features(weekly))
    model = build_model(family, params)
    model.fit(train[FEATURE_COLUMNS], train["Weekly_Sales"])
    forecast = model.predict(val[FEATURE_COLUMNS])
    assert len(forecast) == len(val)
    assert np.isfinite(forecast).all()
    if family == "Naive":
        np.testing.assert_allclose(forecast, val["lag_1"])


def test_sarimax_prediction_matches_statsmodels(weekly):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train, val, _ = split_dates(make_features(weekly))
    model = build_model("SARIMAX", {"p": 2})
    model.fit(train[FEATURE_COLUMNS], train["Weekly_Sales"])
    for store, group in train.groupby("Store"):
        y = group.sort_values("Date")["Weekly_Sales"].to_numpy()
        center, scale = y.mean(), y.std()
        fitted = SARIMAX((y - center) / scale, order=(2, 0, 0), trend="c").fit(
            disp=False, maxiter=200
        )
        first = val[val["Store"] == store].sort_values("Date").head(1)
        expected = center + scale * fitted.forecast(1)[0]
        assert model.predict(first[FEATURE_COLUMNS])[0] == pytest.approx(expected, rel=1e-6)


def test_next_week_uses_latest_sales(weekly):
    features = next_week_features(weekly, True)
    assert len(features) == 2
    assert features["Date"].min() == weekly["Date"].max() + pd.Timedelta(weeks=1)
    assert features["IsHoliday"].eq(1).all()
    assert features["Weekly_Sales"].isna().all()
    expected = weekly.groupby("Store")["Weekly_Sales"].last()
    np.testing.assert_allclose(features.sort_values("Store")["lag_1"], expected)
