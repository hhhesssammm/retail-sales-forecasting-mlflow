import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS

SEED = 42


class NaiveForecast(RegressorMixin, BaseEstimator):
    def __init__(self, lag=1):
        self.lag = lag

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "fitted_")
        return X[f"lag_{self.lag}"].to_numpy(dtype=float)


class StoreSARIMAX(RegressorMixin, BaseEstimator):
    """Fit a separate SARIMAX(p,0,0) with intercept for each store.

    No MA or differencing terms are used, so a one-step forecast can be
    computed from the latest p actual sales and the fitted AR coefficients.
    This lets the same model accept feature rows as the other estimators.
    """

    def __init__(self, p=1):
        self.p = p

    def fit(self, X, y):
        if self.p not in (1, 2):
            raise ValueError("Supported AR orders are 1 and 2")
        data = X[["Store", "trend"]].copy()
        data["sales"] = np.asarray(y, dtype=float)
        self.coefficients_ = {}
        self.nonconverged_stores_ = []
        for store, group in data.groupby("Store"):
            values = group.sort_values("trend")["sales"].to_numpy()
            center = float(values.mean())
            scale = max(float(values.std()), 1.0)
            result = SARIMAX(
                (values - center) / scale,
                order=(self.p, 0, 0),
                trend="c",
            ).fit(disp=False, maxiter=200)
            params = dict(zip(result.param_names, result.params))
            if not np.isfinite(result.params).all():
                raise ValueError(f"Non-finite SARIMAX parameters for store {store}")
            if not result.mle_retvals.get("converged", False):
                self.nonconverged_stores_.append(int(store))
            self.coefficients_[store] = (
                center, scale, float(params["intercept"]),
                [float(params[f"ar.L{i}"]) for i in range(1, self.p + 1)],
            )
        return self

    def predict(self, X):
        check_is_fitted(self, "coefficients_")
        predictions = np.empty(len(X), dtype=float)
        stores = X["Store"].to_numpy()
        for store in pd.unique(stores):
            if store not in self.coefficients_:
                raise ValueError(f"No trained SARIMAX model for store {store}")
            mask = stores == store
            center, scale, intercept, ar = self.coefficients_[store]
            forecast = np.full(mask.sum(), intercept, dtype=float)
            for i, coefficient in enumerate(ar, start=1):
                lag = X.loc[mask, f"lag_{i}"].to_numpy(dtype=float)
                forecast += coefficient * (lag - center) / scale
            predictions[mask] = center + scale * forecast
        return predictions


# A small, fixed search space keeps the comparison reproducible.
CANDIDATES = [
    ("naive_last_week", "Naive", {"lag": 1}),
    ("naive_last_year", "Naive", {"lag": 52}),
    ("linear_intercept", "LinearRegression", {"fit_intercept": True}),
    ("linear_no_intercept", "LinearRegression", {"fit_intercept": False}),
    ("rf_shallow", "RandomForest", {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 3}),
    ("rf_deep", "RandomForest", {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}),
    ("xgb_shallow", "XGBoost", {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05}),
    ("xgb_deep", "XGBoost", {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.03}),
    ("sarimax_ar1", "SARIMAX", {"p": 1}),
    ("sarimax_ar2", "SARIMAX", {"p": 2}),
]


def build_model(family, params):
    if family == "Naive":
        return NaiveForecast(**params)
    if family == "SARIMAX":
        return StoreSARIMAX(**params)
    if family == "LinearRegression":
        estimator = LinearRegression(**params)
    elif family == "RandomForest":
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    elif family == "XGBoost":
        estimator = XGBRegressor(
            objective="reg:squarederror", random_state=SEED, n_jobs=1,
            tree_method="hist", **params,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    # Store is an identifier, not an ordinal numerical variable.
    preprocessor = ColumnTransformer([
        ("store", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Store"]),
        ("numeric", "passthrough", [c for c in FEATURE_COLUMNS if c != "Store"]),
    ])
    return Pipeline([("features", preprocessor), ("model", estimator)])
