#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flu Forecasting with ARIMA - Based on CDC ILI Data (Georgia only)

Goal of this simplified script:
    - Fit an ARIMA model on 4 years of historical data (Georgia)
    - Forecast the 2024–2025 flu season
    - Export all relevant time series (train, test, fitted, forecast, CIs)
      to a single Excel file that you can open directly in Excel.

No missingness experiment, no observation-level experiment, no plotting.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# ---------------------- Config ----------------------
STATE_NAME = "Georgia"

# Path to your wide-format CSV
CSV_PATH = "/Users/hanzeqin/Desktop/Research/llm_for_public health/wide_filled.csv"

DATE_START = pd.Timestamp("2015-09-01")
DATE_END   = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2024-09-01")
TEST_END   = pd.Timestamp("2025-09-30")

# ARIMA hyperparameters
ARIMA_ORDER    = (2, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 52)
MAXITER        = 100

# Output directory: same folder as the input CSV
OUT_DIR = os.path.dirname(CSV_PATH) if CSV_PATH else os.getcwd()


# ---------------------- IO: YEARWEEK (wide) ----------------------
def _parse_yearweek_to_sunday(ser: pd.Series) -> pd.DatetimeIndex:
    """
    Parse YEARWEEK (YYYYWW, ISO week) to the Sunday date of that week.
    Steps:
        1) Use ISO year-week to get Monday
        2) Add 6 days → Sunday
    """
    ser = ser.astype(str).str.strip()
    # %G = ISO year, %V = ISO week, %u = weekday (1=Mon,...,7=Sun)
    monday = pd.to_datetime(ser + "-1", format="%G%V-%u", errors="coerce")
    if monday.isna().any():
        bad = ser[monday.isna()].unique()[:5]
        raise ValueError(f"Unable to parse YEARWEEK (examples): {bad}")
    sunday = monday + pd.to_timedelta(6, unit="D")
    return pd.DatetimeIndex(sunday)


def load_georgia_series_from_wide_csv(path: str) -> pd.Series:
    """
    Load the wide CSV, extract the Georgia column, and:
        - align to weekly Sundays (W-SUN)
        - restrict to DATE_START ~ DATE_END
        - linearly interpolate small gaps
    Returns a pandas Series indexed by Sunday dates.
    """
    df = pd.read_csv(path, low_memory=False)
    if "YEARWEEK" not in df.columns:
        raise ValueError("CSV is missing YEARWEEK column.")
    if STATE_NAME not in df.columns:
        if "GA" in df.columns:
            df = df.rename(columns={"GA": STATE_NAME})
        else:
            raise ValueError("CSV is missing Georgia column (or GA).")

    idx_sun = _parse_yearweek_to_sunday(df["YEARWEEK"])
    g = pd.Series(df[STATE_NAME].astype(float).values, index=idx_sun).sort_index()

    # Restrict to analysis window and align to weekly Sundays
    g = g.loc[(g.index >= DATE_START) & (g.index <= DATE_END)]
    full_index = pd.date_range(
        start=g.index.min().normalize(),
        end=g.index.max().normalize(),
        freq="W-SUN"
    )
    g = g.reindex(full_index)
    missing_before = int(g.isna().sum())

    # Linear interpolation + forward/backward fill
    g = g.interpolate(method="linear", limit_direction="both") \
         .fillna(method="ffill") \
         .fillna(method="bfill")

    print(
        f"[Data] Georgia series: {g.index.min().date()} — {g.index.max().date()}, "
        f"weeks={len(g)}, missing (pre-interp)={missing_before}"
    )
    return g


# ---------------------- Model ----------------------
class FluForecaster:
    """
    Seasonal ARIMA (SARIMAX) forecaster for a single time series.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = ARIMA_ORDER,
        seasonal_order: Tuple[int, int, int, int] = SEASONAL_ORDER,
        maxiter: int = MAXITER
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self.results = None

    def fit(self, train_data: pd.Series) -> None:
        """
        Fit SARIMAX using a numpy array (to avoid index-related quirks).
        """
        y = train_data.dropna().to_numpy()
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        try:
            self.results = model.fit(disp=False, maxiter=self.maxiter)
        except Exception as e:
            print(f"[WARN] Fit failed ({e}); fallback to simpler model (1,0,1)x(1,0,1,52).")
            model = SARIMAX(
                y,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.results = model.fit(disp=False, maxiter=max(50, self.maxiter // 2))

    def predict(self, steps: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forecast the next `steps` points.
        Returns:
            - mean: point forecasts
            - ci:   95% confidence intervals (2D array: [steps, 2])
        """
        fc = self.results.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean).reshape(-1)
        ci = np.asarray(fc.conf_int())
        return mean, ci

    def fitted(self) -> np.ndarray:
        """
        In-sample fitted values on the training window.
        """
        fv = self.results.fittedvalues
        return np.asarray(fv).reshape(-1)


# ---------------------- Splits ----------------------
def make_splits(series: pd.Series, train_years: int) -> Tuple[pd.Series, pd.Series]:
    """
    Create train and test splits based on calendar dates:
        - For 4-year training:
            train:  2020-09-01 ~ 2024-08-31
        - For 9-year training:
            train:  2015-09-01 ~ 2024-08-31
        - Test (for both): 2024-09-01 ~ 2025-09-30
    """
    if train_years == 4:
        train_start = pd.Timestamp("2020-09-01")
    elif train_years == 9:
        train_start = pd.Timestamp("2015-09-01")
    else:
        raise ValueError("train_years must be 4 or 9.")

    train_end = TEST_START - pd.Timedelta(days=1)

    train = series.loc[(series.index >= train_start) & (series.index <= train_end)]
    test  = series.loc[(series.index >= TEST_START) & (series.index <= TEST_END)]

    return train, test


# ---------------------- Main ----------------------
def main():
    print("=" * 80)
    print("Flu Forecasting with ARIMA - Georgia only (export forecast data to Excel)")
    print("=" * 80)
    print(f"[Path] Input CSV: {CSV_PATH}")
    print(f"[Path] Output directory: {OUT_DIR}")

    # 1) Load & preprocess series for Georgia
    series = load_georgia_series_from_wide_csv(CSV_PATH)

    # 2) Make 4-year train / 1-year test split
    train4, test = make_splits(series, train_years=4)
    print(f"[Split] 4y train weeks={len(train4)}, test weeks={len(test)}")

    # 3) Fit ARIMA model on training data and forecast the test period
    model = FluForecaster()
    model.fit(train4)
    fc, ci = model.predict(len(test))
    fitted = model.fitted()

    # 4) Build a unified DataFrame for export
    #    Index: union of train and test dates (in chronological order)
    full_index = train4.index.append(test.index)
    df_pred = pd.DataFrame(index=full_index)

    # Training and test data
    df_pred["train"] = train4
    df_pred["test"]  = test

    # In-sample fitted values (aligned with training period only)
    df_pred.loc[train4.index, "fitted"] = fitted

    # Forecast values and confidence intervals on the test period
    df_pred.loc[test.index, "forecast"] = fc
    if ci is not None and len(ci) == len(test):
        df_pred.loc[test.index, "ci_lower"] = ci[:, 0]
        df_pred.loc[test.index, "ci_upper"] = ci[:, 1]

    # 5) Save to Excel (single sheet)
    excel_path = os.path.join(OUT_DIR, "GA_ARIMA_forecast_example_data.xlsx")
    df_pred.to_excel(excel_path, sheet_name="Georgia_ARIMA", index_label="date")

    print(f"[Saved] Forecast data exported to Excel: {excel_path}")
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
