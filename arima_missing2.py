#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flu Forecasting with ARIMA(1,1,1) - Observation Levels Experiment (All States, 4-year training)
OPTIMIZED VERSION: Parallel Processing + Warm Starting

- Input CSV (wide): YEARWEEK + state columns
- Optimization:
    1. Uses joblib to run states in parallel.
    2. Uses 'warm_start' (passing previous params) to speed up convergence for increasing thresholds.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed  # <--- NEW IMPORTS

warnings.filterwarnings("ignore")

# ---------------------- Paths & Global Config ----------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

CSV_PATH = os.path.join(BASE_DIR, "wide_filled.csv")

DATE_START = pd.Timestamp("2015-09-01")
DATE_END = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2024-09-01")
TEST_END = pd.Timestamp("2025-09-30")

# --- MODEL CONFIGURATION NOTE ---
# Your docstring said "ARIMA(1,1,1) without seasonal component",
# but your code had (0, 1, 1, 52). s=52 makes it VERY slow.
# If you truly want no seasonality, set SEASONAL_ORDER = (0, 0, 0, 0).
# Keeping your original settings below, but Parallelism is required to make s=52 bearable.
ARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 52)
MAXITER = 20

# Observation levels
THRESHOLDS = [0.0, 0.25, 0.33, 0.5, 0.67, 0.75]
THRESHOLD_LABELS = {
    0.0: "0",
    0.25: "1/4",
    0.33: "1/3",
    0.5: "1/2",
    0.67: "2/3",
    0.75: "3/4"
}

# Number of CPU cores to use (-1 means all available)
N_JOBS = -1


# ---------------------- Helper: YEARWEEK -> Sunday ----------------------
def _parse_yearweek_to_sunday(ser: pd.Series) -> pd.DatetimeIndex:
    ser = ser.astype(str).str.strip()
    monday = pd.to_datetime(ser + "-1", format="%G%V-%u", errors="coerce")
    if monday.isna().any():
        bad = ser[monday.isna()].unique()[:5]
        raise ValueError(f"Unable to parse YEARWEEK (example): {bad}")
    sunday = monday + pd.to_timedelta(6, unit="D")
    return pd.DatetimeIndex(sunday)


# ---------------------- Data Loading ----------------------
def load_all_states_from_wide_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df_raw = pd.read_csv(path, low_memory=False)
    if "YEARWEEK" not in df_raw.columns:
        raise ValueError("CSV is missing the 'YEARWEEK' column.")

    state_cols = [c for c in df_raw.columns if c != "YEARWEEK"]
    if not state_cols:
        raise ValueError("No state columns found in CSV besides YEARWEEK.")

    idx_sun = _parse_yearweek_to_sunday(df_raw["YEARWEEK"])
    data = df_raw[state_cols].astype(float)
    data.index = idx_sun
    data = data.sort_index()

    data = data.loc[(data.index >= DATE_START) & (data.index <= DATE_END)]

    full_index = pd.date_range(
        start=data.index.min().normalize(),
        end=data.index.max().normalize(),
        freq="W-SUN"
    )
    data = data.reindex(full_index)
    missing_before = data.isna().sum().sum()
    data = data.interpolate(method="linear", limit_direction="both").ffill().bfill()

    print(
        f"[Data] All states: {data.index.min().date()} -- {data.index.max().date()}, "
        f"weeks={len(data)}, total missing (pre-interp)={missing_before}, "
        f"n_states={data.shape[1]}"
    )
    return data


# ---------------------- Worker Function for Parallelism ----------------------
def process_single_state(
        state_name: str,
        train_series: pd.Series,
        test_series: pd.Series,
        thresholds: List[float]
) -> List[Dict]:
    """
    Process all thresholds for a SINGLE state.
    Running this in one process allows us to use 'warm starting'
    (passing parameters from one threshold to the next).
    """

    state_results = []

    # Basic validity checks
    if test_series.isna().all() or len(test_series.dropna()) < 5:
        return []

    trough_value = float(test_series.min())
    peak_value = float(test_series.max())
    if np.isclose(trough_value, peak_value):
        return []

    # --- WARM STARTING VARIABLES ---
    # We store the trained parameters from the previous threshold
    # to speed up the fit for the next threshold.
    prev_params = None

    for w in thresholds:
        thr_val = (1.0 - w) * trough_value + w * peak_value
        exceed_idx = np.where(test_series.values >= thr_val)[0]

        if len(exceed_idx) == 0:
            continue

        cutoff = int(exceed_idx[0])
        observed = test_series.iloc[:cutoff]
        remaining = test_series.iloc[cutoff:]

        if len(remaining) < 5:
            continue

        # Extended training data
        extended_train = pd.concat([train_series, observed])
        y_train_np = extended_train.dropna().to_numpy()

        # Define Model
        # Note: enforce_stationarity=False makes it faster and more robust for auto-loops
        model = SARIMAX(
            y_train_np,
            order=ARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        try:
            # --- FIT WITH WARM START ---
            # If we have params from a smaller dataset (previous threshold), use them.
            if prev_params is not None:
                res = model.fit(start_params=prev_params, disp=False, maxiter=MAXITER)
            else:
                res = model.fit(disp=False, maxiter=MAXITER)

            # Save params for next iteration (Warm Start)
            prev_params = res.params

        except Exception:
            # Fallback to simple model if complex one fails
            try:
                model_simple = SARIMAX(
                    y_train_np, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False
                )
                res = model_simple.fit(disp=False, maxiter=10)
                # Don't carry over simple params to complex model
                prev_params = None
            except:
                continue

        # Forecast
        try:
            fc_res = res.get_forecast(steps=len(remaining))
            fc = fc_res.predicted_mean
        except:
            continue

        # Metrics
        y_true = remaining.values
        mae = mean_absolute_error(y_true, fc)
        mse = mean_squared_error(y_true, fc)

        pred_peak_idx = int(np.argmax(fc))
        true_peak_idx = int(np.argmax(y_true))
        pte = abs(pred_peak_idx - true_peak_idx)
        pme = float(abs(fc[pred_peak_idx] - y_true[true_peak_idx]))

        state_results.append({
            'state': state_name,
            'threshold': w,
            'threshold_label': THRESHOLD_LABELS.get(w, str(w)),
            'MAE': mae,
            'MSE': mse,
            'peak_timing_error': pte,
            'peak_magnitude_error': pme
        })

    return state_results


# ---------------------- Train/Test Split ----------------------
def make_splits(series_or_df, train_years: int = 4):
    if train_years != 4:
        raise ValueError("This script implements only a 4-year training window.")
    train_start = pd.Timestamp("2020-09-01")
    train_end = TEST_START - pd.Timedelta(days=1)
    train = series_or_df.loc[(series_or_df.index >= train_start) & (series_or_df.index <= train_end)]
    test = series_or_df.loc[(series_or_df.index >= TEST_START) & (series_or_df.index <= TEST_END)]
    return train, test


# ---------------------- Main Experiment Controller ----------------------
def experiment_observation_levels_parallel(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        thresholds: List[float]
) -> Dict[str, List[float]]:
    states = list(train_df.columns)
    print(f"[Exp] Parallel Execution on {len(states)} states using {N_JOBS} cores.")

    # Run Parallel Loop
    # joblib.Parallel returns a list of results (one list per state)
    results_nested = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_state)(
            state,
            train_df[state],
            test_df[state],
            thresholds
        )
        for state in tqdm(states, desc="Processing States", ncols=80)
    )

    # Flatten the list of lists
    flat_results = [item for sublist in results_nested for item in sublist]

    # Convert list of dicts to dict of lists
    final_results = {k: [d[k] for d in flat_results] for k in flat_results[0]} if flat_results else {}

    return final_results


# ---------------------- Plotting ----------------------
def plot_observation_level_impact_all_states(results: Dict[str, List[float]], save_path: str):
    df = pd.DataFrame(results)
    if df.empty:
        print("[WARN] No observation-level results to plot.")
        return

    df = df.sort_values(by='threshold')
    unique_thresholds = sorted(df['threshold'].unique())
    labels = [THRESHOLD_LABELS.get(t, str(t)) for t in unique_thresholds]
    data = [df.loc[df['threshold'] == t, 'MAE'].values for t in unique_thresholds]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showmeans=False)

    means = [float(np.mean(d)) if len(d) else np.nan for d in data]
    plt.plot(range(1, len(unique_thresholds) + 1), means, 'D-', linewidth=2, label='Mean MAE')

    plt.xlabel('Observation Level (Fraction of Seasonal Rise)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Performance Across Observation Levels (ARIMA(1,1,1), All States, 4y Training)')
    plt.legend(loc='best')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")


# ---------------------- Main ----------------------
def main():
    print("=" * 80)
    print("ARIMA(1,1,1) Obs Levels Experiment - Optimized (Parallel + Warm Start)")
    print("=" * 80)

    # 1. Load Data
    try:
        all_states_df = load_all_states_from_wide_csv(CSV_PATH)
    except Exception as e:
        print(f"[Error] Could not load data: {e}")
        return

    # 2. Split
    train4_all, test_all = make_splits(all_states_df, train_years=4)
    print(f"[Split] 4y train weeks = {len(train4_all)}, test weeks = {len(test_all)}")

    # 3. Run Experiment (Parallel)
    obs_results = experiment_observation_levels_parallel(
        train4_all, test_all, THRESHOLDS
    )

    if not obs_results:
        print("[Error] No results generated.")
        return

    # Save Data
    df_obs = pd.DataFrame(obs_results)
    csv_obs_path = os.path.join(BASE_DIR, "ObsLevels_ARIMA111_all_states_boxplot_data_4y.csv")
    df_obs.to_csv(csv_obs_path, index=False)
    print(f"[Saved] {csv_obs_path}")

    # 4. Plot
    obs_fig_path = os.path.join(BASE_DIR, "ObsLevels_ARIMA111_all_states_MAE_4y.png")
    plot_observation_level_impact_all_states(obs_results, save_path=obs_fig_path)

    print("\n" + "=" * 80)
    print(f"Done. Processed {len(df_obs)} total forecasting scenarios.")


if __name__ == "__main__":
    np.random.seed(42)
    main()