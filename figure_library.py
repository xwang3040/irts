import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, RationalQuadratic, ExpSineSquared, WhiteKernel,
    ConstantKernel as C, DotProduct
)
import numpy as np
import matplotlib.pyplot as plt
from google.genai.errors import APIError
from google import genai
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Shared color mapping for all forecast plots
_FORECAST_COLORS = {
    "train_true": "C0",
    "test_true":  "C1",
    "train_mean": "C2",
    "test_mean":  "C3",
    "train_ci":   "C2",  # CI uses same color as mean
    "test_ci":    "C3",
}


def gemini_forecast_plot_ensemble(
    train,
    test,
    preds,
    time_points,
    title="Gemini Forecast (train, test truth, and ensemble prediction)",
    xlabel="Time",
    ylabel="Value",
    ax=None,
    legend_loc="best",
    ci=95.0,
):
    """
    Plot ensemble forecast results with a 95% CI from multiple prediction runs.

    Args:
        train        : 1D array-like of training values.
        test         : 1D array-like of test values (truth).
        preds        : 2D array-like of shape (n_samples, len(test))
                       Ensemble predictions for the test horizon. May contain NaNs.
        time_points  : 1D array-like of length len(train) + len(test)
        title/xlabel/ylabel : plot labels.
        ax           : optional matplotlib Axes.
        legend_loc   : legend location (str).
        ci           : confidence level (e.g., 95.0).

    Returns:
        (fig, ax)
    """
    # Coerce inputs
    train = np.asarray(train, dtype=float).ravel()
    test = np.asarray(test, dtype=float).ravel()
    preds = np.asarray(preds, dtype=float)

    n_train = train.shape[0]
    n_test = test.shape[0]

    if preds.ndim != 2:
        raise ValueError(f"`preds` must be 2D (n_samples, n_test), got shape {preds.shape}.")
    if preds.shape[1] != n_test:
        raise ValueError(
            f"`preds` second dimension ({preds.shape[1]}) must equal len(test) ({n_test})."
        )

    # Coerce time_points to 1D numpy array, accept pandas Index/Series if present
    try:
        import pandas as pd  # optional
        if isinstance(time_points, (pd.Series, pd.Index)):
            x = np.asarray(time_points.to_numpy()).ravel()
        else:
            x = np.asarray(time_points).ravel()
    except Exception:
        x = np.asarray(time_points).ravel()

    if x.shape[0] != n_train + n_test:
        raise ValueError(
            f"`time_points` must have length {n_train + n_test} (got {x.shape[0]})."
        )

    x_train = x[:n_train]
    x_test = x[n_train:]

    # Compute ensemble mean and CI per time step, ignoring NaNs
    mean_pred = np.nanmean(preds, axis=0)

    alpha = (100.0 - ci) / 2.0
    lower = np.nanpercentile(preds, alpha, axis=0)
    upper = np.nanpercentile(preds, 100.0 - alpha, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # === Use the same colors as GP plot ===
    # True series
    ax.plot(
        x_train,
        train,
        label="Train (true)",
        color=_FORECAST_COLORS["train_true"],
    )
    ax.plot(
        x_test,
        test,
        label="Test (true)",
        color=_FORECAST_COLORS["test_true"],
    )

    # Ensemble mean prediction (use same color as "test_mean")
    ax.plot(
        x_test,
        mean_pred,
        "--",
        label="Test (ensemble mean prediction)",
        color=_FORECAST_COLORS["test_mean"],
    )

    # CI band (use same color as "test_ci")
    ax.fill_between(
        x_test,
        lower,
        upper,
        alpha=0.2,
        label=f"{ci:.0f}% CI (ensemble)",
        color=_FORECAST_COLORS["test_ci"],
    )

    # Vertical separator at train/test boundary
    if n_train > 0:
        ax.axvline(x_train[-1], linestyle=":", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig, ax


def gp_forecast_plot(
    res,
    time_points,
    ci=1.96,
    title="GP Forecast with Trend + Period (train t∈[0,1], test from 1)",
    xlabel="Time",
    ylabel="Value",
    show_train_ci=True,
    show_test_ci=True,
    ax=None,
    legend_loc="best"
):
    """
    Plot GP forecast using the result dict produced by `gp_forecast_fit`,
    with an explicit time axis for BOTH train and test.

    Args:
        res         : dict returned by gp_forecast_fit (may contain NaNs in truth)
        time_points : 1D array-like of length n_train + n_test (e.g., pandas.DatetimeIndex,
                      pandas Series, or numpy array). This is used on the x-axis.
        ci          : z-multiplier for CI shading (1.96 ≈ 95% CI)
        title/xlabel/ylabel : plot labels
        show_train_ci, show_test_ci : toggle CI shading
        ax          : optional existing matplotlib Axes to draw on
        legend_loc  : legend location string

    Returns:
        (fig, ax)
    """
    # Unpack from res
    train        = res["train"]
    test         = res["test"]
    y_pred_train = res["y_pred_train"]
    y_pred_test  = res["y_pred_test"]
    y_std_train  = res["y_std_train"]
    y_std_test   = res["y_std_test"]
    n_train      = res["n_train"]
    n_test       = res["n_test"]

    # Validate time axis length
    T = n_train + n_test
    try:
        import pandas as pd  # optional; only used if available
        if isinstance(time_points, (pd.Series, pd.Index)):
            x = np.asarray(time_points.to_numpy()).ravel()
        else:
            x = np.asarray(time_points).ravel()
    except Exception:
        x = np.asarray(time_points).ravel()

    if x.shape[0] != T:
        raise ValueError(f"`time_points` must have length {T} (got {x.shape[0]}).")

    x_train = x[:n_train]
    x_test  = x[n_train:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # === Use the same colors as ensemble plot ===
    # True series
    ax.plot(
        x_train,
        train,
        label="Train (true)",
        color=_FORECAST_COLORS["train_true"],
    )
    ax.plot(
        x_test,
        test,
        label="Test (true)",
        color=_FORECAST_COLORS["test_true"],
    )

    # GP means
    ax.plot(
        x_train,
        y_pred_train,
        "--",
        label="Train (GP mean)",
        color=_FORECAST_COLORS["train_mean"],
    )
    ax.plot(
        x_test,
        y_pred_test,
        "--",
        label="Test (GP mean)",
        color=_FORECAST_COLORS["test_mean"],
    )

    # CIs
    if show_train_ci:
        ax.fill_between(
            x_train,
            y_pred_train - ci * y_std_train,
            y_pred_train + ci * y_std_train,
            alpha=0.15,
            label="Train 95% CI",
            color=_FORECAST_COLORS["train_ci"],
        )
    if show_test_ci:
        ax.fill_between(
            x_test,
            y_pred_test - ci * y_std_test,
            y_pred_test + ci * y_std_test,
            alpha=0.15,
            label="Test 95% CI",
            color=_FORECAST_COLORS["test_ci"],
        )

    # Separator at the boundary (draw at the last train time)
    ax.axvline(x_train[-1], linestyle=":", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig, ax

def gp_forecast_fit(
    train, test, step=1.0, kernel=None, normalize=True, random_state=0,
    period_steps=None, use_trend=True, n_restarts=5, alpha=0
):
    """
    Fit a Gaussian Process to a univariate, evenly-spaced time series and forecast.

    Changes vs your original:
      • Train inputs are scaled to t ∈ [0,1]; test starts at t=1 and increases with the same spacing.
      • y is normalized using the TRAIN mean/std; predictions are converted back to original scale.
      • Kernel can include trend and periodic components. Provide `period_steps` (e.g., 52 for weekly annual).

    Args:
        train, test : 1D arrays
        step        : (ignored for x-scaling, kept for API compatibility)
        kernel      : sklearn kernel. If None, builds (Trend + Smooth + Periodic + MultiScale) + Noise
        normalize   : standardize y using train mean/std before fitting
        random_state: RNG seed
        period_steps: seasonal period in *steps* (e.g., 52). If None, no periodic term is added.
        use_trend   : include linear trend component
        n_restarts  : optimizer restarts
        alpha       : added nugget to K for numerical stability

    Returns:
        dict with predictions/uncertainties on original scale, metrics, plotting arrays, and fitted kernel/GP.
    """
    train = np.asarray(train, float).ravel()
    test  = np.asarray(test,  float).ravel()

    n_train = len(train)
    n_test  = len(test)
    if n_train < 2:
        raise ValueError("Need at least 2 training points for [0,1] scaling.")

    # ------- Build normalized time (evenly spaced) -------
    h = 1.0 / (n_train - 1)
    t_train = (np.arange(n_train) * h).reshape(-1, 1)              # 0, h, ..., 1
    t_test  = (1.0 + np.arange(1, n_test+1) * h).reshape(-1, 1)    # 1+h, 1+2h, ...

    # ------- Observed mask for training targets -------
    obs_mask_tr = np.isfinite(train)
    num_obs = int(obs_mask_tr.sum())
    if num_obs < 2:
        raise ValueError("Need at least 2 observed training values to fit the GP.")

    # Period in normalized time
    periodicity_norm = None
    if period_steps is not None and period_steps > 0:
        periodicity_norm = period_steps * h

    # ------- Normalize y on TRAIN-OBS only -------
    if normalize:
        mu_obs = float(np.nanmean(train))  # mean over observed only
        sd_obs = float(np.nanstd(train))
        if not np.isfinite(sd_obs) or sd_obs == 0:
            sd_obs = 1.0
        y_train_obs = (train[obs_mask_tr] - mu_obs) / sd_obs
        mu, sd = mu_obs, sd_obs
    else:
        mu, sd = 0.0, 1.0
        y_train_obs = train[obs_mask_tr]

    # ------- Kernel (unchanged) -------
    if kernel is None:
        pieces = []
        if use_trend:
            pieces.append(DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)))

        pieces.append(RationalQuadratic(alpha=1.0, length_scale=0.2,
                                        alpha_bounds=(1e-3, 1e3),
                                        length_scale_bounds=(1e-2, 5.0)))

        if periodicity_norm is not None:
            pieces.append(ExpSineSquared(
                length_scale=0.3, length_scale_bounds=(1e-2, 5.0),
                periodicity=periodicity_norm,
                periodicity_bounds=(0.8*periodicity_norm, 1.2*periodicity_norm)
            ))

        base = pieces[0]
        for k in pieces[1:]:
            base = base + k

        kernel = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 10.0))

    # ------- Fit on observed training points only -------
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                  n_restarts_optimizer=n_restarts,
                                  normalize_y=False, random_state=random_state)
    gp.fit(t_train[obs_mask_tr], y_train_obs)

    # ------- Predict on ALL train/test time points -------
    y_mean_tr, y_std_tr = gp.predict(t_train, return_std=True)
    y_mean_te, y_std_te = gp.predict(t_test,  return_std=True)

    # ------- Inverse transform to original scale -------
    y_pred_train = y_mean_tr * sd + mu
    y_pred_test  = y_mean_te * sd + mu
    y_std_train  = y_std_tr * sd
    y_std_test   = y_std_te * sd

    return {
        # Truth (may contain NaNs, preserved)
        "train": train,
        "test": test,
        # Predictions & uncertainty (original scale) for ALL time points
        "y_pred_train": y_pred_train,
        "y_std_train": y_std_train,
        "y_pred_test": y_pred_test,
        "y_std_test": y_std_test,
        "n_train": n_train,
        "n_test": n_test,
        # Model artifacts
        "gp": gp,
        "kernel": gp.kernel_,
        "periodicity_norm": periodicity_norm,
        "normalize_used": normalize,
        "train_mean": mu,
        "train_std": sd,
        "h": h,                       # normalized spacing
        "t_train_norm": t_train,      # normalized inputs (optional for debugging)
        "t_test_norm": t_test
    }

def arima_forecast_plot(
    train,
    test,
    time_points,
    title="ARIMA Forecast (train, test truth, and ARIMA prediction)",
    xlabel="Time",
    ylabel="Value",
    ax=None,
    legend_loc="best",
):
    """
    Plot ARIMA forecast using train/test plus predictions and 95% CI
    read from an Excel file (path is set *inside* this function).

    Excel file format (first sheet):
        col 0: forecast mean
        col 1: lower 95% CI
        col 2: upper 95% CI
    """
    # ---- hardcoded Excel path: change this to your actual file name/path ----
    excel_path = "arima_results.xlsx"
    # -------------------------------------------------------------------------    

    # Coerce inputs
    train = np.asarray(train, dtype=float).ravel()
    test  = np.asarray(test, dtype=float).ravel()

    n_train = train.shape[0]
    n_test  = test.shape[0]

    # --- Read ARIMA predictions and CI from Excel ---
    df = pd.read_excel(excel_path, header=None)

    if df.shape[1] < 3:
        raise ValueError(
            f"Excel file must have at least 3 columns (mean, lower, upper). Got {df.shape[1]}."
        )

    mean_pred = np.asarray(df.iloc[:, 0], dtype=float).ravel()
    lower     = np.asarray(df.iloc[:, 1], dtype=float).ravel()
    upper     = np.asarray(df.iloc[:, 2], dtype=float).ravel()

    if mean_pred.shape[0] != n_test:
        raise ValueError(
            f"Excel predictions length ({mean_pred.shape[0]}) must equal len(test) ({n_test})."
        )

    # --- Time axis handling (same as other plots) ---
    try:
        if isinstance(time_points, (pd.Series, pd.Index)):
            x = np.asarray(time_points.to_numpy()).ravel()
        else:
            x = np.asarray(time_points).ravel()
    except Exception:
        x = np.asarray(time_points).ravel()

    if x.shape[0] != n_train + n_test:
        raise ValueError(
            f"`time_points` must have length {n_train + n_test} (got {x.shape[0]})."
        )

    x_train = x[:n_train]
    x_test  = x[n_train:]

    # --- Set up figure/axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # === Same colors as GP / Gemini plots ===
    # True series
    ax.plot(
        x_train,
        train,
        label="Train (true)",
        color=_FORECAST_COLORS["train_true"],
    )
    ax.plot(
        x_test,
        test,
        label="Test (true)",
        color=_FORECAST_COLORS["test_true"],
    )

    # ARIMA mean prediction (use same color as "test_mean")
    ax.plot(
        x_test,
        mean_pred,
        "--",
        label="Test (ARIMA mean prediction)",
        color=_FORECAST_COLORS["test_mean"],
    )

    # 95% CI band (use same color as "test_ci")
    ax.fill_between(
        x_test,
        lower,
        upper,
        alpha=0.2,
        label="95% CI (ARIMA)",
        color=_FORECAST_COLORS["test_ci"],
    )

    # Vertical separator at train/test boundary
    if n_train > 0:
        ax.axvline(x_train[-1], linestyle=":", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig, ax

import gzip
import pickle
import numpy as np


def load_mae_matrix(path, horizon=52):
    # ---- load the object ----
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)   # 4x54 matrix-like of res dicts

    res_mat = np.asarray(obj, dtype=object)
    n_rows, n_cols = res_mat.shape

    mae_mat = np.full((n_rows, n_cols), np.nan, dtype=float)

    def starts_with_digit(raw):
        # Your reference logic:
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, (list, tuple)) and len(raw) > 0:
            # if raw_texts is a list, take the first element
            text = str(raw[0])
        else:
            text = ""

        # first non-whitespace character
        first_char = ""
        for ch in text.lstrip():
            first_char = ch
            break

        return first_char.isdigit()

    def to_len_horizon(arr, horizon):
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size >= horizon:
            return arr[:horizon]
        pad_len = horizon - arr.size
        return np.pad(arr, (0, pad_len), constant_values=np.nan)

    for i in range(n_rows):
        for j in range(n_cols):
            res = res_mat[i, j]

            raw_text = res.get("raw_texts", None)
            if not starts_with_digit(raw_text):
                # leave mae_mat[i, j] as NaN
                continue

            preds_mean = res["preds_mean"]
            test = res["test"]

            preds_52 = to_len_horizon(preds_mean, horizon)
            test_52  = to_len_horizon(test, horizon)  # test should be 52 but this is safe

            diff = np.abs(preds_52 - test_52)
            mae = np.nanmean(diff)  # ignore NaNs from padding/original
            mae_mat[i, j] = mae

    return mae_mat

def load_gp_mae_matrix(path, rows=(0, 2, 4, 6)):
    """
    Load results_5_missing.pkl.gz (9x54 matrix of res dicts from GP),
    keep only the specified rows (default: 0, 2, 4, 6) and compute MAE
    for each element using y_pred_test and test.

    For each res:
        mae = mean(|y_pred_test - test|)  (NaN-safe)

    Returns
    -------
    mae_mat : np.ndarray, shape (len(rows), 54)
    """
    # Load the object
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)   # expected 9x54 matrix-like of res dicts

    res_mat = np.asarray(obj, dtype=object)

    # Select only the desired rows (0, 2, 4, 6)
    selected = res_mat[list(rows), :]   # shape = (4, 54) if rows=(0,2,4,6)
    n_rows, n_cols = selected.shape

    mae_mat = np.full((n_rows, n_cols), np.nan, dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            res = selected[i, j]

            y_pred = np.asarray(res["y_pred_test"], dtype=float).ravel()
            test   = np.asarray(res["test"],        dtype=float).ravel()

            # Just in case lengths differ slightly, align to min length
            m = min(y_pred.size, test.size)
            if m == 0:
                mae = np.nan
            else:
                diff = np.abs(y_pred[:m] - test[:m])
                mae = np.nanmean(diff)

            mae_mat[i, j] = mae

    return mae_mat


def load_arima_mae_matrix(csv_path, target_missing=(0.0, 0.2, 0.4, 0.6)):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Keep only odd-numbered data rows (pandas indices 1,3,5,...)
    df = df.iloc[1::2].reset_index(drop=True)

    # Column names
    state_col = df.columns[0]
    missing_col = df.columns[1]
    mae_col = df.columns[2]

    # Filter missing_rate values
    df = df[df[missing_col].isin(target_missing)].copy()

    # Determine state order from dataset (first occurrence)
    states = df[state_col].drop_duplicates().tolist()
    n_states = len(states)

    # Prepare output: rows = missing_rate levels, cols = states
    n_missing = len(target_missing)
    mae_mat = np.full((n_missing, n_states), np.nan, dtype=float)

    # Fill matrix
    for j, state in enumerate(states):
        sub = df[df[state_col] == state].copy()

        # index by missing_rate for easy lookup
        sub = sub.set_index(missing_col)

        for i, mr in enumerate(target_missing):
            if mr in sub.index:
                mae_mat[i, j] = float(sub.loc[mr, mae_col])

    return mae_mat

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_missing_rate_boxplots(
    matrices,
    method_names,
    missing_labels,
    figsize=(12, 4),
    ylabel="MAE",
    y_max=None,   # upper y-axis limit
):
    """
    Plot boxplots of MAE for multiple methods across missing rates.

    Assumes:
      matrices = [arima_missing, gp_missing, p1_missing, p2_missing, p3_missing]
      method_names = ["ARIMA", "GP", "L1", "L2", "L3"]
    Each matrix has shape (n_missing_rates, n_states).
    """

    n_methods = len(matrices)
    if n_methods != len(method_names):
        raise ValueError("`matrices` and `method_names` must have same length.")

    n_missing_rates, n_states = matrices[0].shape
    if len(missing_labels) != n_missing_rates:
        raise ValueError("`missing_labels` length must match # of rows in matrices.")

    for m in matrices:
        if m.shape != (n_missing_rates, n_states):
            raise ValueError("All matrices must have shape (n_missing_rates, n_states).")

    # --- Colors ---
    # Soft, visually pleasing box colors
    # ARIMA, GP: warm-ish & green-ish
    # L1, L2, L3: blue gradient from light → medium → dark
    box_face_colors = [
        "#fdbf6f",  # ARIMA (soft orange)
        "#b2df8a",  # GP    (soft green)
        "#c6dbef",  # L1    (light blue)
        "#6baed6",  # L2    (medium blue)
        "#2171b5",  # L3    (dark blue)
    ]

    # Box edge colors: slightly darker versions, last one almost black
    box_edge_colors = [
        "#e08c2c",  # ARIMA edge
        "#7fbf3a",  # GP edge
        "#4a78a4",  # L1 edge
        "#2c7fb8",  # L2 edge
        "#08306b",  # L3 edge
    ]

    # Whiskers & caps in a neutral gray
    whisker_color = "#555555"
    cap_color = "#555555"

    # --- Positions & widths ---
    # ARIMA, GP at 1.0 and 2.0
    # L1, L2, L3 at 3.0, 3.5, 4.0, width 0.3 with gaps
    base_positions = [1.0, 2.0, 3.0, 3.5, 4.0]
    base_widths    = [0.6, 0.6, 0.3, 0.3, 0.3]

    fig, axes = plt.subplots(
        1,
        n_missing_rates,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for i in range(n_missing_rates):
        ax = axes[i]

        # Collect data for this missing rate across methods, dropping NaNs
        data_for_rate = []
        for mat in matrices:
            row = mat[i, :]
            clean = row[~np.isnan(row)]
            data_for_rate.append(clean)

        # Boxplot with custom positions and widths
        bplot = ax.boxplot(
            data_for_rate,
            positions=base_positions,
            widths=base_widths,
            patch_artist=True,
        )

        # Color the boxes
        for patch, fc, ec in zip(bplot["boxes"], box_face_colors, box_edge_colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.5)
            patch.set_alpha(0.9)

        median_color = "#000000"   # black (can change to "#333333" if you prefer softer)
        for median in bplot["medians"]:
            median.set_color(median_color)
            median.set_linewidth(2.0)

        # Whiskers and caps
        for whisker in bplot["whiskers"]:
            whisker.set_color(whisker_color)
            whisker.set_linewidth(1.2)

        for cap in bplot["caps"]:
            cap.set_color(cap_color)
            cap.set_linewidth(1.2)

        # Optional: fliers (outliers) muted
        for flier in bplot.get("fliers", []):
            flier.set_marker("o")
            flier.set_markerfacecolor("#999999")
            flier.set_markeredgecolor="#666666"
            flier.set_alpha(0.5)

        # Labels & axes
        ax.set_title(f"Missing rate: {missing_labels[i]}")
        if i == 0:
            ax.set_ylabel(ylabel)

        ax.set_xticks(base_positions)
        ax.set_xticklabels(method_names, rotation=45)

        if y_max is not None:
            ymin, _ = ax.get_ylim()
            ax.set_ylim(ymin, y_max)

    fig.tight_layout()
    return fig, axes


def load_arima_mae_observation(
    csv_path,
    thresholds=(0.0, 0.25, 0.33, 0.5, 0.67, 0.75),
):
    """
    Read ARIMA MAE observations from CSV and organize into a 6 x N matrix.

    CSV format (with header), columns:
        0: state
        1: threshold
        2: MAE
        (anything else is ignored)

    Output:
        rows    -> thresholds in the given order
        columns -> states in the order they appear in the CSV
        values  -> MAE

    Returns
    -------
    mae_mat : np.ndarray, shape (len(thresholds), N_states)
    """

    df = pd.read_csv(csv_path)

    # Column names
    state_col = df.columns[0]
    thr_col   = df.columns[1]
    mae_col   = df.columns[2]

    # Filter only the thresholds we care about
    df = df[df[thr_col].isin(thresholds)].copy()

    # Get states in the order they first appear
    states = df[state_col].drop_duplicates().tolist()
    n_states = len(states)
    n_thr = len(thresholds)

    # Prepare output matrix: thresholds x states
    mae_mat = np.full((n_thr, n_states), np.nan, dtype=float)

    # Index by threshold for easy lookup
    for j, state in enumerate(states):
        sub = df[df[state_col] == state].copy()
        sub = sub.set_index(thr_col)

        for i, thr in enumerate(thresholds):
            if thr in sub.index:
                mae_mat[i, j] = float(sub.loc[thr, mae_col])

    return mae_mat

import numpy as np
import matplotlib.pyplot as plt


def plot_observation_level_boxplots(
    matrices,
    method_names=("ARIMA", "GP"),
    obs_labels=("0", "1/4", "1/3", "1/2", "2/3", "3/4"),
    figsize=(14, 5),
    ylabel="MAE",
    y_max=None,
):
    """
    Boxplots for observation levels.

    Parameters
    ----------
    matrices : list of np.ndarray
        [observation_arima, observation_gp]
        Each matrix has shape (n_obs_levels, n_states).
    method_names : list/tuple of str
        Names of the two methods (default: ("ARIMA", "GP")).
    obs_labels : list/tuple of str
        Labels for observation levels in order of rows.
        Default: ("0", "1/4", "1/3", "1/2", "2/3", "3/4").
    figsize : tuple
        Figure size.
    ylabel : str
        Y-axis label.
    y_max : float or None
        If not None, set upper y-limit to this value.
    """
    if len(matrices) != 2:
        raise ValueError("`matrices` must contain exactly two matrices: [observation_arima, observation_gp].")

    n_obs_levels, n_states = matrices[0].shape
    if len(obs_labels) != n_obs_levels:
        raise ValueError("`obs_labels` length must match number of rows in matrices.")

    for m in matrices:
        if m.shape != (n_obs_levels, n_states):
            raise ValueError("All matrices must have shape (n_obs_levels, n_states).")

    # Colors: reuse the ARIMA / GP colors
    box_face_colors = [
        "#fdbf6f",  # ARIMA (soft orange)
        "#b2df8a",  # GP    (soft green)
    ]
    box_edge_colors = [
        "#e08c2c",  # ARIMA edge
        "#7fbf3a",  # GP edge
    ]
    median_color = "#000000"
    whisker_color = "#555555"
    cap_color = "#555555"

    # Positions and widths for two methods
    base_positions = [1.0, 2.0]
    base_widths = [0.6, 0.6]

    fig, axes = plt.subplots(
        1,
        n_obs_levels,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for i in range(n_obs_levels):
        ax = axes[i]

        # Collect data for this observation level, dropping NaNs
        data_for_level = []
        for mat in matrices:
            row = mat[i, :]
            clean = row[~np.isnan(row)]
            data_for_level.append(clean)

        bplot = ax.boxplot(
            data_for_level,
            positions=base_positions,
            widths=base_widths,
            patch_artist=True,
        )

        # Color boxes
        for patch, fc, ec in zip(bplot["boxes"], box_face_colors, box_edge_colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.5)
            patch.set_alpha(0.9)

        # Medians (same color for all)
        for median in bplot["medians"]:
            median.set_color(median_color)
            median.set_linewidth(2.0)

        # Whiskers & caps
        for whisker in bplot["whiskers"]:
            whisker.set_color(whisker_color)
            whisker.set_linewidth(1.2)

        for cap in bplot["caps"]:
            cap.set_color(cap_color)
            cap.set_linewidth(1.2)

        # Title & labels
        ax.set_title(f"Observation level: {obs_labels[i]}")
        if i == 0:
            ax.set_ylabel(ylabel)

        ax.set_xticks(base_positions)
        ax.set_xticklabels(method_names, rotation=45)

        if y_max is not None:
            ymin, _ = ax.get_ylim()
            ax.set_ylim(ymin, y_max)

    fig.tight_layout()
    return fig, axes

def _serialize_series(arr, precision=2):
    """
    Turn a numeric iterable into a comma-separated string.
    - precision: exact number of digits after the decimal point
    - NaN / missing -> empty token (i.e., consecutive commas)
    - Example: _serialize_series([0.1, np.nan, 2], precision=4) -> '0.1000,,2.0000'
    """
    a = np.asarray(arr, dtype=float)
    toks = []
    fmt = "{:." + str(precision) + "f}"
    for x in a:
        if np.isnan(x):
            toks.append("")
            continue
        s = fmt.format(x)
        # normalize negative zero like '-0.0000' -> '0.0000'
        if float(s) == 0.0 and s.startswith('-'):
            s = s[1:]
        toks.append(s)
    return ",".join(toks)

def _parse_numbers(text: str, expect_n: int) -> List[float]:
    """MOCK: Parses a comma-separated string into a list of floats,
    handling empty tokens (e.g., ",,").
    """
    try:
        parts = text.strip().split(',')
        if len(parts) != expect_n:
            print(f"Warning: Expected {expect_n} values, got {len(parts)} parts in: {text[:50]}...")
            # Attempt to fill with NaNs if necessary, or just process what we have
            # For this context, we will try to parse up to expect_n elements.
        
        results = []
        for part in parts[:expect_n]:
            try:
                # Replace empty string with NaN
                results.append(float(part.strip()) if part.strip() else np.nan)
            except ValueError:
                # Handle non-numeric text if any slips through (shouldn't with good prompt)
                results.append(np.nan)

        # Pad with NaNs if fewer than expected elements were parsed
        while len(results) < expect_n:
             results.append(np.nan)
             
        return results
        
    except Exception as e:
        print(f"Error parsing text '{text[:50]}...': {e}")
        return [np.nan] * expect_n
    
def gemini_forecast(
    train: List[float] | np.ndarray,
    test: List[float] | np.ndarray,
    model: str = "gemini-2.5-flash",
    *,
    n_samples: int = 1,
    precision: int = 2,
    max_tokens_per_step: int = 8,
    prompt_level: int = 3,
    api_key: str = None,
) -> Dict[str, Any]:
    """
    Forecast len(test) steps using the Google GenAI SDK (Gemini) with a SINGLE prompt string.

    The functionality, input/output, and prompt structure are preserved.
    """
    # 1. Initialize Gemini Client
    try:
        # The SDK automatically uses the GEMINI_API_KEY environment variable.
        client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        raise

    train = np.asarray(train, dtype=float)
    test  = np.asarray(test,  dtype=float)
    steps = int(test.size)
    
    # Early exit for empty test set (result structure preserved)
    if steps <= 0:
        return {
            "preds_mean": np.array([]),
            "rmse": np.nan,
            "all_preds": np.empty((0, 0)),
            "train": train,
            "test": test,
            "raw_texts": [],
            "config": {
                "model": model,
                "n_samples": int(n_samples),
                "precision": int(precision),
                "max_tokens_per_step": int(max_tokens_per_step),
                "prompt_level": int(prompt_level),
            },
            "meta": {"steps": steps, "prompt": ""},
        }

    seq = _serialize_series(train, precision=precision)

    # Clamp prompt_level to [1, 3]
    level = max(1, min(3, int(prompt_level)))

    # --- Build prompt in layers (identical to original) ---
    parts = []

    # Level 1: Task description
    parts.append(
        "TASK DESCRIPTION:\n"
        "You are forecasting future values for a univariate time series.\n"
        "The goal is to predict future values given the historical data.\n"
        "***IMPORTANT: If the historical input sequence contains two consecutive commas (e.g., '1.23,,1.25'), the value between them is missing.***\n"
    )

    # Level 2: Data context (ILI)
    if level >= 2:
        parts.append(
            "\nDATA CONTEXT:\n"
            "- The sequence represents *weekly Percentage of Visits for Influenza-Like Illness (ILI).* \n"
            "- Each value is the percent of all healthcare visits attributed to ILI in a given week.\n"
            "- The historical sequence you see includes data up to around September 2024.\n"
        )

    # Level 3: Modeling guidance
    if level >= 3:
        parts.append(
            "\nMODELING GUIDANCE:\n"
            "- When forecasting, explicitly consider:\n"
            "  • Long-term trend (overall increase, decrease, or stability over multiple seasons).\n"
            "  • Seasonality (typical winter influenza peaks and off-season lows each year).\n"
            "  • Randomness and noise (week-to-week fluctuations and irregular spikes).\n"
            "- Use these components implicitly in your reasoning when extending the series.\n"
        )

    # Output rules (always included)
    max_output_tokens = steps * max_tokens_per_step
    parts.append(
        "\nOUTPUT RULES:\n"
        " - Output ONLY the next values as a comma-separated list.\n"
        " - No spaces, no text, no explanations.\n\n"
        f"INPUT SEQUENCE (comma-separated):\n{seq}\n\n"
        f"TASK: Predict the next {steps} weekly ILI percentage values.\n"
        f"OUTPUT FORMAT EXAMPLE (for 3 steps): 1.23,1.25,1.27\n"
        "Remember: Output ONLY the comma-separated numbers."
    )

    prompt = "".join(parts)

    samples = max(1, int(n_samples))
    raw_texts, preds_list = [], []

    # 2. Call Gemini API for each sample
    for _ in range(samples):
        raw = ""
        try:
            # Use generate_content for text generation.
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            raw = response.text.strip()
            
        except APIError as e:
            raw = f"API_ERROR: {e}"
            print(f"Gemini API Error: {e}")
        except Exception as e:
            raw = f"GENERIC_ERROR: {e}"
            print(f"Generic Error: {e}")
            
        raw_texts.append(raw)
        
        # 3. Parse and store predictions
        preds_list.append(np.asarray(_parse_numbers(raw, steps), dtype=float))

    # 4. Calculate final metrics
    all_preds = np.stack(preds_list, axis=0)                  # (n_samples, steps)
    preds_mean = np.nanmean(all_preds, axis=0) # Use nanmean to handle NaNs from parsing errors
    
    # Calculate RMSE, handling potential NaNs in the prediction mean
    if np.any(np.isnan(preds_mean)):
        valid_indices = ~np.isnan(preds_mean)
        if np.any(valid_indices):
            rmse = float(np.sqrt(np.mean((preds_mean[valid_indices] - test[valid_indices]) ** 2)))
        else:
            rmse = np.nan
    else:
        rmse = float(np.sqrt(np.mean((preds_mean - test) ** 2))) # RMSE vs true test

    # 5. Return the result dictionary
    return {
        "preds_mean": preds_mean,
        "rmse": rmse,
        "all_preds": all_preds,
        "train": train,
        "test": test,
        "raw_texts": raw_texts,
        "config": {
            "model": model,
            "n_samples": samples,
            "precision": int(precision),
            "max_tokens_per_step": int(max_tokens_per_step),
            "prompt_level": level,
        },
        "meta": {
            "steps": steps,
            "prompt": prompt,
        },
    }


def gemini_forecast_plot(
    res,
    time_points,
    title="Gemini Forecast (train, test truth, and prediction)",
    xlabel="Time",
    ylabel="Value",
    ax=None,
    legend_loc="best",
):
    """
    Plot Gemini forecast results (no train predictions or CIs).
    Expects `res` that contains:
        - res["train"]      : 1D array of train values
        - res["test"]       : 1D array of test values (truth)
        - res["preds_mean"] : 1D array of predictions for test horizon

    Args:
        res         : dict from gpt_forecast_llmtime (or similarly shaped)
        time_points : 1D array-like of length len(train) + len(test)
        title/xlabel/ylabel : plot labels
        ax          : optional matplotlib Axes
        legend_loc  : legend location

    Returns:
        (fig, ax)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    train = np.asarray(res["train"], dtype=float)
    test  = np.asarray(res["test"], dtype=float)
    y_pred_test = np.asarray(res["preds_mean"], dtype=float)

    n_train = train.shape[0]
    n_test  = test.shape[0]
    if y_pred_test.shape[0] != n_test:
        raise ValueError(
            f"preds_mean length ({y_pred_test.shape[0]}) must equal test length ({n_test})."
        )

    # Coerce time_points to 1D numpy array, accept pandas Index/Series if present
    try:
        import pandas as pd  # optional
        if isinstance(time_points, (pd.Series, pd.Index)):
            x = np.asarray(time_points.to_numpy()).ravel()
        else:
            x = np.asarray(time_points).ravel()
    except Exception:
        x = np.asarray(time_points).ravel()

    if x.shape[0] != n_train + n_test:
        raise ValueError(
            f"`time_points` must have length {n_train + n_test} (got {x.shape[0]})."
        )

    x_train = x[:n_train]
    x_test  = x[n_train:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # ----- colors: use global _FORECAST_COLORS if available -----
    default_colors = {
        "train_true": "C0",
        "test_true":  "C1",
        "train_mean": "C2",
        "test_mean":  "C3",
        "train_ci":   "C2",
        "test_ci":    "C3",
    }
    color_map = globals().get("_FORECAST_COLORS", default_colors)

    # Plot true series
    ax.plot(
        x_train,
        train,
        label="Train (true)",
        color=color_map["train_true"],
    )
    ax.plot(
        x_test,
        test,
        label="Test (true)",
        color=color_map["test_true"],
    )

    # Plot Gemini predictions on test horizon (use test_mean color, i.e. red C3)
    ax.plot(
        x_test,
        y_pred_test,
        "--",
        label="Test (Gemini prediction)",
        color=color_map["test_mean"],
    )

    # Vertical separator at train/test boundary
    if n_train > 0:
        ax.axvline(x_train[-1], linestyle=":", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig, ax


def arima_forecast(
    train,
    test,
    time_points,
    title="ARIMA Forecast (train, test truth, and ARIMA prediction)",
    xlabel="Time",
    ylabel="Value",
    ax=None,
    legend_loc="best",
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 52),
    maxiter=100,
):
    """
    Fit a seasonal ARIMA model on `train`, forecast over the length of `test`,
    and plot train, test, ARIMA mean prediction, and 95% CI.

    Args:
        train         : 1D array-like, training values.
        test          : 1D array-like, test values (truth for forecast horizon).
        time_points   : 1D array-like of length len(train) + len(test)
                        (e.g., pandas.DatetimeIndex, Series, or numpy array).
        title/xlabel/ylabel : plot labels.
        ax            : optional matplotlib Axes.
        legend_loc    : legend location (str).
        order         : ARIMA order, default (2, 1, 2).
        seasonal_order: seasonal order, default (1, 1, 1, 52).
        maxiter       : maximum iterations for SARIMAX fit.

    Returns:
        (fig, ax)
    """

    # --------- coerce inputs ----------
    train = np.asarray(train, dtype=float).ravel()
    test  = np.asarray(test,  dtype=float).ravel()

    n_train = train.shape[0]
    n_test  = test.shape[0]

    if np.isnan(train).any():
        raise ValueError("`train` contains NaNs. Please handle missing values before calling arima_forecast().")

    # --------- time axis handling ----------
    try:
        if isinstance(time_points, (pd.Series, pd.Index)):
            x = np.asarray(time_points.to_numpy()).ravel()
        else:
            x = np.asarray(time_points).ravel()
    except Exception:
        x = np.asarray(time_points).ravel()

    if x.shape[0] != n_train + n_test:
        raise ValueError(
            f"`time_points` must have length {n_train + n_test} (got {x.shape[0]})."
        )

    x_train = x[:n_train]
    x_test  = x[n_train:]

    # --------- fit ARIMA (SARIMAX) ----------
    y = train

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    try:
        res = model.fit(disp=False, maxiter=maxiter)
    except Exception as e:
        print(f"[WARN] ARIMA fit failed ({e}); falling back to (1,0,1)x(1,0,1,52).")
        model = SARIMAX(
            y,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=max(50, maxiter // 2))

    # in-sample fitted values (length = n_train) – optional if you don’t need them on the plot
    fitted = np.asarray(res.fittedvalues).reshape(-1)

    # out-of-sample forecast (length = n_test)
    fc_res    = res.get_forecast(steps=n_test)
    mean_pred = np.asarray(fc_res.predicted_mean).reshape(-1)

    # 95% CI (default alpha=0.05)
    ci    = np.asarray(fc_res.conf_int(alpha=0.05))
    lower = ci[:, 0]
    upper = ci[:, 1]

    # --------- set up figure ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # --------- use your global color map ----------
    color_map = globals().get("_FORECAST_COLORS", {
        "train_true": "C0",
        "test_true":  "C1",
        "train_mean": "C2",
        "test_mean":  "C3",
        "train_ci":   "C2",
        "test_ci":    "C3",
    })

    # --------- plotting ----------
    # true series
    ax.plot(
        x_train,
        train,
        label="Train (true)",
        color=color_map["train_true"],
    )
    ax.plot(
        x_test,
        test,
        label="Test (true)",
        color=color_map["test_true"],
    )

    # ARIMA mean prediction → uses test_mean color (C3 = red)
    ax.plot(
        x_test,
        mean_pred,
        "--",
        label="Test (ARIMA mean prediction)",
        color=color_map["test_mean"],
    )

    # 95% CI band → uses test_ci color (also C3)
    ax.fill_between(
        x_test,
        lower,
        upper,
        alpha=0.2,
        label="95% CI (ARIMA)",
        color=color_map["test_ci"],
    )

    # vertical separator at train/test boundary
    if n_train > 0:
        ax.axvline(x_train[-1], linestyle=":", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    fig.tight_layout()

    return fig, ax