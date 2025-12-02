# 📘 Zero-Shot Flu Forecasting with Gemini vs. Classical Time-Series Models

## Project Summary

This project investigates the capability of **Large Language Models (LLMs)** to serve as practical forecasters for seasonal influenza under realistic public-health conditions. Specifically, we focus on challenges such as **irregular sampling, missing data, and early-season uncertainty** in time series.

Using **5 years of CDC Influenza-Like Illness (ILI) data** and a fully held-out **2024–2025 season**, we compare the zero-shot forecasting performance of **Gemini 2.5 Flash** against two classical time-series baselines: **Seasonal ARIMA** and **Gaussian Process (GP) regression**.

The LLM receives only a raw, **comma-separated text serialization** of the time series. Crucially, missing values are encoded strictly via `,,` with **no imputation, scaling, or normalization** applied. All models are evaluated using **Mean Absolute Error (MAE)**.

---

## 🚀 Get Started: Demonstration Notebook

A full, runnable demonstration of the core methodology, including data preparation, model predictions, and key result visualizations, is available in the **[`demo.ipynb`](./demo.ipynb)** Jupyter notebook. This is the fastest way to explore our methods and reproduce the findings.

---

## 🔬 Experiments

### 1. Experiment 1 — Impact of Missing Data

We simulated weaker surveillance systems by randomly removing **0%, 20%, 40%, and 60%** of the training data for each U.S. state to assess model robustness.

#### Key Results

* All methods, including Gemini, demonstrated surprising **robustness** even with 60% missing observations.
* **ARIMA & GP** achieved the lowest median MAE (approx. 1.0–1.3).
* **Gemini LLM** performance improved from Prompt Level 1 → 2 → 3.
* At **Prompt Level 3**, Gemini became competitive with GP in median MAE, showcasing impressive zero-shot performance given its minimal input and lack of training/fitting.

### 2. Experiment 2 — Practical Early-Season Forecasting

We mimicked real-world forecasting by revealing only a partial segment of the 2024–2025 season. States received six observation levels (0, ¼, ⅓, ½, ⅔, ¾ of the rise from trough to peak) to test predictive accuracy as the season begins.

#### Key Results

* Both **ARIMA and GP** produced reasonable predictions even at the earliest observation level, however, more information did not lead to better performance and sometimes worse it.
* **GP** remained consistently more **stable** with a tighter Interquartile Range (IQR) across thresholds.
* LLM results for this experiment are **not yet available** due to API rate limits.

---

## ⭐ Key Takeaways

1.  **Zero-shot LLMs can produce meaningful forecasts** for public-health time series, even with raw, irregular, and missing data.
2.  Classical models (**ARIMA, GP**) remain **strong, stable baselines** for this domain.
3.  **Prompt is crucial**—richer prompts with proper guidance improved the Gemini LLM's forecasting performance.
4.  LLMs offer a promising **low-resource alternative**, requiring no model fitting, tuning, or feature engineering, making them highly accessible.

---

## 🚧 Limitations

* Only **Gemini-2.5-Flash (free tier)** was used; more advanced LLMs were not tested.
* LLM forecasts for Experiment 2 remain **incomplete** due to request limits.
* Minimal preprocessing was used (**raw numbers only**); the impact of tokenization, scaling, and representation on LLM performance was not fully explored.
* Results reflect only **ILI%** and a single held-out season.

---

## 🔮 Future Work

* Evaluate additional, more powerful LLMs (Gemini Pro, GPT-5, open-source models).
* Complete early-season forecasting for LLMs (Experiment 2).
* Study how data representation (**tokenization, scaling**) affects LLM forecasting accuracy.
* Extend the methodology to:
    * Multiple indicators (hospitalizations, virology data).
    * Multi-horizon and probabilistic forecasting.
    * Operational public-health use cases.

---

## 👥 Project Members

| Name | Portrait | Contact |
| :--- | :--- | :--- |
| **Hanze Qin** | ![Portrait of Hanze Qin](./img/hanze.jpg) | hqin68@gatech.edu |
| **Xingjian Wang** | <img src="./img/XW_2024.jpg" width="180" alt="Portrait of Xingjian Wang"> | xwang3040@gatech.edu |

---

## 📁 Files

* **📘 Full Write-Up (PDF):** [`CSE8803_Final.pdf`](./CSE8803_Final.pdf)
* **🗂️ Software Package (tar.gz):** [`irts-main.zip`](./irts-main.zip)
