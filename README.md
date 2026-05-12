#  Avocado Price Prediction & Type Classification
---

##  Project Overview

This project applies supervised machine learning and time series analysis to the Avocado dataset to answer two key business questions:

1. **Can we predict avocado prices?** — Using Linear Regression and Time Series (AR/ARIMA)
2. **Can we classify avocado type?** — Distinguishing Organic from Conventional using Logistic Regression and PCA


---

##  Dataset Description

| Column | Description |
|--------|-------------|
| `Date` | Date of observation |
| `AveragePrice` | Average price per avocado |
| `Total Volume` | Total number of avocados sold |
| `type` | Organic or Conventional |
| ` PLU codes(4046`, `4225`, `4770)` | Different avocado categories
| `Total Bags` | Total number of bags sold |
| `Small/Large/XLarge Bags` | Bag size volumes |
| `Region` | Sales region |
| `Year` | Year of observation |

---

##  Tools & Libraries

|Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `pyjanitor` | Data cleaning & reshaping (`pivot_longer`) |
| `numpy` | Numerical operations |
| `plotnine` | Grammar-of-graphics visualisations |
| `seaborn` / `matplotlib` | Heatmaps and additional plots |
| `scikit-learn` | Linear Regression, Logistic Regression, PCA, scaling |
| `statsmodels` | Time Series (AR/ARIMA) modelling |

---

## Project Workflow

### 1. Data Preparation
- Cleaned column names with `pyjanitor`
- Reshaped from wide to long format using `pivot_longer` (twice — for PLU codes and bag types)
- Converted `date` to datetime; extracted `month`, `year`, `day`
- Engineered a `season` feature (Winter, Spring, Summer, Fall)
- Confirmed: **no missing values**, **no duplicates**

---

### 2. Part 1 — Price Prediction

#### Option A: Linear Regression (4 Progressive Models)

| Model | Features | R² | RMSE |
|-------|----------|----|------|
| Model 1 | `total_volume` only | 3.7% | ~0.39 |
| Model 2 | `total_volume` + `season` | 9% | ~38% |
| Model 3 | + `type` | 47% | ~30% |
| Model 4 (Best) | All features incl. `region`, `plu_code`, `bag_type` | ~60% | **~25%** |

**Key insight:** Adding `type` (Organic vs Conventional) was the single biggest improvement. Region and seasonality also matter significantly.

#### Option B: Time Series Analysis (AR Models)

- Checked stationarity, ACF/PACF, and decomposition (trend + seasonality)
- Compared **AR(1)** vs **AR(2)**

| Model | AIC/BIC | p-value | RMSE |
|-------|---------|---------|------|
| AR(1) | Lower  | < 0.05  | **0.07** |
| AR(2) | Higher | > 0.05  | Higher |

**Winner: AR(1) — Time Series outperformed all Linear Regression models with RMSE of 0.07 vs 0.255**

---

### 3. Part 2 — Type Classification (Organic vs Conventional)

#### Option A: Logistic Regression 

- Features: `average_price`, `total_volume`, `season`, `region`
- Target: `type_encoded` (Organic = 1, Conventional = 0)
- Scaled features with `StandardScaler`, encoded categoricals with `get_dummies`

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | ~92% | **92%** |

> Test accuracy matches training accuracy → **no overfitting**, model generalises well.

#### Option B: PCA + Logistic Regression

- Applied PCA before logistic regression
- Optimal components: **3** (capturing 72.5% of variance, per scree plot "elbow")
- Top contributing features: avocado volume and average price

**Logistic Regression gave a higher precision and accuracy than the PCA approach**

---

## Key Findings

- **Avocado prices are driven by type, region, seasonality, and supply level** — not any single factor
- **Fall** consistently shows the highest and most variable average prices; **Winter** is cheapest
- **Organic avocados** are reliably more expensive, but in high-demand regions or peak seasons, conventional prices can overlap, making them hard to distinguish even for a model
- **Time series (AR1)** captured the temporal structure of price data far better than cross-sectional linear regression
- **Wide-to-long reshaping** revealed hidden relationships between PLU codes, bag types, and pricing

---

##  Model Summary

| Task | Model | Result |
|------|-------|--------|
| Price Prediction | Linear Regression (Model 4) | Best R², RMSE = 0.255 |
| Price Prediction | Time Series AR(1) | **RMSE = 0.07** |
| Type Classification | Logistic Regression | **92% accuracy**  |
| Type Classification | PCA + Logistic Regression | Lower accuracy |

---


*Built as part of the SheCodeAfrica Data Science Programme Mini Project*
