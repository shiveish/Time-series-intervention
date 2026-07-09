# 📈 Housing Price Time Series Forecasting

A comprehensive time-series forecasting project that predicts **Boston housing prices** using classical statistical models. The project investigates autoregressive forecasting, trend removal, and the effect of mortgage interest rates on housing prices through **AR**, **ARX**, and **ARMAX** models.

---

## Overview

Housing prices exhibit long-term trends and temporal dependencies, making them well suited for time-series forecasting. This project develops a complete forecasting pipeline beginning with data exploration and ending with model evaluation and robustness analysis.

The workflow includes:

- Data Exploration
- Stationarity Analysis
- Trend Removal
- Autocorrelation Analysis
- Autoregressive Forecasting
- ARX & ARMAX Modeling
- Model Evaluation
- Robustness Analysis

---

## Dataset

### Zillow Home Value Index (ZHVI)

- Region: **Boston, Massachusetts**
- Monthly Housing Prices
- Time Period: **April 2000 – August 2022**
- 893 geographic regions
- 269 monthly observations per region

### FRED Mortgage Interest Rate Dataset

- Weekly 30-Year Fixed Mortgage Rate
- Source: Federal Reserve Economic Data (FRED)
- Converted into monthly averages before model training.

---

# Repository Structure

```
Housing_Price_Time_Series_Forecasting/
│
├── README.md
├── requirements.txt
├── utils.py
│
├── 01_Data_Exploration.py
├── 02_Stationarity_Analysis.py
├── 03_AR_Forecasting.py
├── 04_ARX_ARMAX_Forecasting.py
├── 05_Robustness_Analysis.py
│
├── data_zillow_house_prices.csv
├── data_interest_rates.csv
│
└── figures/
```

---

# Project Workflow

## 1. Data Exploration

The datasets are first inspected to understand their structure and quality.

Performed analyses include:

- Dataset dimensions
- Missing value analysis
- Region statistics
- Housing price visualization
- Mortgage rate inspection

---

## 2. Stationarity Analysis

Time-series forecasting models require stationary data.

The following techniques were applied:

- Augmented Dickey-Fuller (ADF) Test
- Linear Trend Removal
- Quadratic Trend Removal
- Cubic Trend Removal
- Autocorrelation Function (ACF)
- Partial Autocorrelation Function (PACF)

These analyses were used to determine suitable autoregressive model orders.

---

## 3. AR Forecasting

An AutoRegressive model was developed after removing the linear trend using Ordinary Least Squares (OLS).

Two forecasting strategies were evaluated:

- Long-term forecasting
- Rolling (short-term) forecasting

Performance was measured using Mean Squared Error (MSE).

---

## 4. ARX & ARMAX Forecasting

Mortgage interest rates were incorporated as an exogenous variable to evaluate whether macroeconomic information improves prediction accuracy.

Models explored:

- ARX(p)
- ARMAX(p,q)

Multiple parameter combinations were evaluated and compared using both training and testing errors.

---

## 5. Robustness Analysis

A supplementary experiment was conducted to evaluate model stability under a different train-test split.

Original Split:

- Training: 2010–2017
- Testing: 2018–2019

Robustness Split:

- Training: 2010–2019
- Testing: 2020–2021

This analysis examines how the forecasting models perform during periods of significant market shifts, including the COVID-19 housing market.

---

# Models Used

- Ordinary Least Squares (OLS)
- AutoRegressive (AR)
- AutoRegressive with Exogenous Variables (ARX)
- AutoRegressive Moving Average with Exogenous Variables (ARMAX)

---

# Statistical Techniques

- Stationarity Testing
- Linear Detrending
- Polynomial Detrending
- Augmented Dickey-Fuller Test
- Autocorrelation Analysis
- Partial Autocorrelation Analysis
- Rolling Forecast Evaluation

---

# Evaluation Metrics

The models were evaluated using:

- Training Mean Squared Error (MSE)
- Testing Mean Squared Error (MSE)
- Long-Term Forecast Error
- Rolling Forecast Error

---

# Key Findings

- Housing prices exhibit strong non-stationary behavior.
- Linear detrending significantly improves stationarity.
- ACF and PACF analysis indicate that low-order autoregressive models adequately capture temporal dependence.
- Rolling forecasting consistently outperforms one-shot long-term forecasting.
- Mortgage interest rates provide additional predictive information but only modest improvements over the baseline autoregressive model.
- Forecast accuracy deteriorates under structural market changes (2020–2021), highlighting the importance of robustness evaluation.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels

---

# Future Improvements

Potential extensions include:

- Seasonal ARIMA (SARIMA)
- Prophet Forecasting
- Vector AutoRegression (VAR)
- LSTM-based Deep Learning Models
- Additional macroeconomic indicators such as:
  - Inflation
  - Consumer Price Index (CPI)
  - GDP
  - Unemployment Rate

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<username>/Housing_Price_Time_Series_Forecasting.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute the scripts in the following order:

```
01_Data_Exploration.py

↓

02_Stationarity_Analysis.py

↓

03_AR_Forecasting.py

↓

04_ARX_ARMAX_Forecasting.py

↓

05_Robustness_Analysis.py
```

---

# Author

**Shiveish Chetty**

B.Tech, Electrical Engineering  
Indian Institute of Technology Kanpur

---

# License

This project is intended for academic, educational, and research purposes.
