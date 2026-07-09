"""
utils.py

Utility functions for the Housing Price Forecasting project.

Author: Shiveish Chetty
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


# =============================================================================
# Load Housing Dataset
# =============================================================================

def load_housing_data(filepath="data_zillow_house_prices.csv"):
    """
    Load the Zillow housing dataset.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(filepath)


# =============================================================================
# Load Interest Rate Dataset
# =============================================================================

def load_interest_data(filepath="data_interest_rates.csv"):
    """
    Load the mortgage interest rate dataset.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(filepath)


# =============================================================================
# Extract City Time Series
# =============================================================================

def get_city_series(df, city="Boston"):
    """
    Extract the monthly house-price series for a city.

    Parameters
    ----------
    df : DataFrame
        Zillow dataset.

    city : str
        City/Region name.

    Returns
    -------
    pandas.Series
    """

    row = df[df["RegionName"] == city].iloc[0]

    prices = row.iloc[8:]

    prices.index = pd.to_datetime(prices.index)

    prices = prices.astype(float)

    return prices


# =============================================================================
# Train/Test Split
# =============================================================================

def train_test_split_series(
        prices,
        train_start="2010-01-31",
        train_end="2017-12-31",
        test_start="2018-01-31",
        test_end="2019-12-31"
):
    """
    Split a time series into train and test.
    """

    train = prices.loc[train_start:train_end]

    test = prices.loc[test_start:test_end]

    return train, test


# =============================================================================
# Linear Detrending using OLS
# =============================================================================

def detrend_series(train, test):
    """
    Remove a linear trend using OLS.

    Returns
    -------
    train_detrended
    test_detrended
    ols_model
    """

    x_train = np.arange(len(train))

    X_train = sm.add_constant(x_train)

    ols = sm.OLS(train.values, X_train).fit()

    trend_train = ols.predict(X_train)

    train_detrended = train.values - trend_train

    x_test = np.arange(
        len(train),
        len(train) + len(test)
    )

    X_test = sm.add_constant(x_test)

    trend_test = ols.predict(X_test)

    test_detrended = test.values - trend_test

    return train_detrended, test_detrended, ols


# =============================================================================
# Prepare Monthly Interest Rates
# =============================================================================

def prepare_interest_rates(
        interest_df,
        start="2010-01-31",
        end="2019-12-31"
):
    """
    Convert weekly mortgage rates into monthly averages.
    """

    interest = interest_df.copy()

    interest["DATE"] = pd.to_datetime(
        interest["DATE"]
    )

    interest.set_index("DATE", inplace=True)

    monthly = interest.resample("ME").mean()

    monthly = monthly.loc[start:end]

    return monthly


# =============================================================================
# Split Exogenous Variables
# =============================================================================

def split_exogenous(
        monthly_interest,
        train_start="2010-01-31",
        train_end="2017-12-31",
        test_start="2018-01-31",
        test_end="2019-12-31"
):
    """
    Split monthly mortgage rates into train and test.
    """

    exog_train = monthly_interest.loc[
        train_start:train_end
    ]

    exog_test = monthly_interest.loc[
        test_start:test_end
    ]

    return exog_train, exog_test


# =============================================================================
# Mean Squared Error
# =============================================================================

def mse(actual, predicted):
    """
    Compute Mean Squared Error.
    """

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)

    return np.mean((actual - predicted) ** 2)