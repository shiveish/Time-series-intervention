import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import utils

# Load and prepare housing data
housing_df = utils.load_housing_data()

prices = utils.get_city_series(housing_df)

train, test = utils.train_test_split_series(prices)

train_detrended, test_detrended, ols = utils.detrend_series(train, test)


# Long-term
model = ARIMA(train_detrended, order=(2,0,0))
fit = model.fit()

forecast_long = fit.forecast(len(test))

print("May Long MSE:", ((forecast_long[4]-test_detrended[4])**2)/1000)
print("Overall Long:", np.mean((forecast_long-test_detrended)**2)/1000)

# Short-term
rolling = train_detrended.copy()

forecast_short = []

for y in test_detrended:

    fit = ARIMA(rolling, order=(2,0,0)).fit()

    forecast_short.append(fit.forecast()[0])

    rolling = np.append(rolling, y)

forecast_short = np.array(forecast_short)

print("May Short:", ((forecast_short[4]-test_detrended[4])**2)/1000)

print("Overall Short:",
      np.mean((forecast_short-test_detrended)**2)/1000)
