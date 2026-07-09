import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

housing_df = pd.read_csv("data_zillow_house_prices.csv")

boston = housing_df[housing_df["RegionName"]=="Boston"].iloc[0]

prices = boston.iloc[8:]
prices.index = pd.to_datetime(prices.index)
prices = prices.astype(float)
prices_2010 = prices.loc["2010-01-31":"2017-12-31"]

result = adfuller(prices_2010)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
print(round(result[1], 3))
import numpy as np

x = np.arange(len(prices_2010))

coef = np.polyfit(x, prices_2010, 1)
trend = np.polyval(coef, x)

linear_removed = prices_2010 - trend

print(adfuller(linear_removed)[1])
coef = np.polyfit(x, prices_2010, 2)
trend = np.polyval(coef, x)

quadratic_removed = prices_2010 - trend

print(adfuller(quadratic_removed)[1])
coef = np.polyfit(x, prices_2010, 3)
trend = np.polyval(coef, x)

cubic_removed = prices_2010 - trend

print(adfuller(cubic_removed)[1])
print("Original :", adfuller(prices_2010)[1])
print("Linear   :", adfuller(linear_removed)[1])
print("Quadratic:", adfuller(quadratic_removed)[1])
print("Cubic    :", adfuller(cubic_removed)[1])

##-----------------------------------------##
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# ACF Plot
plot_acf(linear_removed, lags=20)
plt.show()

# PACF Plot
plot_pacf(linear_removed, lags=20)
plt.show()

# Numerical ACF
acf_values = acf(linear_removed, nlags=20)

print("ACF Values")
for i, val in enumerate(acf_values):
    print(f"Lag {i}: {val:.4f}")

# Numerical PACF
pacf_values = pacf(linear_removed, nlags=20)

print("\nPACF Values")
for i, val in enumerate(pacf_values):
    print(f"Lag {i}: {val:.4f}")
