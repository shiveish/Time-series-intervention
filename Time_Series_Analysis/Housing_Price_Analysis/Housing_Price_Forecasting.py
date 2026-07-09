import pandas as pd

# Load dataset
df = pd.read_csv("data_zillow_house_prices.csv")

# Q1
print("Number of rows:", df.shape[0])

# Q2
missing = df.isnull().sum().sum()
print("Total missing values:", missing)

if missing > 0:
    print("Yes, there are quite a few missing data.")
else:
    print("No missing data.")

# Q3
print("Number of region types:", df["RegionType"].nunique())
print(df["RegionType"].value_counts())

# Q4
print("\nFirst row time series:")
print(df.iloc[0, 8:])

# Convert DATE column to datetime
df["DATE"] = pd.to_datetime(df["DATE"])

# Q1
print("First year:", df["DATE"].dt.year.iloc[0])

# Q2
missing = df.isnull().sum().sum()
print("Missing values:", missing)

if missing == 0:
    print("No, all the data are given.")
else:
    print("Dataset contains missing values.")

# Q3
print("Minimum mortgage rate:", df["MORTGAGE30US"].min())
print("Maximum mortgage rate:", df["MORTGAGE30US"].max())

# Q4
print("\nSampling interval:")
print(df["DATE"].diff().value_counts().head())

###

boston = df[df["RegionName"] == "Boston"].iloc[0]
prices = boston.iloc[8:]
prices.index = pd.to_datetime(prices.index)
prices = prices.astype(float)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(prices)
plt.title("Boston House Prices")
plt.xlabel("Year")
plt.ylabel("Price")
plt.grid(True)
plt.show()

#Data Stationary Analysis

prices_2010 = prices.loc["2010-01-31":"2017-12-31"]
from statsmodels.tsa.stattools import adfuller

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

#-----------------------------------------#
from statsmodels.tsa.arima.model import ARIMA

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

#-----------------------------------------#
#1. Convert weekly interest rates to monthly averages
interest = interest_df.copy()

interest["DATE"] = pd.to_datetime(interest["DATE"])
interest.set_index("DATE", inplace=True)

monthly_interest = interest.resample("M").mean()
#2. Match the housing dates
monthly_interest = monthly_interest.loc["2010-01-31":"2019-12-31"]
#3. Split into train/test
exog_train = monthly_interest.loc["2010-01-31":"2017-12-31"]

exog_test = monthly_interest.loc["2018-01-31":"2019-12-31"]

#4. ARX Model Calibration
train_mse = {}

for p in range(1,5):

    model = ARIMA(
        endog=train_detrended,
        exog=exog_train,
        order=(p,0,0),
        trend="n"
    )

    fit = model.fit()

    pred = fit.predict(
        start=0,
        end=len(train_detrended)-1,
        exog=exog_train
    )

    mse = np.mean((train_detrended-pred)**2)

    train_mse[p]=mse

print(train_mse)
#Out of sample MSE
test_mse={}

for p in range(1,5):

    model = ARIMA(
        endog=train_detrended,
        exog=exog_train,
        order=(p,0,0),
        trend="n"
    )

    fit=model.fit()

    pred=[]

    rolling=train_detrended.copy()
    rolling_exog=exog_train.copy()

    for i in range(len(test_detrended)):

        fit=ARIMA(
            rolling,
            exog=rolling_exog,
            order=(p,0,0),
            trend="n"
        ).fit()

        forecast=fit.forecast(
            steps=1,
            exog=exog_test.iloc[[i]]
        )[0]

        pred.append(forecast)

        rolling=np.append(rolling,test_detrended[i])
        rolling_exog=pd.concat([rolling_exog,exog_test.iloc[[i]]])

    test_mse[p]=np.mean((np.array(pred)-test_detrended)**2)

print(test_mse)
#5. ARMAX Calibration
results={}

for p in range(1,5):

    for q in [1,5,10]:

        try:

            model=ARIMA(
                endog=train_detrended,
                exog=exog_train,
                order=(p,0,q),
                trend="n"
            )

            fit=model.fit()

            pred=fit.predict(
                start=0,
                end=len(train_detrended)-1,
                exog=exog_train
            )

            mse=np.mean((train_detrended-pred)**2)

            results[(p,q)]=mse

        except:

            pass

print(results)
#Find best training model
best=min(results,key=results.get)

print(best)
#Out-of-sample search
results_test={}

for p in range(1,5):

    for q in [1,5,10]:

        try:

            preds=[]

            rolling=train_detrended.copy()
            rolling_exog=exog_train.copy()

            for i in range(len(test_detrended)):

                fit=ARIMA(
                    rolling,
                    exog=rolling_exog,
                    order=(p,0,q),
                    trend="n"
                ).fit()

                pred=fit.forecast(
                    steps=1,
                    exog=exog_test.iloc[[i]]
                )[0]

                preds.append(pred)

                rolling=np.append(rolling,test_detrended[i])
                rolling_exog=pd.concat([rolling_exog,exog_test.iloc[[i]]])

            mse=np.mean((np.array(preds)-test_detrended)**2)

            results_test[(p,q)]=mse

        except:

            pass

print(results_test)
#Best-test model
best=min(results_test,key=results_test.get)

print(best)
