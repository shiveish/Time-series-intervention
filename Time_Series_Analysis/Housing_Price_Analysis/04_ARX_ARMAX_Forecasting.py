from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import utils

# Load and prepare datasets
housing_df = utils.load_housing_data()
interest_df = utils.load_interest_data()

prices = utils.get_city_series(housing_df)

train, test = utils.train_test_split_series(prices)

train_detrended, test_detrended, ols = utils.detrend_series(train, test)

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
