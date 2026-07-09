# =============================================================================
# OPTIONAL ANALYSIS
# Robustness Study: Effect of Changing the Training and Testing Data
# =============================================================================

print("=" * 80)
print("OPTIONAL ANALYSIS")
print("Performance Shift under Different Train/Test Splits")
print("=" * 80)

print("""
Objective:
----------
Evaluate the robustness of the AR(2) and ARX(2) forecasting models by extending
the training period from 2010-2017 to 2010-2019 and shifting the test period
to 2020-2021.

The preprocessing pipeline remains identical:
    • Linear trend removal using OLS
    • Short-term (rolling) forecasting
    • AR(2) and ARX(2) models
Only the train/test split is changed.
""")

# =============================================================================
# STEP 1 : NEW TRAIN / TEST SPLIT
# =============================================================================

train = prices.loc["2010-01-31":"2019-12-31"]
test = prices.loc["2020-01-31":"2021-12-31"]

print("\nTraining samples :", len(train))
print("Testing samples  :", len(test))

# =============================================================================
# STEP 2 : REMOVE LINEAR TREND (OLS)
# =============================================================================

x_train = np.arange(len(train))
X_train = sm.add_constant(x_train)

ols = sm.OLS(train.values, X_train).fit()

trend_train = ols.predict(X_train)
train_detrended = train.values - trend_train

x_test = np.arange(len(train), len(train) + len(test))
X_test = sm.add_constant(x_test)

trend_test = ols.predict(X_test)
test_detrended = test.values - trend_test

print("\nLinear trend removed successfully.")

# =============================================================================
# STEP 3 : PREPARE MONTHLY INTEREST RATES
# =============================================================================

monthly_interest = interest.copy()

monthly_interest.index = pd.to_datetime(monthly_interest.index)

monthly_interest = monthly_interest.resample("M").mean()

monthly_interest = monthly_interest.loc["2010-01-31":"2021-12-31"]

exog_train = monthly_interest.loc["2010-01-31":"2019-12-31"]
exog_test = monthly_interest.loc["2020-01-31":"2021-12-31"]

# =============================================================================
# STEP 4 : AR(2) MODEL
# =============================================================================

print("\n" + "=" * 80)
print("AR(2) MODEL")
print("=" * 80)

ar_model = ARIMA(
    train_detrended,
    order=(2, 0, 0),
    trend="n"
)

ar_fit = ar_model.fit()

train_prediction = ar_fit.predict(
    start=0,
    end=len(train_detrended) - 1
)

train_mse_ar = np.mean((train_detrended - train_prediction) ** 2)

rolling_train = train_detrended.copy()

predictions = []

for actual in test_detrended:

    fit = ARIMA(
        rolling_train,
        order=(2, 0, 0),
        trend="n"
    ).fit()

    forecast = fit.forecast()[0]

    predictions.append(forecast)

    rolling_train = np.append(
        rolling_train,
        actual
    )

predictions = np.array(predictions)

test_mse_ar = np.mean(
    (predictions - test_detrended) ** 2
)

print(f"Training MSE : {train_mse_ar:,.2f}")
print(f"Testing  MSE : {test_mse_ar:,.2f}")

# =============================================================================
# STEP 5 : ARX(2) MODEL
# =============================================================================

print("\n" + "=" * 80)
print("ARX(2) MODEL")
print("=" * 80)

arx_model = ARIMA(
    train_detrended,
    exog=exog_train,
    order=(2, 0, 0),
    trend="n"
)

arx_fit = arx_model.fit()

train_prediction = arx_fit.predict(
    start=0,
    end=len(train_detrended) - 1,
    exog=exog_train
)

train_mse_arx = np.mean(
    (train_detrended - train_prediction) ** 2
)

rolling_train = train_detrended.copy()
rolling_exog = exog_train.copy()

predictions = []

for i, actual in enumerate(test_detrended):

    fit = ARIMA(
        rolling_train,
        exog=rolling_exog,
        order=(2, 0, 0),
        trend="n"
    ).fit()

    forecast = fit.forecast(
        steps=1,
        exog=exog_test.iloc[[i]]
    )[0]

    predictions.append(forecast)

    rolling_train = np.append(
        rolling_train,
        actual
    )

    rolling_exog = pd.concat(
        [
            rolling_exog,
            exog_test.iloc[[i]]
        ]
    )

predictions = np.array(predictions)

test_mse_arx = np.mean(
    (predictions - test_detrended) ** 2
)

print(f"Training MSE : {train_mse_arx:,.2f}")
print(f"Testing  MSE : {test_mse_arx:,.2f}")

# =============================================================================
# STEP 6 : SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = pd.DataFrame({
    "Model": ["AR(2)", "ARX(2)"],
    "Training MSE": [
        train_mse_ar,
        train_mse_arx
    ],
    "Testing MSE": [
        test_mse_ar,
        test_mse_arx
    ]
})

print(summary)

print("\nObservations")
print("-" * 80)

print("""
• The training period was extended by two years (2018–2019).

• The testing period now consists of the market behaviour during
  2020–2021.

• This period includes significant structural changes in the housing
  market, making forecasting considerably more challenging.

• Comparing these MSE values with the original experiment allows us to
  evaluate how sensitive AR and ARX models are to changes in the
  training/testing distribution.

• This experiment serves as a robustness analysis rather than a core
  component of the forecasting pipeline.
""")