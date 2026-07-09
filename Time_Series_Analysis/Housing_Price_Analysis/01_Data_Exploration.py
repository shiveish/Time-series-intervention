import pandas as pd

# Load dataset
df = pd.read_csv("data_zillow_house_prices.csv")
df_interest = pd.read_csv("data_interest_rates.csv")

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
df_interest["DATE"] = pd.to_datetime(df_interest["DATE"])

# Q1
print("First year:", df_interest["DATE"].dt.year.iloc[0])

# Q2
missing = df_interest.isnull().sum().sum()
print("Missing values:", missing)

if missing == 0:
    print("No, all the data are given.")
else:
    print("Dataset contains missing values.")

# Q3
print("Minimum mortgage rate:", df_interest["MORTGAGE30US"].min())
print("Maximum mortgage rate:", df_interest["MORTGAGE30US"].max())

# Q4
print("\nSampling interval:")
print(df_interest["DATE"].diff().value_counts().head())

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
