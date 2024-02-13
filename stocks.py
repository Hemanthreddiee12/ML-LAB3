
import numpy as np
import pandas as pd
file_path = "Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
df.head()


# Calculating mean and variance of Price
price_mean = df["Price"].mean()
price_variance = df["Price"].var()

print("Mean of Price:", price_mean)
print("Variance of Price:", price_variance)

# Select Wednesday prices and calculating sample mean
wednesday_mean=df[df['Day']=='Wed']['Price'].mean()

print("Sample mean of Wednesday prices:", wednesday_mean)
print("Comparison with population mean:", wednesday_mean - price_mean)

# Select April prices and calculating sample mean
april_prices = df[df['Month']=='Apr']["Price"]
april_mean = april_prices.mean()

print("Sample mean of April prices:", april_mean)
print("Comparison with population mean:", april_mean - price_mean)

# Probability of loss
loss_probability = len(df[df["Chg%"] < 0]) / len(df)
print("Probability of making a loss:", loss_probability)
