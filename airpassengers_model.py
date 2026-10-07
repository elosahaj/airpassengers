import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

#dane
df = pd.read_csv("AirPassengers.csv")

df["Month"] = pd.to_datetime(df["Month"])

#print(df.columns)

df = df.set_index("Month")
#print(df.head())


### ### WSTĘPNA WIZUALIZACJA ### ### 

# plt.figure(figsize=(12,6))
# plt.plot(df["#Passengers"])
# plt.title("Air passengers")
# plt.ylabel("Passengers")
# plt.grid(True)
# plt.show()

### ### SPLIT ### ###

train = df.loc[:'1958-12-01']
test = df.loc['1959-01-01':]

# print(len(train))
# print(len(test))

# print(train.tail())
# print(test.head())


# print split
# plt.figure(figsize=(12,6))

# plt.plot(train.index, train["#Passengers"], label="Train")
# plt.plot(test.index, test["#Passengers"], label="Test")

# plt.legend()
# plt.grid(True)
# plt.show()



### ### NAIVE FORECAST ### ###

last_value = train["#Passengers"].iloc[-1]
#print(last_value)
naive_forecast = [last_value]*len(test)


# plt.figure(figsize=(12,6))
# plt.plot(train.index, train["#Passengers"], label="Train")
# plt.plot(test.index, test["#Passengers"], label="Actual")
# plt.plot(test.index, naive_forecast, label="Naive forecast")

# plt.legend()
# plt.grid(True)
# plt.show()


### ### METRYKA ### ###

mae = mean_absolute_error(test["#Passengers"], naive_forecast) 

rmse = np.sqrt(mean_squared_error(test["#Passengers"], naive_forecast))

mape = mean_absolute_percentage_error(test["#Passengers"], naive_forecast)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")