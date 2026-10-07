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

print(df.columns)

df = df.set_index("Month")
print(df.head())


#wizualizacja
plt.figure(figsize=(12,6))
plt.plot(df["#Passengers"])
plt.title("Air passengers")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

