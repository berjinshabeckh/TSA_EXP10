# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

## Name:H.Berjin Shabeck
## Reg no:212222240018
## Date:30/10/2024

### AIM:
To implement SARIMA model using python for student score data.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

data = pd.read_csv('score.csv')

# Step 2: Plot the Time Series
plt.plot(data['Scores'])
plt.title("Scores over Time")
plt.xlabel("Index")
plt.ylabel("Scores")
plt.show()

# Step 3: Stationarity Check (ADF Test)
result = adfuller(data['Scores'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] > 0.05:
    print("Series is non-stationary")
else:
    print("Series is stationary")

# Step 4: Plot ACF and PACF
plot_acf(data['Scores'])
plt.show()
plot_pacf(data['Scores'])
plt.show()

# Step 5: Fit SARIMA Model
# Use (p, d, q) x (P, D, Q, s) configuration.
# For example, let's assume initial parameters based on visual inspection
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12  # Assuming monthly seasonality; adjust `s` as needed

model = SARIMAX(data['Scores'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_result = model.fit(disp=False)
print(sarima_result.summary())

# Step 6: Forecasting
forecast_steps = 5
forecast = sarima_result.forecast(steps=forecast_steps)
print("Forecasted Values:", forecast)

# Step 7: Evaluate Model Predictions
# Split data into train and test for evaluation (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Scores'][:train_size], data['Scores'][train_size:]
model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
fitted_model = model.fit(disp=False)

# Forecast the test period
predictions = fitted_model.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print("Mean Squared Error:", mse)

# Optionally, plot the predictions vs actual values
plt.plot(test.values, label='Actual')
plt.plot(predictions, label='Predicted', color='red')
plt.legend()
plt.title("SARIMA Predictions vs Actual Scores")
plt.show()

```

### OUTPUT:
![download](https://github.com/user-attachments/assets/046222a3-e2bb-40cc-8b55-554c135983f6)

![download](https://github.com/user-attachments/assets/890d15a4-02ff-40e2-9b47-ba84f38782f6)

![download](https://github.com/user-attachments/assets/356e8a33-b256-4894-b25d-c7ffe3497858)

![download](https://github.com/user-attachments/assets/59a848ef-fa92-4e60-a0ff-546ec7465325)

![image](https://github.com/user-attachments/assets/53cac200-ce50-40e0-8ec2-68b8f29056bd)

### RESULT:
Thus the program run successfully based on the SARIMA model.
