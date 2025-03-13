#%%%
'''Importing libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%
'''Loading the dataset'''
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/county_unemployment.csv')

#%%
zipcode = 'geoId/06001'
data = df[df['place'] == 'geoId/01001'].iloc[:, 1:]  # Extract time-series columns

# Transpose and reset index for easier handling
data = data.T.reset_index()
data.columns = ['Date', 'Value']
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

#%%
plt.figure(figsize=(12, 6))
plt.plot(data['Value'], label=f'Zip Code: {zipcode}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'Time Series Data for Zip Code {zipcode}')
plt.legend()
plt.show()

# %%


train_size = len(data) - 3  # Use all but the last 3 months for training
train = data['Value'][:train_size]
test = data['Value'][train_size:]

# Build SARIMA model with adjusted parameters
order = (5, 1, 2)           # Adjust p, d, q based on PACF/ACF
seasonal_order = (1, 1, 1, 12)  # Adjust P, D, Q based on seasonality

sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
sarima_results = sarima_model.fit()

# Forecast the next 3 months
forecasted_values = sarima_results.forecast(steps=3)

# Evaluate model performance
mae = mean_absolute_error(test.values, forecasted_values.values)
rmse = np.sqrt(mean_squared_error(test.values, forecasted_values.values))
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Plot actual vs forecasted values for the last 3 months
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual', marker='o')
plt.plot(test.index, forecasted_values.values, label='Forecasted', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'SARIMA Model Forecast for Last 3 Months for Zip Code {zipcode}')
plt.legend()
plt.show()
# %%
