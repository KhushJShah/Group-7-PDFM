'''This file explores the possbility of auto-regressor model for the unemployment dataset.'''

#%%%
'''Importing libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

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
result = adfuller(data['Value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] > 0.05:
    print("Time series is not stationary. Differencing is required.")
    data['Value_diff'] = data['Value'].diff().dropna()
else:
    print("Time series is stationary.")

# Plot ACF and PACF to determine lag order
plot_acf(data['Value'].dropna(), lags=50)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Value'].dropna(), lags=50)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
# %%
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Build and train AR model
lag_order = 6  # Choose based on PACF plot
ar_model = AutoReg(train['Value'], lags=lag_order).fit()

# Make predictions on test set
predictions = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
# %%
mae = mean_absolute_error(test['Value'], predictions)
rmse = np.sqrt(mean_squared_error(test['Value'], predictions))
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.title(f'AR Model Predictions for Zip Code {zipcode}')
plt.show()
# %%
