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

#%%
# Calculate rolling mean and variance
window_size = 12  # Example window size
data['Rolling_Mean'] = data['Value'].rolling(window_size).mean()
data['Rolling_Variance'] = data['Value'].rolling(window_size).var()

# Plot rolling mean and variance
plt.figure(figsize=(12, 6))
plt.plot(data['Rolling_Mean'], label='Rolling Mean')
plt.plot(data['Rolling_Variance'], label='Rolling Variance')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'Rolling Mean and Variance for Zip Code {zipcode}')
plt.legend()
plt.show()

#%%

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

#%%
# Perform seasonal differencing (e.g., every 12 months)
data['Value_diff'] = data['Value'] - data['Value'].shift(1)

# Drop NaN values after differencing
data_diff = data['Value_diff'].dropna()
data_diff_df = pd.DataFrame(data_diff).reset_index(drop=True)
#%%
# Calculate rolling mean and variance again
window_size = 12  # Example window size
data_diff_df['Rolling_Mean'] = data_diff_df.iloc[:, 0].rolling(window_size).mean()
data_diff_df['Rolling_Variance'] = data_diff_df.iloc[:, 0].rolling(window_size).var()

# Plot rolling mean and variance after differencing
plt.figure(figsize=(12, 6))
plt.plot(data_diff_df['Rolling_Mean'], label='Rolling Mean')
plt.plot(data_diff_df['Rolling_Variance'], label='Rolling Variance')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'Rolling Mean and Variance After Differencing for Zip Code {zipcode}')
plt.legend()
plt.show()


#%%
print(data_diff.head())  # Check the first few rows
print(type(data_diff))   # Check the type of data_diff

#%%
data_diff_numeric = pd.to_numeric(data_diff, errors='coerce')

# Check for any NaN values introduced during conversion
print(data_diff_numeric.isnull().sum())

# Drop any NaN values if present
data_diff_numeric = data_diff_numeric.dropna()
#%%
# Ensure data_diff is a 1D numeric array
data_diff_array = data_diff_numeric.values

# Apply ADF test
result = adfuller(data_diff_array)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] > 0.05:
    print("Time series is not stationary after differencing.")
else:
    print("Time series is stationary after differencing.")


#%%
# Plot ACF and PACF for differenced data
plt.figure(figsize=(12, 6))
plot_acf(data_diff_numeric, lags=50)
plt.title('Autocorrelation Function (ACF) of Differenced Data')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data_diff_numeric, lags=50)
plt.title('Partial Autocorrelation Function (PACF) of Differenced Data')
plt.show()


# %%
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# Split data into train and test sets
forecast_steps=3
train_size = len(data) - forecast_steps # Use all but the last 3 months for training
train = data['Value'][:train_size]
test = data['Value'][train_size:]

# Build and train AR model
lag_order = 12  # Choose based on PACF plot or domain knowledge
ar_model = AutoReg(train, lags=lag_order).fit()

# Forecast the next 3 months
forecasted_values = ar_model.forecast(steps=forecast_steps)

# Evaluate model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test, forecasted_values)
rmse = np.sqrt(mean_squared_error(test, forecasted_values))
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Plot actual vs forecasted values for the last 3 months
plt.figure(figsize=(12, 6))
plt.plot(range(len(train), len(train) + len(test)), test.values, label='Actual', marker='o')
plt.plot(range(len(train), len(train) + len(test)), forecasted_values, label='Forecasted', linestyle='--', marker='x')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'AR Model Forecast for Last 3 Months for Zip Code {zipcode}')
plt.legend()
plt.show()




# %%
