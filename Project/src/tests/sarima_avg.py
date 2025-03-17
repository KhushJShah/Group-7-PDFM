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
geoids = df['place'].unique()

# Initialize lists to store metrics for all geoIds
mae_list = []
rmse_list = []

#%%
for geoid in geoids:
    print(f"Processing geoId: {geoid}")
    
    # Filter data for the current geoId
    data = df[df['place'] == geoid].iloc[:, 1:]
    
    # Transpose and reset index for easier handling
    data = data.T.reset_index()
    data.columns = ['Date', 'Value']
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Split data into train and test sets (last 3 months for testing)
    train_size = len(data) - 3  # Use all but the last 3 months for training
    train = data['Value'][:train_size]
    test = data['Value'][train_size:]
    
    # Build SARIMA model with adjusted parameters
    order = (5, 1, 2)           # Adjust p, d, q based on PACF/ACF
    seasonal_order = (1, 1, 1, 12)  # Adjust P, D, Q based on seasonality
    
    try:
        sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        sarima_results = sarima_model.fit(disp=False)
        
        # Forecast the next 3 months
        forecasted_values = sarima_results.forecast(steps=3)
        
        # Evaluate model performance
        mae = mean_absolute_error(test.values, forecasted_values.values)
        rmse = np.sqrt(mean_squared_error(test.values, forecasted_values.values))
        
        # Store metrics
        mae_list.append(mae)
        rmse_list.append(rmse)
        
        print(f"GeoID: {geoid}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    except Exception as e:
        print(f"Error processing geoId {geoid}: {e}")
        continue

# %%
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)

print(f"\nAverage MAE across all geoIds: {average_mae:.2f}")
print(f"Average RMSE across all geoIds: {average_rmse:.2f}")
# %%
