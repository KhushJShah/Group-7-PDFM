#%%%
'''Importing libraries'''
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
#%%
'''Loading the dataset'''
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/county_unemployment.csv')
def load_data(file_path):
    return pd.read_csv(file_path)
#%%
def prepare_data(df):
    results = []
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        type = zipcode_df['Type'].iloc[0]
        
        # Extract time series data and convert index to datetime
        time_series_columns = [col for col in zipcode_df.columns if col.startswith('19') or col.startswith('20')]
        ts_data = zipcode_df[time_series_columns].mean(axis=0)
        ts_data.index = pd.to_datetime(ts_data.index)  # Convert index to datetime
        
        # Split data into training and testing sets
        train_data = ts_data[:-6]
        test_data = ts_data[-6:]
        
        # Fit SARIMA model
        if type == 'Non-Stationary':
            if county in ['Eureka', 'Lincoln']:
                model = SARIMAX(train_data, order=(2,0,1), seasonal_order=(2,0,1,12))
            else:
                model = SARIMAX(train_data, order=(1,0,1), seasonal_order=(1,0,1,12))
        else:
            model = SARIMAX(train_data, order=(1,0,0), seasonal_order=(0,0,0,12))
            
        model_fit = model.fit(disp=False)
        
        # Generate forecasts and get fitted values
        forecast = model_fit.get_forecast(steps=6).predicted_mean
        fitted = model_fit.get_prediction().predicted_mean  # In-sample predictions
        
        # Calculate errors
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot training data
        ax1.plot(train_data.index, train_data, label='Training Data', color='blue')
        ax1.plot(train_data.index, fitted, label='Fitted Values', linestyle='--', color='red')
        ax1.set_title(f'Training Data and Fitted Values for {county}, {state} (Zipcode: {zipcode})')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Unemployment Rate')
        ax1.legend()
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot test data
        ax2.plot(test_data.index, test_data, label='Test Data', color='green')
        ax2.plot(test_data.index, forecast, label='Forecast', linestyle='--', color='red')
        ax2.set_title(f'Test Data and Forecast for {county}, {state} (Zipcode: {zipcode})')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Unemployment Rate')
        ax2.legend()
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot to file
        plt.savefig(f'sarima_forecast_{zipcode}.png', dpi=300)
        plt.close()
        
        # Append results
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'Type': type
        })
    
    return results

#%%
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

# Main function
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'  # Replace with your dataset path
    output_path = 'sarima_results.csv'
    
    df = load_data(file_path)
    results = prepare_data(df)
    save_results(results, output_path)

if __name__ == "__main__":
    main()

# %%
