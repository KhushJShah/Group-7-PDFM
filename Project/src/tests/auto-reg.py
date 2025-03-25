#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
# %%
def load_data(file_path):
    return pd.read_csv(file_path)
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')


#%%
def prepare_data(df):
    results = []
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        ts_type = zipcode_df['Type'].iloc[0]
        
        # Extract time series data
        ts_cols = [col for col in zipcode_df.columns if col.startswith(('19', '20'))]
        ts_data = zipcode_df[ts_cols].mean(axis=0)
        ts_data.index = pd.to_datetime(ts_data.index)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        if ts_type == 'Non-Stationary':
            # --- Differencing and Model Setup ---
            ts_diff = ts_data.diff(12).dropna()
            train_diff = ts_diff[:-6]
            
            # Determine lag
            ar_lag = 2 if county in ['Eureka', 'Lincoln'] else 1
            
            # --- Model Training ---
            model = AutoReg(train_diff, lags=ar_lag)
            model_fit = model.fit()
            fitted_values = model_fit.predict()
            
            # --- Index Alignment for Training Data ---
            start_idx = 12 + ar_lag  # 12 for differencing + AR lag
            end_idx = start_idx + len(fitted_values)
            train_indices = ts_data.index[start_idx:end_idx]  # Correct alignment
            
            # --- Plot Training Data ---
            ax1.plot(ts_data.iloc[:len(ts_data)-6], label='Original Training Data', color='blue')
            ax1.plot(train_indices, fitted_values + ts_data.iloc[start_idx-12:end_idx-12].values, 
                    label='Fitted Values', linestyle='--', color='orange')
            ax1.set_title(f'Training Data - {county}, {state}')

            # --- Forecast Handling ---
            forecast_diff = model_fit.forecast(steps=6)
            forecast_original = np.cumsum(forecast_diff) + ts_data.iloc[-18]
            
            # --- Plot Test Data ---
            ax2.plot(ts_data.index[-6:], ts_data[-6:], label='Actual Test Data', color='green')
            ax2.plot(ts_data.index[-6:], forecast_original, label='Forecast', linestyle='--', color='red')
            ax2.set_title(f'Test Forecast - {county}, {state}')

            # --- Error Calculation ---
            mae = mean_absolute_error(ts_data[-6:], forecast_original)
            rmse = np.sqrt(mean_squared_error(ts_data[-6:], forecast_original))

        else:
            # STATIONARY DATA HANDLING
            train_data = ts_data[:-6]
            test_data = ts_data[-6:]
            
            # Fit model with lag=1
            model = AutoReg(train_data, lags=1)
            model_fit = model.fit()
            
            # Get predictions using model's built-in indices
            fitted_values = model_fit.predict()
            forecast = model_fit.forecast(steps=6)
            
            # Plot training results using model's indices
            ax1.plot(train_data, label='Original Training Data', color='blue')
            ax1.plot(fitted_values.index, fitted_values,  # Critical fix
                    label='Fitted Values', 
                    linestyle='--', 
                    color='orange')
            ax1.set_title(f'Training Data - {county}, {state}')
            
            # Plot test results
            ax2.plot(test_data.index, test_data, label='Actual Test Data', color='green')
            ax2.plot(test_data.index, forecast, 
                    label='Forecast', 
                    linestyle='--', 
                    color='red')
            ax2.set_title(f'Test Forecast - {county}, {state}')
            
            # Calculate errors
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))

        # Common plot elements
        for ax in [ax1, ax2]:
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f'ar_results_{zipcode}.png', dpi=300)
        plt.close()
        
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'Type': ts_type
        })
    
    return results


# %%
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

# Main function
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'  # Replace with your dataset path
    output_path = 'results.csv'
    
    df = load_data(file_path)
    results = prepare_data(df)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
# %%
