#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%%
# Function to perform grid search for AutoRegressor hyperparameters
def autoreg_grid_search(train_data, test_data, orders):
    best_score_rmse, best_cfg = float("inf"), None
    best_score_mae, best_cfg_mae = float("inf"), None
    results = []
    for order in orders:
        try:
            # Fit AutoRegressor model
            model = AutoReg(train_data, lags=order)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate RMSE and MAE
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mae = mean_absolute_error(test_data, forecast)
            results.append((order, rmse, mae))
            
            # Update best score and configuration for RMSE
            if rmse < best_score_rmse:
                best_score_rmse, best_cfg_rmse = rmse, order
            
            # Update best score and configuration for MAE
            if mae < best_score_mae:
                best_score_mae, best_cfg_mae = mae, order
        except:
            continue
    return best_cfg_rmse, best_score_rmse, best_cfg_mae, best_score_mae, results

#%%
# Load your dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare data for modeling
def prepare_data(df):
    results = []
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        
        # Extract time series data
        ts_data = zipcode_df[[col for col in zipcode_df.columns if col.startswith('19') or col.startswith('20') or col.startswith('199')]].mean(axis=0)
        
        # Split data into training and testing sets
        train_data = ts_data[:-6]
        test_data = ts_data[-6:]
        
        # Define hyperparameter range
        orders = range(1, 15)  # Orders from 1 to 14
        
        # Perform grid search for AutoRegressor hyperparameters
        best_cfg_rmse, best_score_rmse, best_cfg_mae, best_score_mae, results = autoreg_grid_search(train_data, test_data, orders)
        
        # Append results
        results_dict = {
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'Best RMSE Order': best_cfg_rmse,
            'Best RMSE Score': round(best_score_rmse, 2),
            'Best MAE Order': best_cfg_mae,
            'Best MAE Score': round(best_score_mae, 2),
        }
        results.append(results_dict)
    
    return results

#%%
# Save results to CSV
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

# Main function
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'  # Replace with your dataset path
    output_path = 'autoreg_grid_search_results.csv'
    
    df = load_data(file_path)
    results = prepare_data(df)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
