#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product

#%%

# Function to perform grid search for SARIMA hyperparameters
def sarima_grid_search(train_data, test_data, p_values, d_values, q_values, P_values, D_values, Q_values, m):
    best_score_rmse, best_cfg = float("inf"), None
    best_score_mae, best_cfg_mae = float("inf"), None
    results = []
    for (p, d, q, P, D, Q) in product(p_values, d_values, q_values, P_values, D_values, Q_values):
        try:
            # Fit SARIMA model
            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)
            
            # Forecast
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate RMSE and MAE
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mae = mean_absolute_error(test_data, forecast)
            results.append(((p, d, q, P, D, Q, m), rmse, mae))
            
            # Update best score and configuration for RMSE
            if rmse < best_score_rmse:
                best_score_rmse, best_cfg_rmse = rmse, (p, d, q, P, D, Q, m)
            
            # Update best score and configuration for MAE
            if mae < best_score_mae:
                best_score_mae, best_cfg_mae = mae, (p, d, q, P, D, Q, m)
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
        type = zipcode_df['Type'].iloc[0]
        
        # Extract time series data
        ts_data = zipcode_df[[col for col in zipcode_df.columns if col.startswith('19') or col.startswith('20') or col.startswith('199')]].mean(axis=0)
        
        # Split data into training and testing sets
        train_data = ts_data[:-6]
        test_data = ts_data[-6:]
        
        # Define hyperparameter ranges
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        m = 12
        
        # Perform grid search for SARIMA hyperparameters
        best_cfg_rmse, best_score_rmse, best_cfg_mae, best_score_mae, results = sarima_grid_search(train_data, test_data, p_values, d_values, q_values, P_values, D_values, Q_values, m)
        
        # Append results
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'Best RMSE Config': best_cfg_rmse,
            'Best RMSE Score': round(best_score_rmse, 2),
            'Best MAE Config': best_cfg_mae,
            'Best MAE Score': round(best_score_mae, 2),
            'Type': type
        })
    
    return results

#%%
# Save results to CSV
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

# Main function
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'  # Replace with your dataset path
    output_path = 'sarima_grid_search_results.csv'
    
    df = load_data(file_path)
    results = prepare_data(df)
    save_results(results, output_path)

if __name__ == "__main__":
    main()

# %%
